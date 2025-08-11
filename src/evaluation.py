import mlflow
import dotenv
import polars as pl
from pathlib import Path
from datetime import datetime

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from _s3.lakefs import get_exact_commit, replace_branch
from _pydantic.common import LakeFSConf, MLFlowConf, S3Conf
from _pydantic.train_test import TrainParams, TestParams, TestSummary, MLFlowModel
from _pydantic.train_test import ModelRegisTags, ModelBestTags

from _ml.tester import Tester
from _ml.utils import MetricTester
from _monitoring.reporter import Reporter
from _pydantic.report import ReportConf
from training import run as training_run

from prefect import flow, task


@task
def get_data_commit_id(
    data_source_repo: str, lfs_cfg: LakeFSConf, check_date: bool = False
) -> str:
    commit = get_exact_commit(data_source_repo, lfs_cfg, return_id = False)
    if not commit:
        raise RuntimeError(f'Can\'t get latest commit from "{data_source_repo}"')

    # Creation date is an epoch with no timezone (naive/local time)
    # We only need the date, but we can also extract the time if needed
    commit_date = datetime.fromtimestamp(commit.creation_date)
    days_since_commit = (datetime.now() - commit_date).days
    if check_date and days_since_commit >= 28:
        raise RuntimeError(f'Last commit was {days_since_commit} days ago')

    print(f'Using commit "{commit.id[:8]}" from "{data_source_repo}"')
    print(f'Commit was pushed {days_since_commit} days ago ({commit_date})')

    return commit.id

@task
def get_best_model_version(mlf_model: MLFlowModel, mlf_cfg: MLFlowConf) -> str | None:
    mlf_cfg.expose_auth_to_env()
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    client = mlflow.MlflowClient()

    # If found, this model will be tested with the latest data later
    print(f'Getting the best model version from "{mlf_model.model_registry_name}"')
    alias = mlf_model.best_version_alias

    try:
        # Try getting the best model using a version alias
        version = client.get_model_version_by_alias(
            mlf_model.model_registry_name,
            alias = alias
        ).version
    except mlflow.exceptions.RestException as err:
        # If there's no version under that alias
        if err.error_code == 'INVALID_PARAMETER_VALUE':
            version = None
        else:
            raise err

    print(
        f'Alias "{alias}" is tied to model version "{version}"' if version
        else f'No model version under the alias "{alias}" yet'
    )

    return version

@task
def evaluate_model(
    version: str, params: TestParams, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf,
    s3_cfg: S3Conf
) -> TestSummary:
    if not (params.csv_dir or params.img_dir or params.data_commit_id):
        raise ValueError('Data sources or commit id can\'t be empty')

    print(
        f'Evaluating model version "{version}" with data from commit id '
        f'"{params.data_commit_id[:8]}"'
    )
    
    mlf_cfg.expose_auth_to_env()
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    client = mlflow.MlflowClient()

    model_info = client.get_model_version(mlf_model.model_registry_name, version)
    # BUG: MLFlow may return empty model id, we may want to fix it manually
    tester = Tester(
        model_name = model_info.name,
        model_version = model_info.version,
        model_uri = model_info.source,
        model_id = model_info.model_id
    )

    # The data source will be used here to make the prediction
    df = tester.predict(params, mlf_cfg, s3_cfg)
    # The model id (if not None) will be tied here with the evaluation
    summary = tester.run_evaluation(df, params, mlf_cfg)

    print(f'Generated evaluation run id "{summary.run_id}"')
    return summary

@task
def set_best_model_version(
    summary: TestSummary, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf
) -> str:
    mlf_cfg.expose_auth_to_env()
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    client = mlflow.MlflowClient()

    print('Setting version alias for the best model')

    client.set_registered_model_alias(
        mlf_model.model_registry_name,
        mlf_model.best_version_alias,
        summary.model_version
    )

    # Tags to add to the best model
    tags = ModelBestTags(
        model_marked_best_at = ModelBestTags.datetime_now(),
        test_data_commit_id = summary.data_commit_id
    ).model_dump()

    for key, val in tags.items():
        client.set_model_version_tag(
            mlf_model.model_registry_name,
            version = summary.model_version,
            key = key,
            value = val
        )

    print(
        f'Marked model version "{summary.model_version}" with '
        f'alias "{mlf_model.best_version_alias}"'
    )

    return summary.model_version

@task
def generate_report(
    ref_commit_id: str | None, summary: TestSummary, mlf_cfg: MLFlowConf,
    report_cfg: ReportConf, table_name: str = 'monitoring'
) -> bool:
    if not ref_commit_id:
        print('No reference data yet, no report generated')
        return False
    
    print('Generating drift monitoring report')
    print(f'* Current model version = {summary.model_version}')
    print(f'* Current commit id = {summary.data_commit_id}')
    print(f'* Reference commit id = {ref_commit_id}')

    # Pass info about the current data and evaluation result
    cur_df = Tester.load_prediction(summary.run_id, mlf_cfg, error_ok = False)
    reporter = Reporter(summary, cur_df)

    # Also get the reference evaluation data (for drift comparison)
    # This assumes that it uses the same metric as the current data
    ref_run_id = Tester.search_evaluation(ref_commit_id, summary, mlf_cfg)
    ref_df = Tester.load_prediction(ref_run_id, mlf_cfg, error_ok = True)

    if isinstance(ref_df, pl.DataFrame):
        # Generate the drift report and write it to a database
        report_df = reporter.generate_report(ref_run_id, ref_commit_id, ref_df, summary)
        reporter.write_report_to_db(report_df, table_name, report_cfg)
        print(f'Written report to database table "{table_name}"')
        return True

    print('Can\'t find reference data, no report generated')
    return False

@flow
def run(
    data_source_repo: str, data_source_creds: LakeFSConf, train_params: TrainParams,
    regis_tags: ModelRegisTags, model_registry: MLFlowModel, mlflow_creds: MLFlowConf,
    report_creds: ReportConf
):
    # Get the latest data commit id from lakeFS repo
    commit_id = get_data_commit_id(data_source_repo, data_source_creds)
    # Replace the branch name with the exact commit id for preciseness
    data_source_repo = replace_branch(data_source_repo, commit_id)
    # Also get the previous commit id for drift monitoring purpose later
    prev_commit_id = get_exact_commit(data_source_repo, data_source_creds, '~1')

    # Add the required params for training
    train_params.data_commit_id = commit_id
    train_params.csv_dir = data_source_repo
    train_params.img_dir = data_source_repo + '/images'
    # Generate the test params based on the training params
    test_params = train_params.to_test()

    # Get the current best model version, or return None if not exist
    cur_model = get_best_model_version(model_registry, mlflow_creds)
    # Evaluate the model (if exist) using the current data
    cur_summary = evaluate_model(
        cur_model, test_params, model_registry, mlflow_creds,
        data_source_creds.as_s3()
    ) if cur_model else None

    metric = MetricTester(
        threshold = test_params.metric_threshold,
        min_is_better = test_params.metric_min
    )

    # Train a new model if we don't have any registered best model yet
    # Or if the test score is considered unsafe compared to the threshold
    if not cur_model or not metric.is_safe(cur_summary.score):
        new_model = training_run(
            data_source_repo, data_source_creds,
            train_params, regis_tags,
            model_registry, mlflow_creds
        )

        new_summary = evaluate_model(
            new_model, test_params, model_registry, mlflow_creds,
            data_source_creds.as_s3()
        )

        if metric.is_safe(new_summary.score):
            if cur_model and metric.is_better(cur_summary.score, new_summary.score):
                # If the current model is better than the new model
                # Generate report but no need to change the best model
                set_best_model_version(cur_summary, model_registry, mlflow_creds)
                generate_report(prev_commit_id, cur_summary, mlflow_creds, report_creds)
            else:
                # If there's no current model, or the new model is better
                # Generate report and also change the best model
                set_best_model_version(new_summary, model_registry, mlflow_creds)
                generate_report(prev_commit_id, new_summary, mlflow_creds, report_creds)
        else:
            # If both current and new model test summaries are still bad
            # Generate report and choose the better model for now
            # The alert must be managed separately by the dashboard
            if not cur_model or metric.is_better(new_summary.score, cur_summary.score):
                set_best_model_version(new_summary, model_registry, mlflow_creds)
                generate_report(prev_commit_id, new_summary, mlflow_creds, report_creds)
            else:
                set_best_model_version(cur_summary, model_registry, mlflow_creds)
                generate_report(prev_commit_id, cur_summary, mlflow_creds, report_creds)
    else:
        # If the current model test summary is still good
        # Generate report and no need to train a new model
        set_best_model_version(cur_summary, model_registry, mlflow_creds)
        generate_report(prev_commit_id, cur_summary, mlflow_creds, report_creds)


if __name__ == '__main__':
    dotenv.load_dotenv(
        '.env.prod' if Path('.env.prod').exists()
        else '.env.dev'
    )

    class ParseArgs(BaseSettings):
        """Train a model using data sourced from S3."""

        model_config = SettingsConfigDict(
            cli_parse_args = True,
            cli_kebab_case = True,
            validate_assignment = True
        )

        data_source_repo: str = Field(alias = 'TRAIN_DATA_SOURCE')
        data_source_creds: LakeFSConf = Field(default_factory = LakeFSConf)

        train_params: TrainParams = Field(default_factory = TrainParams)
        regis_tags: ModelRegisTags = Field(default_factory = ModelRegisTags)

        model_registry: MLFlowModel = Field(default_factory = MLFlowModel)
        mlflow_creds: MLFlowConf = Field(default_factory = MLFlowConf)

        report_creds: ReportConf = Field(default_factory = ReportConf)

    args = ParseArgs()

    run(
        args.data_source_repo, args.data_source_creds,
        args.train_params, args.regis_tags,
        args.model_registry, args.mlflow_creds,
        args.report_creds
    )