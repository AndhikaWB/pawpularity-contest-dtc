import mlflow
import dotenv
import warnings
from pathlib import Path
from datetime import datetime

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from _s3.lakefs import get_exact_commit, replace_branch
from _pydantic.common import LakeFSConf, MLFlowConf, S3Conf
from _pydantic.train_test import (
    TestParams, TestResult, TrainParams, TrainTags, MLFlowModel
)

from _ml.tester import Tester
from _ml.utils import MetricTester
from training import run as training_run


def get_data_commit_id(
    data_source_repo: str, lfs_cfg: LakeFSConf, check_date: bool = True
) -> str | tuple[str, str]:
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
            alias = mlf_model.best_version_alias
        ).version
    except mlflow.exceptions.RestException as err:
        # If there's no version under that alias
        if err.error_code == 'INVALID_PARAMETER_VALUE':
            version = None
        else:
            raise err

    print(
        f'Alias "{alias}" is tied to model version "{version}"' if version
        else f'No model version found under the alias "{alias}"'
    )

    return version


def evaluate_model(
    version: str, params: TestParams, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf,
    s3_cfg: S3Conf
) -> TestResult:
    if not (params.csv_dir or params.img_dir or params.data_commit_id):
        raise ValueError('Data sources or commit id can\'t be empty')

    print(
        f'Testing model version "{version}" with data from commit id '
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
    result = tester.run_evaluation(df, params, mlf_cfg)

    return result


def set_best_model_version(
    result: TestResult, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf
) -> str:
    mlf_cfg.expose_auth_to_env()
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        mlf_model.model_registry_name,
        mlf_model.best_version_alias,
        result.model_version
    )

    tags = {
        # NOTE: Check the training script for possible tag conflicts
        'model_marked_best_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_data_commit_id': result.data_commit_id
    }

    for key, val in tags.items():
        client.set_model_version_tag(
            mlf_model.model_registry_name,
            version = result.model_version,
            key = key,
            value = val
        )

    return result.model_version


def run(
    data_source_repo: str, data_source_creds: LakeFSConf, train_params: TrainParams,
    train_tags: TrainTags, model_registry: MLFlowModel, mlflow_creds: MLFlowConf
):
    # Get the latest data commit id from lakeFS repo
    commit_id = get_data_commit_id(data_source_repo, data_source_creds)
    # Replace the branch name with the exact commit id for preciseness
    data_source_repo = replace_branch(data_source_repo, commit_id)

    # Add the required params for training
    train_params.data_commit_id = commit_id
    train_params.csv_dir = data_source_repo
    train_params.img_dir = data_source_repo + '/images'
    # Generate the test params based on the training params
    test_params = train_params.to_test()

    # Get the current best model version, or return None if not exist
    cur_model = get_best_model_version(model_registry, mlflow_creds)
    # Evaluate the model (if exist) using the current data
    cur_result = evaluate_model(
        cur_model, test_params, model_registry, mlflow_creds,
        data_source_creds.as_s3()
    ) if cur_model else None

    metric = MetricTester(
        threshold = test_params.metric_threshold,
        min_is_better = test_params.metric_min
    )

    # Train a new model if we don't have any registered best model yet
    # Or if the test score is considered unsafe compared to the threshold
    if not (cur_model or metric.is_safe(cur_result.score)):
        new_model = training_run(
            data_source_repo, data_source_creds,
            train_params, train_tags,
            model_registry, mlflow_creds
        )

        new_result = evaluate_model(
            new_model, test_params, model_registry, mlflow_creds,
            data_source_creds.as_s3()
        )

        print(
            # Report credentials
            # Database credentials?
        )

        if metric.is_safe(new_result.score):
            if cur_model and metric.better_than(cur_result.score, new_result.score):
                # If the current model is better than the new model
                # Generate report but no need to change the best model
                set_best_model_version(cur_result, model_registry, mlflow_creds)
                print(
                    # Current model
                    # Current test run (to get input dataframe)
                    # Current data commit id (to check previous test run as reference)
                    # MLFlow model registry
                    # MLFlow credentials
                )
            else:
                # If there's no current model, or the new model is better
                # Generate report and also change the best model
                set_best_model_version(new_result, model_registry, mlflow_creds)
                pass
        else:
            # If both model test results are still bad
            # Generate report and choose the better model for now
            # The alert is managed by the dashboard, not by this code
            if metric.better_than(cur_result.score, new_result.score):
                set_best_model_version(cur_result, model_registry, mlflow_creds)
            else:
                set_best_model_version(new_result, model_registry, mlflow_creds)
            
    else:
        # If the current model test result is still good
        # Generate report and no need to train a new model
        set_best_model_version(cur_result, model_registry, mlflow_creds)


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

        # No need for factory if it doesn't read environment variable
        train_params: TrainParams = TrainParams()
        train_tags: TrainTags = TrainTags()

        model_registry: MLFlowModel = Field(default_factory = MLFlowModel)
        mlflow_creds: MLFlowConf = Field(default_factory = MLFlowConf)

        @field_validator('train_tags', mode = 'after')
        @classmethod
        def check_default_tags(cls, value: TrainTags):
            if value == TrainTags():
                warnings.warn(
                    f'Using default author name ({value.author}) and other tags as '
                    'MLFLow run tags. You may want to review/change this later'
                )

            return value

    args = ParseArgs()

    run(
        args.data_source_repo, args.data_source_creds,
        args.train_params, args.train_tags,
        args.model_registry, args.mlflow_creds
    )