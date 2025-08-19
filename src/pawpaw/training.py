import dotenv
import tempfile
from pathlib import Path
from pawpaw import logger
from copy import deepcopy

import mlflow
from pawpaw.ml.trainer import Trainer
from pawpaw.ml.model import PawDataLoader, PawModel

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pawpaw.pydantic_.common import S3Conf, LakeFSConf, MLFlowConf
from pawpaw.pydantic_.train_test import TrainParams, TrainSummary, MLFlowModel
from pawpaw.pydantic_.train_test import ModelRegisTags

from pawpaw.s3.lakefs import replace_branch, get_exact_commit


def get_data_commit_id(data_source_repo: str, lfs_cfg: LakeFSConf) -> str:
    logger.debug(f'Getting latest commit from "{data_source_repo}"')

    commit_id = get_exact_commit(data_source_repo, lfs_cfg, return_id = True)
    if not commit_id:
        raise RuntimeError(f'Can\'t get latest commit info')

    logger.info(f'Using commit "{commit_id[:8]}" from "{data_source_repo}"')
    return commit_id


def model_training(
    params: TrainParams, s3_cfg: S3Conf, mlf_cfg: MLFlowConf
) -> TrainSummary:
    if not (params.csv_dir or params.img_dir or params.data_commit_id):
        raise ValueError('Data source or commit id can\'t be empty')

    # Use the same cache dir for train and val data
    # Only do this if you're certain there's no file conflict
    cache_dir = tempfile.TemporaryDirectory()

    train_loader = PawDataLoader(
        params.csv_dir + '/train.csv',
        img_dir = params.img_dir,
        is_train_data = True,
        batch_size = params.batch_size,
        img_size = params.img_size,
        s3_cfg = s3_cfg,
        cache_dir = cache_dir.name
    )

    val_loader = PawDataLoader(
        params.csv_dir + '/val.csv',
        img_dir = params.img_dir,
        is_train_data = False,
        batch_size = params.batch_size,
        img_size = params.img_size,
        s3_cfg = s3_cfg,
        cache_dir = cache_dir.name
    )

    model = PawModel()
    trainer = Trainer(model, train_loader, val_loader)
    logger.debug(f'Preparing to train "{model.__class__.__name__}" model')

    # Don't modify the original params directly
    params = deepcopy(params)
    # Setup optimizer, early stop callback, and other things
    # This will add optimizer name, etc. to the returned params
    params = trainer.prep_training(params)
    # Log the params to MLFlow and train the model
    run_id = trainer.start_training(params, mlf_cfg)

    # Get the best model only and delete other models from this run
    # The summary will contain run id, best model URI, and metric info
    logger.debug(f'Keeping only the best model from run id "{run_id}"')
    summary = trainer.get_best_model(run_id, params, mlf_cfg, delete_others = True)
    # Clean back the cache dir
    cache_dir.cleanup()

    return summary


def register_model(
    summary: TrainSummary, regis_tags: ModelRegisTags, mlf_model: MLFlowModel,
    mlf_cfg: MLFlowConf, finished_only: bool = True
) -> str:
    if not (summary.run_id or summary.model_uri):
        raise ValueError('Run id or model URI can\'t be empty')

    mlf_cfg.expose_auth_to_env()
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    run = mlflow.get_run(summary.run_id)

    # Don't register if the run failed or stopped by the user
    if finished_only and run.info.status != 'FINISHED':
        raise RuntimeError(
            f'Expected status "FINISHED" but got "{run.info.status}" on '
            f'run id "{summary.run_id}"'
        )

    # Tags to add when registering the model
    regis_tags.model_registered_at = regis_tags.datetime_now()
    regis_tags.train_data_commit_id = summary.data_commit_id

    # MLFlow will print the details after this
    logger.debug(f'Preparing to register the model from run id "{summary.run_id}"')

    # Register the best model from the last run
    status = mlflow.register_model(
        summary.model_uri,
        name = mlf_model.model_registry_name,
        tags = regis_tags.model_dump()
    )
    
    logger.info(
        f'Registered model version "{status.version}" under '
        f'"{mlf_model.model_registry_name}"'
    )

    # By default, the version will be incremental number
    # However, the returned type will always be string
    return status.version


def run(
    data_source_repo: str, data_source_creds: LakeFSConf, train_params: TrainParams,
    regis_tags: ModelRegisTags, model_registry: MLFlowModel, mlflow_creds: MLFlowConf
) -> str:
    # Get the latest data commit id from lakeFS repo
    commit_id = get_data_commit_id(data_source_repo, data_source_creds)
    # Replace the branch name with the exact commit id for preciseness
    data_source_repo = replace_branch(data_source_repo, commit_id)

    # Add the required params for training
    train_params.data_commit_id = commit_id
    train_params.csv_dir = data_source_repo
    train_params.img_dir = data_source_repo + '/images'

    train_summary = model_training(
        train_params, data_source_creds.as_s3(), mlflow_creds
    )

    version = register_model(
        train_summary, regis_tags, model_registry, mlflow_creds, finished_only = True
    )

    return version


def main():
    dotenv.load_dotenv(
        '.env.prod' if Path('.env.prod').exists() else '.env.dev',
        override = False
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

    args = ParseArgs()

    run(
        args.data_source_repo, args.data_source_creds,
        args.train_params, args.regis_tags,
        args.model_registry, args.mlflow_creds
    )


if __name__ == '__main__':
    main()