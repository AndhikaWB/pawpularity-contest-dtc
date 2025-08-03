import warnings
import tempfile
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import mlflow
from _ml.trainer import Trainer
from _ml.model import PawDataLoader, PawModel

import dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from _pydantic.common import S3Conf, LakeFSConf, MLFlowConf
from _pydantic.train_test import TrainParams, TrainTags, MLFlowModel, TrainResult

from _s3.lakefs import replace_branch, get_exact_commit


def get_data_commit_id(data_source_repo: str, lfs_cfg: LakeFSConf) -> str:
    commit_id = get_exact_commit(data_source_repo, lfs_cfg, return_id = True)
    if not commit_id:
        raise RuntimeError(f'Can\'t get latest commit from "{data_source_repo}"')

    print(f'Using commit "{commit_id[:8]}" from "{data_source_repo}"')
    return commit_id


def model_training(
    params: TrainParams, tags: TrainTags, s3_cfg: S3Conf, mlf_cfg: MLFlowConf
) -> TrainResult:
    if not (params.csv_dir or params.img_dir or params.data_commit_id):
        raise ValueError('Data sources or commit id can\'t be empty')

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
    print(f'Preparing to train "{model.__class__.__name__}" model')

    # Don't modify the original params directly
    params = deepcopy(params)
    # Setup optimizer, early stop callback, and other things
    # This will add optimizer name, etc. to the returned params
    params = trainer.prep_training(params)
    # Log the params and tags to MLFlow and train the model
    run_id = trainer.start_training(params, tags, mlf_cfg)

    # Get the best model only and delete other models from this run
    # The result will contain run id, best model URI, and metric info
    print(f'Keeping only the best model from run id "{run_id}"')
    result = trainer.get_best_model(run_id, params, mlf_cfg, delete_others = True)
    # Clean back the cache dir
    cache_dir.cleanup()

    return result


def register_model(
    result: TrainResult, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf,
    finished_only: bool = True
) -> str:
    if not (result.run_id or result.model_uri):
        raise ValueError('Run id or model URI can\'t be empty')

    mlf_cfg.expose_auth_to_env()
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    run = mlflow.get_run(result.run_id)

    # Don't register if the run failed or stopped by the user
    if finished_only and run.info.status != 'FINISHED':
        raise RuntimeError(
            f'Expected status "FINISHED" but got "{run.info.status}" on '
            f'run id "{result.run_id}"'
        )

    # MLFlow will print the details after this
    print('Preparing to register the model')
    # Register the best model from the last run
    status = mlflow.register_model(
        result.model_uri,
        mlf_model.model_registry_name,
        tags = {
            # NOTE: Check the evaluation script for possible tag conflicts
            'model_registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_data_commit_id': result.data_commit_id
        }
    )

    # By default, the version will be incremental number
    # However, the returned type will always be string
    return status.version


def run(
    data_source_repo: str, data_source_creds: LakeFSConf, train_params: TrainParams,
    train_tags: TrainTags, model_registry: MLFlowModel, mlflow_creds: MLFlowConf
) -> str:
    # Get the latest data commit id from lakeFS repo
    commit_id = get_data_commit_id(data_source_repo, data_source_creds)
    # Replace the branch name with the exact commit id for preciseness
    data_source_repo = replace_branch(data_source_repo, commit_id)

    # Add the required params for training
    train_params.data_commit_id = commit_id
    train_params.csv_dir = data_source_repo
    train_params.img_dir = data_source_repo + '/images'

    train_result = model_training(
        train_params, train_tags, data_source_creds.as_s3(), mlflow_creds
    )

    version = register_model(
        train_result, model_registry, mlflow_creds, finished_only = True
    )

    return version


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