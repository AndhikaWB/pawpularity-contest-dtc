import warnings
import tempfile
from pathlib import Path
from copy import deepcopy

import mlflow
from _ml.trainer import Trainer
from _ml.model import PawDataLoader, PawModel

import dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from _pydantic.shared import S3Conf, LakeFSConf, MLFlowConf
from _pydantic.train_test import TrainParams, TrainTags, MLFlowModel

import lakefs
from _s3.utils import get_repo_branch


def get_data_commit_id(data_source_repo: str, lfs_cfg: LakeFSConf) -> str:
    client = lakefs.Client(**dict(lfs_cfg))

    # Get the branch name or commit id from the input URI
    # E.g. s3://my-repo/main or s3://my-repo/64675c312d48be7e
    repo_id, ref_id = get_repo_branch(data_source_repo)
    ref = lakefs.Reference(repo_id, ref_id, client = client)

    # But we should only return a commit id, not branch name
    # This commit id will be logged as MLFlow parameter later
    commit_id = ref.get_commit().id
    print(f'Using commit "{commit_id[:8]}" from data repo "{repo_id}"')

    return commit_id


def model_training(
    params: TrainParams, tags: TrainTags, s3_cfg: S3Conf, mlf_cfg: MLFlowConf
) -> TrainParams:
    if not (params.csv_dir and params.img_dir):
        raise ValueError('CSV and image directory can\'t be empty')

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
    # This will add the run id to the returned params
    params = trainer.start_training(params, tags, mlf_cfg)

    # Get the best model only and delete other models from that run
    # This will add the best model URI to the returned params
    print('Filtering and keeping only the best model')
    params = trainer.get_best_model(params, mlf_cfg, delete_others = True)
    # Clean back the cache dir
    cache_dir.cleanup()

    return params


def register_model(
    params: TrainParams, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf,
    finished_only: bool = True
) -> str:
    if not (params.run_id and params.best_model_uri):
        raise ValueError('Run id and model URI can\'t be empty')

    mlf_cfg.expose_auth_to_env()
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    run = mlflow.get_run(params.run_id)

    # Don't register if the run failed or stopped by the user
    if finished_only and run.info.status != 'FINISHED':
        raise RuntimeError(
            f'Expected status "FINISHED" but got "{run.info.status}" on '
            f'run id "{params.run_id}"'
        )

    # MLFlow will print the details after this
    print('Preparing to register the model')

    status = mlflow.register_model(
        params.best_model_uri,
        mlf_model.registered_model_name
    )

    # By default, the version will be incremental number
    # However, the returned type will always be string
    return status.version


def run(
    data_source_repo: str, data_source_creds: LakeFSConf, train_params: TrainParams,
    train_tags: TrainTags, model_registry: MLFlowModel, mlflow_creds: MLFlowConf
):
    # Get the last data commit id from lakeFS and add it as params
    commit_id = get_data_commit_id(data_source_repo, data_source_creds)
    train_params.data_commit_id = commit_id

    # Set CSV and image directory based on the source repo
    # We don't use pathlib because it will break the S3 URI
    train_params.csv_dir = data_source_repo
    train_params.img_dir = data_source_repo + '/images'
    train_params.epochs = 1

    train_params = model_training(
        train_params, train_tags, data_source_creds.as_s3(), mlflow_creds
    )

    register_model(
        train_params, model_registry, mlflow_creds, finished_only = True
    )


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