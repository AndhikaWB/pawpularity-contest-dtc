# Bypass line length limit
# ruff: noqa: E501

from argparse import SUPPRESS
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLFlowModel(BaseSettings):
    """Used when registering, testing, or promoting an MLFlow model.

    Args:
        model_registry_name (str): Name to use when loading or registering a model. Each
            time we register a model under this name, it will be added as a new model
            version.
        best_version_alias (str, optional): Alias to set for the best model version,
            e.g. after doing an evaluation and comparing the metric. Defaults to 'best'.
    """

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False)

    # TODO: Make it so we can switch environment/promote model easily
    model_registry_name: Annotated[str, Field(validation_alias = 'DEV_MODEL_REGISTRY_NAME')]
    best_version_alias: Annotated[str, Field(validation_alias = 'DEV_BEST_VERSION_ALIAS')] = 'best'


class TestParams(BaseSettings):
    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False)

    # Metric name here must match the name returned by "mlflow.evaluate" function
    # So it may be different from training metric (e.g. rmse vs root_mean_square_error)
    metric: Annotated[str, Field(validation_alias = 'TEST_METRIC')]
    metric_min: Annotated[bool, Field(validation_alias = 'TEST_METRIC_MIN')] = True
    metric_threshold: Annotated[float, Field(validation_alias = 'TEST_METRIC_THRESHOLD')]

    # To filter from other types of run
    context: Annotated[str, Field(description = SUPPRESS)] = 'testing'

    # Will be set after getting the repo data from lakeFS
    data_commit_id: Annotated[str, Field(description = SUPPRESS)] = None
    csv_dir: Annotated[str, Field(description = SUPPRESS)] = None
    img_dir: Annotated[str, Field(description = SUPPRESS)] = None

    # Just to configure the data loader
    img_size: Annotated[tuple[int, int], Field(validation_alias = 'TRAIN_IMG_SIZE')] = (128, 128)
    batch_size: Annotated[int, Field(ge = 1, validation_alias = 'TRAIN_BATCH_SIZE')] = 64


class TrainParams(BaseSettings):
    """Parameters that will be used to configure the training process, and logged to
    MLFlow when starting the training run.
    """

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False)

    # User inputs, configurable via CLI arguments or environment variables
    img_size: Annotated[tuple[int, int], Field(validation_alias = 'TRAIN_IMG_SIZE')] = (128, 128)
    seed: Annotated[int, Field(validation_alias = 'TRAIN_SEED')] = 1337
    lr: Annotated[float, Field(ge = 0.00, validation_alias = 'TRAIN_LR')] = 0.001
    batch_size: Annotated[int, Field(ge = 1, validation_alias = 'TRAIN_BATCH_SIZE')] = 64
    epochs: Annotated[int, Field(ge = 1, validation_alias = 'TRAIN_EPOCH')] = 1
    patience: Annotated[int, Field(ge = 1, validation_alias = 'TRAIN_PATIENCE')] = 5

    # To filter from other types of run
    context: Annotated[str, Field(description = SUPPRESS)] = 'training'

    # Will be set after getting the repo data from lakeFS
    data_commit_id: Annotated[str, Field(description = SUPPRESS)] = None
    csv_dir: Annotated[str, Field(description = SUPPRESS)] = None
    img_dir: Annotated[str, Field(description = SUPPRESS)] = None

    # Will be set when preparing for training
    optimizer: Annotated[str, Field(description = SUPPRESS)] = None
    criterion: Annotated[str, Field(description = SUPPRESS)] = None
    # There's currently no standard for the metric name used here
    # It's totally up to the creator who made the training script
    monitor: Annotated[str, Field(description = SUPPRESS)] = None
    monitor_min: Annotated[bool, Field(description = SUPPRESS)] = True

    def to_test(self) -> TestParams:
        """Try initializing the `TestParams` based on this instance data. Other required
        parameters must be provided via environment variables.
        """

        return TestParams(
            data_commit_id = self.data_commit_id,
            csv_dir = self.csv_dir,
            img_dir = self.img_dir,
            img_size = self.img_size,
            batch_size = self.batch_size
        )


class TrainTags(BaseSettings):
    """Tags to log to MLFlow when starting a run/training. Note that these run tags are
    not the same as model tags. You can/must set separate tags for the model later.

    Args:
        author (str, optional): The responsible author name. Defaults to 'Bot'.
        framework (str, optional): Main framework of the ML model (e.g. TensorFlow,
            PyTorch). Defaults to 'PyTorch'.
        model (str, optional): Model type/variant (e.g. XGBoost, CNN). You can use
            anything but models with the same type/variant should ideally have a
            matching tag. Defaults to 'Simple CNN'.
        extension (str, optional): File extension to help differentiate early
            development stage. `ipynb` is used for prototype while `py` is more
            production ready. Defaults to 'py'.
    """

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False)

    author: Annotated[str, Field(validation_alias = 'TRAIN_TAG_AUTHOR')] = 'Bot'
    framework: Annotated[str, Field(validation_alias = 'TRAIN_TAG_FRAMEWORK')] = 'PyTorch'
    model: Annotated[str, Field(description = 'TRAIN_TAG_MODEL')] = 'Simple CNN'
    extension: Annotated[str, Field(validation_alias = 'TRAIN_TAG_EXTENSION')] = 'py'


class TrainSummary(BaseModel):
    run_id: str
    data_commit_id: str
    model_uri: str

    metric: str
    metric_min: bool
    score: float


class TestSummary(BaseModel):
    run_id: str
    data_commit_id: str
    model_uri: str

    model_version: str
    model_registry_name: str

    metric: str
    metric_min: bool
    metric_threshold: float
    score: float