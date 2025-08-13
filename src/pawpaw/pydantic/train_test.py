# Bypass line length limit
# ruff: noqa: E501

from datetime import datetime

from typing import Annotated
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict, CLI_SUPPRESS


class MLFlowModel(BaseSettings):
    """Used when registering, loading, or promoting an MLFlow model.

    Args:
        environment (str, optional): Environment name to use alongside the raw model
            name. Useful if you want to promote or use multiple model environments.
            Defaults to 'dev'.
        raw_model_name (str): Name to use when registering or searching a model. Each
            time we register a model under this name, it will be added as a new model
            version. For a better practice, please use `model_registry_name` instead, it
            adds the environment before the raw model name.
        best_version_alias (str, optional): Alias for the best model version. Used to
            differentiate it from other model version, and for easy model loading later.
            Defaults to 'best'.
    """

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    # You can change the environment if you want to promote the model later
    environment: Annotated[str, Field(validation_alias = 'TRAIN_MODEL_REGISTRY_ENV')] = 'dev'
    raw_model_name: Annotated[str, Field(validation_alias = 'TRAIN_MODEL_REGISTRY_NAME')]
    best_version_alias: Annotated[str, Field(validation_alias = 'TEST_BEST_VERSION_ALIAS')] = 'best'

    @computed_field
    @property
    def model_registry_name(self) -> str:
        """Return the environment together with the model name (e.g. `dev.model_name`).
        This is the suggested way by MLFlow for registering a model.
        """

        return f'{self.environment}.{self.raw_model_name}'


class TestParams(BaseSettings):
    """Parameters that will be used to configure the testing process, and logged to
    MLFlow when starting an evaluation run. Can also be used as parameters when serving
    the model.
    """

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    # Metric name here must match the name returned by "mlflow.evaluate" function
    # So it may be different from training metric (e.g. rmse vs root_mean_square_error)
    metric: Annotated[str, Field(validation_alias = 'TEST_METRIC')]
    metric_min: Annotated[bool, Field(validation_alias = 'TEST_METRIC_MIN')] = True
    metric_threshold: Annotated[float, Field(validation_alias = 'TEST_METRIC_THRESHOLD')]

    # To filter from other types of run
    context: Annotated[str, Field(description = CLI_SUPPRESS)] = 'testing'

    # Will be set after getting the repo data from lakeFS
    data_commit_id: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    csv_dir: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    img_dir: Annotated[str, Field(description = CLI_SUPPRESS)] = None

    # Just to configure the data loader
    img_size: Annotated[tuple[int, int], Field(validation_alias = 'TRAIN_IMG_SIZE')] = (128, 128)
    batch_size: Annotated[int, Field(ge = 1, validation_alias = 'TRAIN_BATCH_SIZE')] = 64


class TrainParams(BaseSettings):
    """Parameters that will be used to configure the training process, and logged to
    MLFlow when starting a training run.
    """

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    # User inputs, configurable via CLI arguments or environment variables
    img_size: Annotated[tuple[int, int], Field(validation_alias = 'TRAIN_IMG_SIZE')] = (128, 128)
    seed: Annotated[int, Field(validation_alias = 'TRAIN_SEED')] = 1337
    lr: Annotated[float, Field(ge = 0.00, validation_alias = 'TRAIN_LR')] = 0.001
    batch_size: Annotated[int, Field(ge = 1, validation_alias = 'TRAIN_BATCH_SIZE')] = 64
    epochs: Annotated[int, Field(ge = 1, validation_alias = 'TRAIN_EPOCH')] = 1
    patience: Annotated[int, Field(ge = 1, validation_alias = 'TRAIN_PATIENCE')] = 5

    # To filter from other types of run
    context: Annotated[str, Field(description = CLI_SUPPRESS)] = 'training'

    # Will be set after getting the repo data from lakeFS
    data_commit_id: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    csv_dir: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    img_dir: Annotated[str, Field(description = CLI_SUPPRESS)] = None

    # Will be set when preparing for training
    optimizer: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    criterion: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    # There's currently no standard for the metric name used here
    # It's totally up to the creator who made the training script
    monitor: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    monitor_min: Annotated[bool, Field(description = CLI_SUPPRESS)] = True

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


class TrainSummary(BaseModel):
    """Result of a model training process from MLFlow, containing the run id, model URI,
    and other useful info. The term "summary" is used because "result" is just too
    common, and can be confusing when mixed with all other variables.
    """

    run_id: str
    data_commit_id: str
    model_uri: str

    metric: str
    metric_min: bool
    score: float


class TestSummary(BaseModel):
    """Result of a model testing process from MLFlow, containing the run id, model URI,
    and other useful info. The term "summary" is used because "result" is just too
    common, and can be confusing when mixed with all other variables.
    """

    run_id: str
    data_commit_id: str
    model_uri: str

    model_version: str
    model_registry_name: str

    metric: str
    metric_min: bool
    metric_threshold: float
    score: float


class ModelRegisTags(BaseModel):
    """Tags to attach (to MLFlow model) when registering a new model version."""

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    model_registered_at: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    train_data_commit_id: Annotated[str, Field(description = CLI_SUPPRESS)] = None

    framework: Annotated[str, Field(validation_alias = 'MODEL_TAG_FRAMEWORK')] = 'PyTorch'
    variant: Annotated[str, Field(description = 'MODEL_TAG_VARIANT')] = 'Simple CNN'

    @staticmethod
    def datetime_now() -> str:
        """Get the current date and time as string (MLFlow may reject tag value other
        than string).
        """

        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class ModelBestTags(BaseModel):
    """Tags to attach (to MLFlow model) when registering an alias for the current
    best model version.
    """

    model_marked_best_at: str
    test_data_commit_id: str

    @staticmethod
    def datetime_now() -> str:
        """Get the current date and time as string (MLFlow may reject tag value other
        than string).
        """

        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')