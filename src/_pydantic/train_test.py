import torchmetrics as tm
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, CLI_SUPPRESS


class MLFlowModel(BaseSettings):
    """Used when registering or testing an MLFlow model.

    Args:
        registered_model_name (str): Name to check or register the model. Each time we
            register a model under this name, it will be added as a new model version.
        best_version_alias (str, optional): Alias to set for the best model version
            after comparing the metric. Defaults to 'best'.
        compare_metric (str): Metric class name that will be used to compare different
            model versions. It checks the class name from the `torchmetrics` library.
        compare_metric_min (bool): Whether min or max value is better for the metric.
            Defaults to True (min is better).
    """

    model_config = SettingsConfigDict(extra = 'allow', validate_by_name = True)

    registered_model_name: Annotated[str, Field(validation_alias = 'TEST_REGISTERED_MODEL_NAME')]
    best_version_alias: Annotated[str, Field(validation_alias = 'TEST_BEST_VERSION_ALIAS')] = 'best'

    # Not neccesarily the same metric as the one used for training
    compare_metric: Annotated[str, Field(validation_alias = 'TEST_COMPARE_METRIC')]
    compare_metric_min: Annotated[bool, Field(validation_alias = 'TEST_COMPARE_METRIC_MIN')] = True

    def get_metric_class(self):
        pass


class TrainParams(BaseModel):
    """Parameters that will be used or logged to MLFlow when training, but can be used
    outside the training context too (e.g. for testing).
    """

    model_config = ConfigDict(extra = 'allow')

    # User inputs
    img_size: tuple[int, int] = (128, 128)
    seed: int = 1337
    lr: Annotated[float, Field(ge = 0.00)] = 0.001
    batch_size: Annotated[int, Field(ge = 1)] = 64
    epochs: Annotated[int, Field(ge = 1)] = 20
    patience: Annotated[int, Field(ge = 1)] = 5

    # Will be set after getting repo data from lakeFS
    data_commit_id: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    csv_dir: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    img_dir: Annotated[str, Field(description = CLI_SUPPRESS)] = None

    # Will be set when preparing training
    optimizer: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    criterion: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    monitor: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    monitor_min: Annotated[bool, Field(description = CLI_SUPPRESS)] = True

    # Will be set after training
    run_id: Annotated[str, Field(description = CLI_SUPPRESS)] = None
    best_model_uri: Annotated[str, Field(description = CLI_SUPPRESS)] = None


class TrainTags(BaseModel):
    """Tags to log to MLFlow when starting a run/training. Note that these run tags are
    not the same as model tags. You can/must set separate tags for the model later.

    Args:
        author (str, optional): The responsible author name. Defaults to 'Bot'.
        framework (str, optional): Main framework of the ML model (e.g. TensorFlow,
            PyTorch). Defaults to 'PyTorch'.
        model (str, optional): Model type/variant (e.g. XGBoost, CNN). You can use
            anything but models with the same type/variant should always have matching
            tag. Defaults to 'Simple CNN'.
        extension (str, optional): File extension to help differentiate development
            stage. `ipynb` is used for early prototype while `py` is more production
            ready. Defaults to 'py'.
    """

    model_config = ConfigDict(extra = 'allow')

    author: Annotated[str, Field(description = 'Author')] = 'Bot'
    framework: Annotated[str, Field(description = 'Framework')] = 'PyTorch'
    model: Annotated[str, Field(description = 'Model type/variant')] = 'Simple CNN'
    extension: Annotated[str, Field(description = 'File extension')] = 'py'
