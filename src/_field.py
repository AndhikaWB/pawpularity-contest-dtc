import warnings
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLFlowSettings(BaseSettings):
    # Allow declaring extra parameters not listed below
    model_config = ConfigDict(extra = 'allow')

    # Will read some values from environment variables if exist
    tracking_uri: Annotated[str, Field(alias = 'MLFLOW_TRACKING_URI')] = 'localhost:5000'
    exp_name: Annotated[str, Field(alias = 'MLFLOW_EXPERIMENT_NAME')] = 'pawpaw-experiment'
    reg_model_name: Annotated[str, Field(description = 'Name to track model versions')] = 'dev.pawpaw-model'
    best_ver_alias: Annotated[str, Field(description = 'Alias for the best model version')] = 'best'


class LakeFSSettings(BaseSettings):
    model_config = ConfigDict(extra = 'allow')

    # lakeFS doesn't read environment variables, I made these up my own
    host: Annotated[str, Field(alias = 'LAKECTL_SERVER_ENDPOINT_URL')] = 'http://localhost:8000'
    username: Annotated[str, Field(alias = 'LAKECTL_CREDENTIALS_ACCESS_KEY_ID')]
    password: Annotated[str, Field(alias = 'LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY')]
    repo_id: str
    branch: str

class TrainParams(BaseModel):
    model_config = ConfigDict(extra = 'allow')

    csv_path: str = 'data/train.csv'
    img_dir: str = 'data/train'
    model_dir: str = 'model'
    sample_size: Annotated[int, Field(ge = 100)] = 1000
    img_size: Annotated[tuple[int], Field(min_length = 2, max_length = 2)] = (128, 128)
    seed: int = 1337
    lr: float = 0.001
    batch_size: Annotated[int, Field(ge = 1)] = 64
    epochs: Annotated[int, Field(ge = 1)] = 20
    monitor: str = 'val_bce'
    patience: Annotated[int, Field(ge = 1)] = 5

    # # Will be set dynamically later (not as user inputs)
    # optimizer: str | None = None
    # criterion: str | None = None


class TrainTags(BaseModel):
    model_config = ConfigDict(extra = 'allow')

    author: Annotated[str, Field(description = 'Author name')]
    lib: Annotated[str, Field(description = 'Library', examples = ['PyTorch', 'mixed'])]
    model: Annotated[str, Field(description = 'Model type', examples = ['CNN'])]
    ext: Annotated[str, Field(description = 'File extension', examples = ['ipynb'])]


class TrainSettings(BaseSettings):
    """Train a model for predicting pet popularity. Use the `--help` flag for help."""

    model_config = SettingsConfigDict(
        cli_parse_args = True,
        cli_kebab_case = True,
        # Validate type when adding tags later
        validate_assignment = True
    )

    params: TrainParams = TrainParams()
    tags: Annotated[TrainTags, Field(validate_default = False)] = None

    @model_validator(mode = 'after')
    def check_tags(self):
        if not self.tags:
            warnings.warn(
                '"tags" is currently not set. You need to set this manually later!',
                UserWarning
            )

        return self