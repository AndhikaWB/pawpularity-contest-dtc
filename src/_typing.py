from typing import Annotated
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class TrainParams(BaseModel):
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

class TrainTags(BaseModel):
    developer: str
    model: str
    format: str
    type: str

class TrainSettings(BaseSettings):
    """Please call the program with the `--help` flag for help."""
    model_config = SettingsConfigDict(cli_parse_args = True)

    tags: TrainTags
    params: TrainParams