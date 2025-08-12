# Bypass line length limit
# ruff: noqa: E501

import os
from typing import Annotated

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ValidOrNone:
    """Fallback to a `None` value upon a failed Pydantic model instantiation. You can
    pass this as the `default_factory` parameter.

    Args:
        model (BaseModel): The Pydantic `BaseModel` or `BaseSettings` that you are
            trying to instantiate.

    Examples:
        ```
        class Test:
            s3_cfg: S3Conf | None = Field(default_factory = ValidOrNone(S3Conf))
        ```
    """

    def __init__(self, model: BaseModel):
        self.model = model

    def __call__(self):
        try:
            return self.model()
        except ValidationError:
            pass


class S3Conf(BaseSettings):
    # BUG: If both params and env variables exist, the final value may be inconsistent
    # We should check this by comparing "my_var.key" with "my_var.model_dump()[key]"
    model_config = SettingsConfigDict(validate_by_name = True, validate_by_alias = False, extra = 'allow')
    
    endpoint_url: Annotated[str, Field(alias = 'AWS_ENDPOINT_URL')] = 'http://localhost:9000'
    aws_access_key_id: Annotated[str, Field(alias = 'AWS_ACCESS_KEY_ID')]
    aws_secret_access_key: Annotated[str, Field(alias = 'AWS_SECRET_ACCESS_KEY')]


class MinIOConf(BaseSettings):
    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    # MinIO actually doesn't read any environment variable to set the endpoint URL
    # There was MINIO_SERVER_URL but it has been deprecated and may cause issues if used
    endpoint: Annotated[str, Field(validation_alias = 'MINIO_ENDPOINT_URL')] = 'http://localhost:9000'
    # These two variables will still be read by MinIO though
    access_key: Annotated[str, Field(validation_alias = 'MINIO_ROOT_USER')]
    secret_key: Annotated[str, Field(validation_alias = 'MINIO_ROOT_PASSWORD')]

    def as_s3(self) -> S3Conf:
        return S3Conf(
            endpoint_url = self.endpoint,
            aws_access_key_id = self.access_key,
            aws_secret_access_key = self.secret_key
        )


class LakeFSConf(BaseSettings):
    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    host: Annotated[str, Field(validation_alias = 'LAKECTL_SERVER_ENDPOINT_URL')] = 'http://localhost:8000'
    username: Annotated[str, Field(validation_alias = 'LAKECTL_CREDENTIALS_ACCESS_KEY_ID')]
    password: Annotated[str, Field(validation_alias = 'LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY')]

    def as_s3(self) -> S3Conf:
        return S3Conf(
            endpoint_url = self.host,
            aws_access_key_id = self.username,
            aws_secret_access_key = self.password
        )


class MLFlowConf(BaseSettings):
    """MLFlow credentials containing the server tracking URI, username, password, and
    experiment name.

    Args:
        tracking_uri (str, optional): The MLFlow server that the client will connect to.
            Defaults to 'http://localhost:5000'.
        username (str, optional): Username for accessing the server. Defaults to None.
        password (str, optional): Password for accessing the server, must be at least
            12 characters long if set. Defaults to None.
        experiment_name (str): The default experiment for tracking the runs. Every time
            you train a model, a new run will be created under this experiment.
    """

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    # Will read aliases from environment variable if they exist
    tracking_uri: Annotated[str, Field(validation_alias = 'MLFLOW_TRACKING_URI')] = 'http://localhost:5000'
    username: Annotated[str | None, Field(validation_alias = 'MLFLOW_TRACKING_USERNAME')] = None
    password: Annotated[str | None, Field(validation_alias = 'MLFLOW_TRACKING_PASSWORD')] = None
    experiment_name: Annotated[str, Field(validation_alias = 'MLFLOW_EXPERIMENT_NAME')]

    def expose_auth_to_env(self):
        """Expose username and password as environment variables. Currently, this is the
        only way to pass them to MLFLow, because auth is considered an add-on.
        """

        if self.username and self.password:
            os.environ['MLFLOW_TRACKING_USERNAME'] = self.username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = self.password