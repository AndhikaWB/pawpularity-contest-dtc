from typing import Annotated

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeployConf(BaseSettings):
    """Used when serving/deploying flow to Prefect remote server."""

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    environment: Annotated[str, Field(validation_alias = 'ORCHESTRATION_DEPLOYMENT_ENV')] = 'dev'
    raw_deployment_name: Annotated[str, Field(validation_alias = 'ORCHESTRATION_DEPLOYMENT_NAME')]

    @computed_field
    @property
    def deployment_name(self) -> str:
        """Return the environment together with the flow deployment name (e.g.
        `dev.my_deployment`).
        """

        return f'{self.environment}.{self.raw_deployment_name}'