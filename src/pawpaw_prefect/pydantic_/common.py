from typing import Annotated

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeployConf(BaseSettings):
    """Used when serving/deploying flow to Prefect remote server.

    Args:
        environment (str, optional): Environment name to use alongside the raw
            deployment name. Defaults to 'dev'.
        raw_deployment_name (str): Name for the deployment (there can be multiple
            workflows assigned to one deployment). For a better practice, please use
            `deployment_name` instead, it adds the environment before the raw deployment
            name (e.g. `dev.deployment_name`).
    """

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