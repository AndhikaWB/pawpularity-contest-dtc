# Bypass line length limit
# ruff: noqa: E501

import re
from typing import Annotated
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReportConf(BaseSettings):
    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False)

    host: Annotated[str, Field(validation_alias = 'REPORT_DB_HOST')] = 'http://localhost:5432'
    database: Annotated[str, Field(validation_alias = 'REPORT_DB_DATABASE')]
    username: Annotated[str, Field(validation_alias = 'REPORT_DB_USERNAME')]
    password: Annotated[str, Field(validation_alias = 'REPORT_DB_PASSWORD')]

    def postgresql_uri(self):
        """Return the PostgreSQL connection URI."""

        # Don't include the http:// or https:// part
        host = re.sub(r'[\w]+:\/\/', '', self.host)

        return f'postgresql://{self.username}:{self.password}@{host}/{self.database}'