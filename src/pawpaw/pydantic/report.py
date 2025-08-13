# Bypass line length limit
# ruff: noqa: E501

import re
from datetime import datetime

from typing import Annotated
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReportConf(BaseSettings):
    """Database credentials used to write the drift monitoring report."""

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    host: Annotated[str, Field(validation_alias = 'REPORT_DB_HOST')] = 'http://localhost:5432'
    database: Annotated[str, Field(validation_alias = 'REPORT_DB_DATABASE')]
    username: Annotated[str, Field(validation_alias = 'REPORT_DB_USERNAME')]
    password: Annotated[str, Field(validation_alias = 'REPORT_DB_PASSWORD')]

    def postgresql_uri(self):
        """Return as PostgreSQL connection URI."""

        # Don't include the http:// or https:// part
        host = re.sub(r'[\w]+:\/\/', '', self.host)

        return f'postgresql://{self.username}:{self.password}@{host}/{self.database}'


class ReportSchema(BaseModel):
    """Report (dataframe) schema to be uploaded to database."""

    time: datetime
    run_id_current: str
    run_id_reference: str
    commit_id_current: str
    commit_id_reference: str
    column_name: str
    method: str
    value_average: float
    value_threshold: float
    safe_count: int
    alert_count: int
    alert_ratio: float
    alert_ratio_threshold: float
    alert: bool