import re
from typing import Annotated
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReportConf(BaseSettings):
    """Database credentials used to write the drift monitoring report.

    Args:
        host (str): Endpoint URL. Defaults to 'http://localhost:5432'.
        database (str): Database name.
        username (str): Username to log in to the database.
        password (str): Password.
    """

    model_config = SettingsConfigDict(validate_by_name = True, validate_default = False, extra = 'allow')

    host: Annotated[str, Field(validation_alias = 'REPORT_DATABASE_HOST')] = 'http://localhost:5432'
    database: Annotated[str, Field(validation_alias = 'REPORT_DATABASE_DBNAME')]
    username: Annotated[str, Field(validation_alias = 'REPORT_DATABASE_USERNAME')]
    password: Annotated[str, Field(validation_alias = 'REPORT_DATABASE_PASSWORD')]

    def postgresql_uri(self) -> str:
        """Return as PostgreSQL connection URI."""

        # Don't include the http:// or https:// part
        host = re.sub(r'[\w]+:\/\/', '', self.host)

        return f'postgresql://{self.username}:{self.password}@{host}/{self.database}'


class ReportSchema(BaseModel):
    """Report (`DataFrame`) schema to be uploaded to database (e.g. the average metric
    score, alert ratio, and the alert conclusion itself)."""

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