import os
import dotenv
import warnings
import unittest
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings

from pawpaw.pydantic_.serve import ServeConf
from pawpaw.pydantic_.report import ReportConf
from pawpaw.pydantic_.common import LakeFSConf, MLFlowConf
from pawpaw.pydantic_.train_test import TrainParams, TestParams, MLFlowModel
from pawpaw.pydantic_.train_test import ModelRegisTags


def check_alias_against_env(model: type[BaseSettings]):
    if not issubclass(model, BaseSettings):
        raise TypeError(f'"{model.__name__}" is not a "BaseSettings" class')

    for field in model.model_fields.values():
        # Check if the field can read from environment variable
        # By checking if the field has an alias or validation alias
        if field.alias or field.validation_alias:
            # Check if that alias exists in our environment variable
            # By getting that alias value from our environment
            cur_env_value = os.getenv(field.validation_alias, None)

            if field.is_required():
                try:
                    # Check if the current value is a compatible field type
                    # Will also check other constraints like max length, etc.
                    model(**{field.validation_alias: cur_env_value})
                except ValidationError as e:
                    for error in e.errors():
                        if field.validation_alias in error['loc']:
                            # This can be caused by incorrect model_config
                            # Or just an alias error (e.g. missing from env)
                            raise ValidationError.from_exception_data(
                                model.__name__,
                                [error]
                            )
            else:
                if not cur_env_value:
                    # Optional field that can read from environment variable
                    # But we don't have that alias set in our environment
                    warnings.warn(
                        f'Alias "{field.validation_alias}" on model "{model.__name__}" '
                        f'is not set, will fallback to "{field.default}"'
                    )


class EnvironmentTesting(unittest.TestCase):
    """Test if we have the environment variables that are needed by the models."""

    def setUp(self):
        # Workaround if the code is not executed as a __main__ script
        # The environment is guaranteed to be refreshed this way
        dotenv.load_dotenv(
            '.env.prod' if Path('.env.prod').exists() else '.env.dev',
            override = False
        )

    def test_lakefs_config(self):
        check_alias_against_env(LakeFSConf)

    def test_mlflow_config(self):
        check_alias_against_env(MLFlowConf)

    def test_train_params(self):
        check_alias_against_env(TrainParams)
    
    def test_test_params(self):
        check_alias_against_env(TestParams)
    
    def test_model_registry(self):
        check_alias_against_env(MLFlowModel)
    
    def test_model_register_tags(self):
        check_alias_against_env(ModelRegisTags)
    
    def test_report_config(self):
        check_alias_against_env(ReportConf)

    def test_serve_config(self):
        check_alias_against_env(ServeConf)


if __name__ == '__main__':
    unittest.main()