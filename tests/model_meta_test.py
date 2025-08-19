import os

from pydantic_settings import BaseSettings

from pawpaw.pydantic_.serve import ServeConf
from pawpaw.pydantic_.report import ReportConf
from pawpaw.pydantic_.common import S3Conf, LakeFSConf, MLFlowConf
from pawpaw.pydantic_.train_test import TrainParams, TestParams, MLFlowModel
from pawpaw.pydantic_.train_test import ModelRegisTags


def compare_metadata(ref_model: type[BaseSettings], cur_model: type[BaseSettings]):
    if not issubclass(cur_model, BaseSettings):
        raise TypeError(f'"{cur_model.__name__}" is not a "BaseSettings" class')

    metadata_to_check = [
        # Allow instantiating a field via the alias and/or the real name
        # This is important especially if we read value from environment variable
        'validate_by_alias', 'validate_by_name',
        # Extra can decide whether to save the name/alias as a main/extra field
        # If we fetch the main but our value is in extra, the value we get may be wrong
        'extra'
    ]

    for field in cur_model.model_fields.values():
        if field.alias or field.validation_alias:
            for metadata in metadata_to_check:
                # Current metadata must be match the reference metadata
                # This ensure we have consistent behavior on both models
                cur_meta = ref_model.model_config.get(metadata)
                ref_meta = ref_model.model_config.get(metadata)
                assert cur_meta == ref_meta, (
                    f'"{cur_model.__name__}:{metadata}" should be {ref_meta}, but got '
                    f'{cur_meta} instead'
                )


class TestMetadata:
    """Test metadata consistency across all models that can read values from
    environment variables.
    """

    def test_s3_config(self):
        # Environment variable (second priority)
        false_secret_key = 'falsesecretkey'
        os.environ['AWS_SECRET_ACCESS_KEY'] = false_secret_key
        # Direct argument (first priority)
        true_secret_key = 'truesecretkey'

        # Under the right config, true secret will be prioritized
        # Hence we're checking if our config are the right one
        s3_cfg = S3Conf(
            endpoint_url = 'http://localhost:9000',
            aws_access_key_id = 'mysecretid',
            aws_secret_access_key = true_secret_key
        )

        key_direct = s3_cfg.aws_secret_access_key
        key_dump = s3_cfg.model_dump()['aws_secret_access_key']

        # These may fail depending on the model config
        assert key_direct == true_secret_key, (
            f'"{s3_cfg.__name__}.aws_secret_access_key" should be {true_secret_key}, '
            f'but got {key_direct} instead'
        )

        assert key_direct == key_dump, (
            f'"{s3_cfg.__name__}.aws_secret_access_key" should have the same value as '
            f'from "model_dump()", but "model_dump()" has {key_dump} instead'
        )

    def test_lakefs_config(self):
        compare_metadata(S3Conf, LakeFSConf)

    def test_mlflow_config(self):
        compare_metadata(S3Conf, MLFlowConf)

    def test_train_params(self):
        compare_metadata(S3Conf, TrainParams)
    
    def test_test_params(self):
        compare_metadata(S3Conf, TestParams)
    
    def test_model_registry(self):
        compare_metadata(S3Conf, MLFlowModel)

    def test_model_register_tags(self):
        compare_metadata(S3Conf, ModelRegisTags)
    
    def test_model_best_tags(self):
        compare_metadata(S3Conf, ModelRegisTags)
    
    def test_report_config(self):
        compare_metadata(S3Conf, ReportConf)

    def test_serve_config(self):
        compare_metadata(S3Conf, ServeConf)