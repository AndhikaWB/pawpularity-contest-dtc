import os
import dotenv
import unittest
from pathlib import Path

from pawpaw.pydantic.common import LakeFSConf, S3Conf
from pawpaw.pydantic.train_test import TrainParams, TestParams, MLFlowModel


class ParameterTesting(unittest.TestCase):
    def test_train_params(self):
        batch_size = 999
        os.environ['TRAIN_BATCH_SIZE'] = str(batch_size)

        params = TrainParams()
        assert params.batch_size == batch_size
    
    def test_test_params(self):
        metric = 'mean_absolute_error'
        os.environ['TEST_METRIC'] = metric

        params = TestParams()
        assert params.metric == metric
    
    def test_model_registry(self):
        env_name = 'dev'
        os.environ['TRAIN_MODEL_REGISTRY_ENV'] = env_name

        model_name = 'mymodelname'
        os.environ['TRAIN_MODEL_REGISTRY_NAME'] = model_name

        registry = MLFlowModel()
        assert registry.model_registry_name == f'{env_name}.{model_name}'
    
    def test_lakefs_config(self):
        secret_id = 'mysupersecretid'
        os.environ['LAKECTL_CREDENTIALS_ACCESS_KEY_ID'] = secret_id

        lakefs_cfg = LakeFSConf()
        assert lakefs_cfg.username == secret_id

    def test_s3_config(self):
        true_secret_key = 'truesecretkey'
        false_secret_key = 'falsesecretkey'
        os.environ['AWS_SECRET_ACCESS_KEY'] = false_secret_key

        s3_cfg = S3Conf(
            endpoint_url = 'http://localhost:9000',
            aws_access_key_id = 'mysecretid',
            aws_secret_access_key = true_secret_key
        )

        key_direct = s3_cfg.aws_secret_access_key
        key_dump = s3_cfg.model_dump()['aws_secret_access_key']

        # This may fail depending on the Pydantic settings
        # Hence I'm checking if we're using the correct settings
        assert key_direct == key_dump
        assert key_direct == true_secret_key


if __name__ == '__main__':
    dotenv.load_dotenv(
        '.env.prod' if Path('.env.prod').exists()
        else '.env.dev'
    )

    unittest.main()