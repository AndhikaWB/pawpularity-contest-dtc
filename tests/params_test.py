import os
import dotenv
import unittest
from pathlib import Path

from src._pydantic.common import LakeFSConf
from src._pydantic.train_test import TrainParams, TestParams, MLFlowModel


dotenv.load_dotenv(
    '.env.prod' if Path('.env.prod').exists()
    else '.env.dev'
)

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
        model_name = 'dev.mymodelname'
        os.environ['DEV_MODEL_REGISTRY_NAME'] = model_name

        registry = MLFlowModel()
        assert registry.model_registry_name == model_name
    
    def test_lakefs_config(self):
        secret_id = 'mysupersecretid'
        os.environ['LAKECTL_CREDENTIALS_ACCESS_KEY_ID'] = secret_id

        lakefs_cfg = LakeFSConf()
        assert lakefs_cfg.username == secret_id


if __name__ == '__main__':
    unittest.main()