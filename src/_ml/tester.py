import tempfile
import polars as pl

import torch
from lightning import Fabric
from _ml.model import PawDataLoader

import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.models.evaluation import EvaluationResult

from _pydantic.train_test import TestParams, TestResult
from _pydantic.common import MLFlowConf, S3Conf


class Tester:
    def __init__(
        self, model_name: str, model_version: str, model_uri: str,
        model_id: str | None = None
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.model_uri = model_uri
        self.model_id = model_id

        # Try to get the model id manually if possible
        # This is optional by nice to have for tracking purpose
        if not model_id and model_uri.startswith('models:/'):
            self.model_id = model_uri[len('models:/'):]

    def get_dataframe():
        pass

    def predict(
        self, params: TestParams, mlf_cfg: MLFlowConf, s3_cfg: S3Conf
    ) -> pl.DataFrame:
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)

        # Assume the test file is always located on S3
        df = pl.scan_csv(
            params.csv_dir + '/test.csv',
            storage_options = dict(s3_cfg)
        ).collect()

        # Cache dir for storing downloaded files
        cache_dir = tempfile.TemporaryDirectory()

        test_loader = PawDataLoader(
            df,
            img_dir = params.img_dir,
            # Don't shuffle or augment the data
            is_train_data = False,
            batch_size = params.batch_size,
            img_size = params.img_size,
            s3_cfg = s3_cfg,
            cache_dir = cache_dir.name
        )

        fabric = Fabric(accelerator = 'gpu')
        test_loader = fabric.setup_dataloaders(test_loader)

        # TODO: Use pyfunc to support all MLFlow model flavours
        # The model signature needs to be changed to Pandas/Numpy
        model = mlflow.pytorch.load_model(self.model_uri)
        model = fabric.setup_module(model)
        model.eval()

        # Get batched predictions from the model
        preds = torch.cat([
            model(ds['image'], ds['features'])
            for ds in test_loader
        ]).numpy(force = True)

        # Add the stacked predictions as a single column
        df = df.insert_column(len(df.columns), pl.Series('Prediction', preds))
        # Clean back the cache dir
        cache_dir.cleanup()

        return df

    def run_evaluation(
        self, df: pl.DataFrame, params: TestParams, mlf_cfg: MLFlowConf
    ) -> TestResult:
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
        mlflow.set_experiment(mlf_cfg.experiment_name)

        with mlflow.start_run():
            dataset = PandasDataset(
                df.to_pandas(),
                name = 'lakeFS',
                source = HTTPDatasetSource(params.csv_dir),
                digest = params.data_commit_id[:8]
            )

            # log_input will only save the column schema and the source path/URL
            mlflow.log_input(dataset, context = params.context)
            # To save it as a file too, we need to call log_table separately
            # The saved file can be loaded for drift monitoring purpose later
            mlflow.log_table(df.to_pandas(), 'prediction.json')
            # Also log the params, just like when training
            mlflow.log_params(dict(params))

            result = mlflow.evaluate(
                data = dataset,
                predictions = 'Prediction',
                targets = 'Pawpularity',
                model_type = 'regressor',
                model_id = self.model_id
            )

            # HACK: Type hint for the evaluation result
            if type(result) == EvaluationResult:
                result = result

        return TestResult(
            run_id = result.run_id,
            data_commit_id = params.data_commit_id,
            model_uri = self.model_uri,
            model_version = self.model_version,
            model_registry_name = self.model_name,
            metric = params.metric,
            metric_min = params.metric_min,
            score = result.metrics[params.metric]
        )

    @staticmethod
    def load_prediction(
        run_id: str, mlf_cfg: MLFlowConf, artifact_path: str = 'prediction.json'
    ) -> pl.DataFrame:
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)

        # Assumed to be logged when during the evaluation run
        df = mlflow.load_table(artifact_path, run_ids = [run_id])
        return pl.from_pandas(df)