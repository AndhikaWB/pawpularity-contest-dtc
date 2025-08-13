import tempfile
import warnings
import polars as pl

import torch
from lightning import Fabric
from pawpaw.ml.model import PawDataLoader

import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.models.evaluation import EvaluationResult
from mlflow.exceptions import MlflowException

from pawpaw.pydantic.common import MLFlowConf, S3Conf
from pawpaw.pydantic.train_test import TestParams, TestSummary


class Tester:
    """Helper class for testing/evaluating a registered MLFlow model. It will save the
    evaluation data as MLFlow artifact, which can also be used for drift monitoring
    purpose later.
    """

    def __init__(
        self, model_name: str, model_version: str, model_uri: str,
        model_id: str | None = None
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.model_uri = model_uri
        self.model_id = model_id

        # Try to get the model id manually if possible
        # This is optional but nice to have for tracking later
        if not model_id and model_uri.startswith('models:/'):
            self.model_id = model_uri[len('models:/'):]
        else:
            warnings.warn(
                'Model id is not provided and can\'t be extracted, '
                'but this is useful for tracking purpose later'
            )

    def predict(
        self, params: TestParams, mlf_cfg: MLFlowConf, s3_cfg: S3Conf
    ) -> pl.DataFrame:
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)

        # Assume the test file is always located on S3
        df = pl.scan_csv(
            params.csv_dir + '/test.csv',
            storage_options = s3_cfg.model_dump()
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
        ])

        # TODO: Investigate why the prediction shape is 2D
        preds = preds.view(-1).numpy(force = True)
        # Add the stacked predictions as a single column
        df = df.insert_column(len(df.columns), pl.Series('Prediction', preds))
        # Clean back the cache dir
        cache_dir.cleanup()

        return df

    def run_evaluation(
        self, df: pl.DataFrame, params: TestParams, mlf_cfg: MLFlowConf
    ) -> TestSummary:
        """Run an evaluation for the model (via MLFlow) and log the model prediction
        result as an MLFlow artifact.
        """

        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
        mlflow.set_experiment(mlf_cfg.experiment_name)

        with mlflow.start_run():
            dataset = PandasDataset(
                df.to_pandas(),
                name = 'lakeFS',
                targets = 'Pawpularity',
                predictions = 'Prediction',
                source = HTTPDatasetSource(params.csv_dir),
                digest = params.data_commit_id[:8]
            )

            # log_input will only save the column schema and the source path/URL
            mlflow.log_input(dataset, context = params.context)
            # To save it as a file too, we need to call log_table separately
            # The saved file can be loaded for drift monitoring purpose later
            mlflow.log_table(df.to_pandas(), 'prediction.json')
            # Also log the params, just like when training
            mlflow.log_params(params.model_dump())

            result = mlflow.evaluate(
                data = dataset,
                model_type = 'regressor',
                model_id = self.model_id
            )

            # HACK: Type hint for the evaluation result
            if isinstance(result, EvaluationResult):
                pass

        return TestSummary(
            run_id = result.run_id,
            data_commit_id = params.data_commit_id,
            model_uri = self.model_uri,
            model_version = self.model_version,
            model_registry_name = self.model_name,
            metric = params.metric,
            metric_min = params.metric_min,
            metric_threshold = params.metric_threshold,
            score = result.metrics[params.metric]
        )

    @staticmethod
    def search_evaluation(
        ref_commit_id: str, summary: TestSummary, mlf_cfg: MLFlowConf
    ) -> str | None:
        """Search an existing evaluation run which match a certain criteria, such as
        the name of the metric used, and the specific commit id. If there are multiple
        runs that matches the criteria, only the run id with the best metric score will
        be returned.
        """

        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)

        # Find existing evaluation run with a specific commit id
        logged_runs = mlflow.search_runs(
            experiment_names = [mlf_cfg.experiment_name],
            filter_string = 'params.context = \'testing\' AND '
                f'params.data_commit_id = \'{ref_commit_id}\' AND '
                f'params.metric = \'{summary.metric}\'',
            order_by = [
                f'metrics.{summary.metric} {'ASC' if summary.metric_min else 'DESC'}'
            ],
            output_format = 'list'
        )

        if logged_runs:
            # Return the run id that has the best metric result
            return logged_runs[0].info.run_id

        return None

    @staticmethod
    def load_prediction(
        run_id: str, mlf_cfg: MLFlowConf, artifact_path: str = 'prediction.json',
        error_ok: bool = False
    ) -> pl.DataFrame | None:
        """Load the prediction result from an existing evaluation run, assuming the run
        id is already known beforehand. This is preferred over re-running the same
        evaluation again, which can be expensive and slow depending on the model.
        """

        if error_ok and not run_id:
            return None

        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)

        try:
            # Assumed to be logged when running the evaluation run
            df = mlflow.load_table(artifact_path, run_ids = [run_id])
        except MlflowException as err:
            if error_ok and err.error_code == 'RESOURCE_DOES_NOT_EXIST':
                return None

        return pl.from_pandas(df)