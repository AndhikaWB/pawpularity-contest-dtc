import polars as pl

from evidently import Dataset, DataDefinition, Report, Regression
from evidently.metrics import DriftedColumnsCount, ColumnCount, ValueDrift, RMSE
from evidently.tests import eq, lt

from _ml.tester import Tester

class EvidentlyReport:
    def get_reference_run():
        pass

    def mlflow_metric_converter(self, metric: str) -> object:
        """Convert MLFlow metric to Evidently metric class."""

        metrics = {
            'root_mean_squared_error': RMSE
        }

        return metrics[metric]

    def get_evidently_report(
        self, cur_df: pl.DataFrame, ref_df: pl.DataFrame | None, metric_name: str,
        threshold: float
    ):
        data_def = DataDefinition(
            id_column = 'Id',
            categorical_columns = [
                col for col in cur_df.columns
                if col not in ['Id', 'Pawpularity', 'Prediction']
            ],
            regression = [
                Regression(target = 'Pawpularity', prediction = 'Prediction')
            ]
        )

        # Current data and current best model
        cur_data = Dataset.from_pandas(
            cur_df.to_pandas(),
            data_definition = data_def
        )

        # Previous data and previous best model
        ref_data = Dataset.from_pandas(
            ref_df.to_pandas(),
            data_definition = data_def
        ) if ref_df else None

        # Convert MLFlow metric to Evidently metric
        metric = self.mlflow_metric_converter(metric_name)

        report = Report([
            DriftedColumnsCount(),
            ValueDrift(column = 'Prediction'),
            # Column count including id, pawpularity, and prediction
            ColumnCount(tests = [ eq(15) ]),
            # The metric value must be less than the threshold
            metric(tests = [ lt(threshold) ])
        ], include_tests = True)

        report = report.run(cur_data, ref_data)