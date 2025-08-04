import polars as pl
from datetime import datetime

import mlflow
from _ml.tester import Tester
from _pydantic.common import MLFlowConf
from _pydantic.train_test import TestSummary

import torch
import nannyml as nml
from nannyml.thresholds import ConstantThreshold
from _pydantic.report import ReportConf


class Reporter:
    def __init__(self, cur_commit_id: str, cur_df: pl.DataFrame):
        self.cur_commit_id = cur_commit_id
        self.cur_df = cur_df

    def get_reference_dataframe(
        self, ref_commit_id: str, summary: TestSummary, mlf_cfg: MLFlowConf
    ) -> pl.DataFrame:
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)

        # Find existing evaluation run with a specific commit id
        logged_runs = mlflow.search_runs(
            experiment_names = [mlf_cfg.experiment_name],
            filter_string = f'params.data_commit_id = \'{ref_commit_id}\' AND '
                f'params.metric = \'{summary.metric}\'',
            order_by = [
                f'metrics.{summary.metric} {'ASC' if summary.metric_min else 'DESC'}'
            ],
            output_format = 'list'
        )

        if logged_runs:
            # Try to load the prediction dataframe from that run
            df = Tester.load_prediction(logged_runs[0].info.run_id, mlf_cfg)
            return df
        
        return None

    def metric_converter(self, mlflow_metric: str) -> str:
        """Convert MLFlow metric to NannyML metric (if available)."""

        metrics = {
            'mean_absolute_error': 'mae',
            'mean_squared_error': 'mse',
            'root_mean_squared_error': 'rmse',
            'mean_absolute_percentage_error': 'mape'
        }

        return metrics[mlflow_metric]

    def generate_report(
        self, ref_commit_id: str, ref_df: pl.DataFrame, summary: TestSummary,
        chunk_size: int = 1000
    ) -> pl.DataFrame:
        if type(ref_df) != pl.DataFrame:
            raise ValueError('Reference data can\'t be empty')

        # Disable NannyML analytics
        nml.disable_usage_logging()

        cur_df = self.cur_df.to_pandas()
        ref_df = ref_df.to_pandas()

        # All columns except image id
        all_cols = [col for col in cur_df.columns if col != 'Id']
        # Target and prediction
        cont_cols = ['Pawpularity', 'Prediction']
        # Binary features, can be treated as categorical or continuous
        bin_cols = [col for col in all_cols if col not in cont_cols]

        # Performance metric for the prediction result (e.g. RMSE)
        metric = self.metric_converter(summary.metric)

        # After reading a bit, Jensen-Shannon seems perfect for me
        # It's not too sensitive like KS, but can be slow for big data
        drift_method = 'jensen_shannon'
        # Jensen-Shannon score is not the same as p-value
        # 0 means identical data and 1 means very different
        drift_threshold = 0.1

        # How many drifted chunks are considered safe, compared to total chunks
        # If you have 10 chunks and 6 of them drifted, then it's 0.6 ratio
        # Since 0.6 is more than 0.4, we can say that column/prediction drifted
        alert_ratio_threshold = 0.4

        calc = nml.UnivariateDriftCalculator(
            column_names = all_cols,
            treat_as_categorical = bin_cols,
            continuous_methods = [drift_method],
            categorical_methods = [drift_method],
            # Threshold to decide the column drift status (True if drifted)
            thresholds = {drift_method: ConstantThreshold(upper = drift_threshold)},
            # Each chunk has it's own drift score and drift status
            # We may want to use average or majority vote later
            chunk_size = chunk_size
        )

        calc.fit(ref_df)
        drift_result = calc.calculate(cur_df).to_df()

        calc = nml.PerformanceCalculator(
            metrics = [metric],
            problem_type = 'regression',
            y_pred = 'Prediction',
            y_true = 'Pawpularity',
            thresholds = {metric: ConstantThreshold(upper = summary.metric_threshold)},
            chunk_size = chunk_size
        )

        calc.fit(ref_df)
        perf_result = calc.calculate(cur_df).to_df()

        table = []
        current_time = datetime.now()

        for name in bin_cols + ['Prediction']:
            result = drift_result if name not in 'Prediction' else perf_result
            method = drift_method if name not in 'Prediction' else metric
            hierarchy = [name, method] if name not in 'Prediction' else [method]

            value_col = result[*hierarchy, 'value']
            # Threshold is the number, alert is the drift status (True or False)
            # Each chunk can have different alert, but still use the same threshold
            thres_col = result[*hierarchy, 'upper_threshold']
            alert_col = result[*hierarchy, 'alert']

            # How many chunks drifted (status = True), in ratio between 0 and 1
            # 1 means all chunks drifted, 0 means none drifted
            alert_ratio = alert_col.value_counts(normalize = True).get(True, 0)
            # How many chunks drifted (the actual count, not ratio)
            alert_count = alert_col.value_counts().get(True, 0)

            table.append({
                'datetime': current_time,
                'model_version_current': summary.model_version,
                # Commit id of the data (from lakeFS)
                'commit_id_current': self.cur_commit_id,
                # Reference for drift, uses previous commit by default
                'commit_id_reference': ref_commit_id,
                'column_name': name,
                'method': method,
                # The average drift/score/error value of all chunks
                'value_average': value_col.mean(),
                # Threshold is always constant, aggregation doesn't matter
                'value_threshold': thres_col.min(),
                'alert_count': alert_count,
                'alert_ratio': alert_ratio,
                'alert_ratio_threshold': alert_ratio_threshold,
                # Whether the alert ratio pass the threshold or not (True/False)
                'alert': alert_ratio >= alert_ratio_threshold
            })
        
        return pl.DataFrame(table)
    
    def write_report_to_db(
        self, df: pl.DataFrame, table_name: str, report_cfg: ReportConf
    ) -> int:
        affected_rows = df.write_database(
            table_name,
            if_table_exists = 'append',
            connection = report_cfg.postgresql_uri()
        )

        return affected_rows
