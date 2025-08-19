import polars as pl
from datetime import datetime

import torch
import nannyml as nml
from nannyml.thresholds import ConstantThreshold

from pawpaw.pydantic_.train_test import TestSummary
from pawpaw.pydantic_.report import ReportSchema, ReportConf


class Reporter:
    """Helper class for generating drift report between current and reference data.
    
    Use `generate_report` to generate the report (as `DataFrame`), and upload the
    report to database with `write_report_to_db` if needed (e.g. to be used by Grafana).
    """

    def __init__(self, summary: TestSummary, df: pl.DataFrame):
        self.cur_df = df
        self.cur_run_id = summary.run_id
        self.cur_commit_id = summary.data_commit_id

        # Current and reference data must use the same metric
        self.metric = summary.metric
        self.metric_threshold = summary.metric_threshold

    @staticmethod
    def metric_converter(mlflow_metric: str) -> str:
        """Convert MLFlow metric to NannyML metric (if available)."""

        metrics = {
            'mean_absolute_error': 'mae',
            'mean_squared_error': 'mse',
            'root_mean_squared_error': 'rmse',
            'mean_absolute_percentage_error': 'mape'
        }

        return metrics[mlflow_metric]

    def generate_report(
        self, ref_run_id: str, ref_commit_id: str, ref_df: pl.DataFrame,
        drift_threshold: float = 0.1, alert_ratio_threshold: float = 0.3,
        chunk_size: int = 1000
    ) -> pl.DataFrame:
        """Generate drift report between current data and reference data.

        Args:
            ref_run_id (str): MLFlow run id associated with the reference data.
            ref_commit_id (str): Data commit id associated with the reference data.
            ref_df (pl.DataFrame): Reference data to be compared with the current data.
            drift_threshold (float, optional): How many differences between current and
                reference data to be considered a drift (only used for data drift, not
                metric drift). Uses a value between 0 and 1, 0 means identical data and
                1 means total difference. Defaults to 0.1.
            alert_ratio_threshold (float, optional): How many share of drifted chunks to
                finally announce that the whole column/data, has in fact, drifted. Used
                for both data and metric drift, and has a value between 0 and 1.
                Defaults to 0.3.
            chunk_size (int, optional): Used to divide the data into multiple chunks.
                Defaults to 1000 rows. Each chunk will have it's own drift and alert
                status, refer to the `alert_ratio_threshold` above for how the final
                decision is made. 

        Raises:
            ValueError: If there's no reference data or commit id (so we can't calculate
                the drift).

        Returns:
            pl.DataFrame: Dataframe containing info about the drift and alert status,
                the metric name and drift method used, and other useful info. Can be
                uploaded to a database later.
        """

        if not isinstance(ref_df, pl.DataFrame) or not ref_commit_id:
            raise ValueError('Reference data or commit id can\'t be empty')

        # Disable NannyML analytics
        nml.disable_usage_logging()

        cur_df = self.cur_df.to_pandas()
        ref_df = ref_df.to_pandas()

        # All columns except image id
        all_cols = [col for col in self.cur_df.columns if col != 'Id']
        # Target and prediction
        cont_cols = ['Pawpularity', 'Prediction']
        # Binary features, can be treated as categorical or continuous
        bin_cols = [col for col in all_cols if col not in cont_cols]

        # Used only for data drift, for performance metric please see below
        # JS is not too sensitive like KS, but can be slow for big data
        drift_method = 'jensen_shannon'
        # Performance metric for the prediction result (e.g. RMSE)
        metric = self.metric_converter(self.metric)

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
            thresholds = {metric: ConstantThreshold(upper = self.metric_threshold)},
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
            safe_count = alert_col.value_counts().get(False, 0)

            table.append(
                ReportSchema(
                    time = current_time,
                    # MLFlow run id containing the evaluation data
                    # You can trace the model URI, etc. from this run id
                    run_id_current = self.cur_run_id,
                    run_id_reference = ref_run_id,
                    # Data commit id used in the evaluation run
                    commit_id_current = self.cur_commit_id,
                    commit_id_reference = ref_commit_id,
                    column_name = name,
                    method = method,
                    # The average drift/score/error value of all chunks
                    value_average = value_col.mean(),
                    # Value threshold to be considered dangerous
                    value_threshold = thres_col.item(),
                    # The number of safe/alerting chunks
                    safe_count = safe_count,
                    alert_count = alert_count,
                    alert_ratio = alert_ratio,
                    alert_ratio_threshold = alert_ratio_threshold,
                    # Whether the alert ratio pass the threshold or not
                    alert = alert_ratio >= alert_ratio_threshold
                ).model_dump()
            )

        return pl.DataFrame(table)
    
    @staticmethod
    def write_report_to_db(
        df: pl.DataFrame, table_name: str, report_cfg: ReportConf
    ) -> int:
        affected_rows = df.write_database(
            table_name,
            if_table_exists = 'append',
            connection = report_cfg.postgresql_uri()
        )

        return affected_rows
