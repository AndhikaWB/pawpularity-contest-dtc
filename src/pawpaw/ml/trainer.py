import tempfile
import polars as pl

import mlflow
from mlflow.data.meta_dataset import MetaDataset
from mlflow.types import Schema, ColSpec, DataType
from mlflow.data.http_dataset_source import HTTPDatasetSource

import torch
from torch import nn
from torch.utils.data import DataLoader
from lightning import Fabric
import torchmetrics as tm

from pawpaw.pydantic.common import MLFlowConf
from pawpaw.ml.utils import LossMetric, EarlyStopping, QSave
from pawpaw.pydantic.train_test import TrainParams, TrainSummary


class Trainer:
    """Helper class to train a new PyTorch model for predicting pet pawpularity. It
    utilizes MLFlow to track the training process and log the training result.
    """

    def __init__(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def prep_training(self, params: TrainParams) -> TrainParams:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params.lr)
        self.criterion = torch.nn.BCELoss()

        self.metrics = {
            'bce': LossMetric(torch.nn.BCELoss),
            'rmse': tm.MeanSquaredError(squared = False)
        }

        self.cb = {
            'early_stop': EarlyStopping(
                monitor = 'val_bce',
                patience = params.patience,
                mode = 'min'
            )
        }

        # ----------

        # Save the optimizer and criterion name to log to MLFlow later
        # Fabric may change the class name later, so we save them early
        params.optimizer = self.optimizer.__class__.__name__
        params.criterion = self.criterion.__class__.__name__
        params.monitor = 'val_bce'
        params.monitor_min = True

        print('MLFlow parameters:', params.model_dump())

        # ----------

        # Initiate Lightning Fabric with GPU accelerator
        # Without this, we have to call "to_device" everywhere
        self.fabric = Fabric(accelerator = 'gpu')

        # Set all tensors on these objects to use GPU by wrapping them as Fabric classes
        # Once wrapped, the class name and some of its properties will change too
        # You can access the original class by adding ".module" or ".optimizer"
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.train_loader = self.fabric.setup_dataloaders(self.train_loader)
        self.val_loader = self.fabric.setup_dataloaders(self.val_loader)
        for key in self.metrics.keys():
            self.metrics[key] = self.fabric.setup_module(self.metrics[key])

        # Added optimizer and criterion name to params
        # You can reuse this later or just discard it
        return params

    def get_input_metadata(self, params: TrainParams):
        if not (params.csv_dir or params.data_commit_id):
            raise ValueError('Data source or commit id can\'t be empty')

        # Try to get the underlaying dataframe from the data loader
        df = getattr(self.train_loader.dataset, 'df', None)
        column_schemas = None

        if isinstance(df, pl.DataFrame):
            column_specs = []

            for i in range(len(df.columns)):
                # Convert Polars data type to Python
                dtype = df.dtypes[i].to_python()
                # Then convert again from Python to MLFlow data type
                # Note that not all data types are supported by MLFlow
                for val in DataType:
                    if val.to_python() == dtype:
                        dtype = val

                column_specs.append(
                    ColSpec(type = dtype, name = df.columns[i])
                )

            column_schemas = Schema(column_specs)

        return MetaDataset(
            HTTPDatasetSource(params.csv_dir),
            name = 'lakeFS',
            # If too long, an error will be raised
            digest = params.data_commit_id[:8],
            # If unsupported, an error will be raised
            schema = column_schemas
        )

    def start_training(self, params: TrainParams, mlf_cfg: MLFlowConf) -> str:
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
        mlflow.set_experiment(mlf_cfg.experiment_name)
        mlflow.enable_system_metrics_logging()

        with mlflow.start_run() as run:
            # Logs for current and all epochs
            logs = {}
            history = {}

            # We may not always have access to the run params later
            # But we can tie the dataset info to the model directly
            metadata = self.get_input_metadata(params)

            # Log things that won't change
            mlflow.log_params(params.model_dump())
            mlflow.log_input(metadata, context = params.context)
            print(f'Started run {run.info.run_name} ({run.info.run_id})')

            # Reset early stop state
            self.cb['early_stop'].on_train_begin()

            # ----------

            for epoch in range(1, params.epochs + 1):
                # ----------
                # Training epoch start

                self.model.train()

                for step, ds in enumerate(self.train_loader):
                    preds = self.model(ds['image'], ds['features'])
                    loss = self.criterion(preds, ds['target'])

                    # Backward pass
                    self.optimizer.zero_grad()
                    self.fabric.backward(loss)
                    # Update parameters (weights)
                    self.optimizer.step()

                    for name in self.metrics:
                        self.metrics[name](preds, ds['target'])

                # ----------
                # Training epoch end

                for name in self.metrics:
                    logs[name] = self.metrics[name].compute().item()
                    self.metrics[name].reset()

                # ----------
                # Validation epoch start

                self.model.eval()

                with torch.no_grad():
                    for step, ds in enumerate(self.val_loader):
                        preds = self.model(ds['image'], ds['features'])

                        for name in self.metrics:
                            self.metrics[name](preds, ds['target'])

                # ----------
                # Validation epoch end

                for name in self.metrics:
                    logs['val_' + name] = self.metrics[name].compute().item()
                    self.metrics[name].reset()

                # ----------
                # Misc at the end of each epoch

                logs['epoch'] = epoch
                self.cb['early_stop'].on_epoch_end(epoch, logs)
                print(f'End of epoch {epoch}: {logs}')

                # Append current epoch logs to history
                for name in logs.keys():
                    result = history.get(name, [])
                    history[name] = result + [ logs[name] ]

                # Export current model and history
                if self.cb['early_stop'].best_epoch == epoch:
                    print('Saving best model so far...')

                    # Unwrap Fabric class as normal PyTorch class
                    _model = self.model.module
                    _optimizer = self.optimizer.optimizer

                    model_info = mlflow.pytorch.log_model(
                        _model,
                        name = _model.__class__.__name__,
                        step = epoch,
                        conda_env = 'conda.yaml',
                        signature = mlflow.models.infer_signature(
                            model_input = {
                                'image': ds['image'].numpy(force = True),
                                'features': ds['features'].numpy(force = True)
                            },
                            model_output = preds.numpy(force = True)
                        )
                    )

                    # Local folder for temporarily storing other file artifacts
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        torch.save(_optimizer.state_dict(), tmp_dir + '/optimizer.pth')
                        QSave.save(str(_model), tmp_dir + '/model.txt')
                        QSave.save(history, tmp_dir + '/history.json')

                        # Copy file artifacts to MLFlow remote artifact folder
                        mlflow.log_artifacts(tmp_dir)

                    # If this is the best epoch, attach metrics with the model
                    # This is so we can filter and compare with other models later
                    mlflow.log_metrics(
                        logs, epoch, model_id = model_info.model_id, dataset = metadata
                    )
                else:
                    # If not the best epoch, don't attach metrics to the model
                    mlflow.log_metrics(logs, epoch, dataset = metadata)

                self.fabric.barrier()

                # Stop training if signaled by early stop
                if self.cb['early_stop'].stop_training:
                    print('Early stopping...')
                    # Append the best metrics at the end of training
                    # MLFlow only shows the latest epoch metrics by default
                    mlflow.log_metrics(self.cb['early_stop'].best_logs, epoch + 1)
                    break

        return run.info.run_id

    def get_best_model(
        self, run_id: str, params: TrainParams, mlf_cfg: MLFlowConf, delete_others: bool
    ) -> TrainSummary:
        if not (run_id or params.monitor):
            raise ValueError('Run id or metric to compare can\'t be empty')

        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)

        client = mlflow.MlflowClient()
        experiment_id = client.get_experiment_by_name(mlf_cfg.experiment_name)
        experiment_id = experiment_id.experiment_id

        # Sort models from the last run by a specific metric score
        # In our case, each model can only be tied with one epoch only
        # In other words, each epoch has a unique model and metrics
        logged_models = client.search_logged_models(
            experiment_ids = [ experiment_id ],
            filter_string = f'source_run_id = \'{run_id}\'',
            order_by = [
                {
                    'field_name': f'metrics.{params.monitor}',
                    'ascending': True if params.monitor_min else False
                }
            ]
        )

        if not logged_models:
            raise RuntimeError(f'No model is found under run id "{run_id}"')

        # Delete every models except the best model
        if delete_others:
            for i in logged_models[1:]:
                client.delete_logged_model(i.model_id)

        # Get the best model URI from this run
        best_model_uri = logged_models[0].model_uri
        # Also get the specific metric score tied to that model
        # TODO: Check behavior if the model is tied with more than 1 epoch
        for metric in reversed(logged_models[0].metrics):
            if metric.key == params.monitor:
                metric_value = metric.value

        return TrainSummary(
            run_id = run_id,
            data_commit_id = params.data_commit_id,
            model_uri = best_model_uri,
            metric = params.monitor,
            metric_min = params.monitor_min,
            score = metric_value
        )