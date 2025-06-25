import os
import shutil
import mlflow
import polars as pl

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import torchmetrics as tm
from lightning.fabric import Fabric

from _model import PawDataset, PawModel
from _helper import EarlyStopping, LossWrapper, QSave
from _field import TrainSettings, TrainParams, TrainTags, MLFlowSettings

from prefect import task
from prefect.assets import materialize


def read_data(cfg: TrainParams) -> tuple:
    df = pl.read_csv(cfg.csv_path)
    df = df.sample(cfg.sample_size, shuffle = True, seed = cfg.seed)

    # Train-validation split (80/20)
    df_val = df.tail(int(0.2 * len(df)))
    df_train = df.head(len(df) - len(df_val))

    return df_train, df_val

def preprocess_data(df: pl.DataFrame, cfg: TrainParams, is_train_data: bool) -> DataLoader:
    # Resize all images to have the same size
    img_transform = [ v2.Resize(cfg.img_size) ]
    # Convert all data types to have the same type
    transform = [ v2.ToDtype(torch.float32) ]

    # When training, apply random transformations
    # Otherwise, leave the image untouched
    if is_train_data:
        img_transform += [
            v2.RandomChoice([
                v2.RandomAffine(
                    # 2D Rotation
                    degrees = [-180, 180],
                    # 3D rotation
                    shear = [-25, 25]
                ),
                v2.ColorJitter(
                    contrast = [0.9, 1.1],
                    saturation = [0.9, 1.1],
                    hue = [-0.1, 0.1]
                )
            ])
        ]

    # Pass the dataset to the dataloader
    loader = DataLoader(
        PawDataset(
            df,
            img_dir = cfg.img_dir,
            img_transform = v2.Compose(img_transform),
            transform = v2.Compose(transform)
        ),
        batch_size = cfg.batch_size,
        shuffle = True if is_train_data else False
    )

    return loader

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.prep_training(cfg)

    def prep_training(self, cfg: TrainParams):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = cfg.lr)
        self.criterion = torch.nn.BCELoss()

        self.metrics = {
            'bce': LossWrapper(torch.nn.BCELoss),
            'rmse': tm.MeanSquaredError(squared = False)
        }

        self.cb = {
            'early_stop': EarlyStopping(
                monitor = cfg.monitor,
                patience = cfg.patience,
                mode = 'min'
            )
        }

        # ----------

        # Fabric will change things so we should save some info before
        cfg.optimizer = self.optimizer.__class__.__name__
        cfg.criterion = self.criterion.__class__.__name__
        self.model_str = str(self.model)

        # ----------

        # Initiate Fabric to move all tensors to GPU
        # Without having to call "to_device" everywhere
        self.fabric = Fabric(accelerator = 'gpu')

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.train_loader = self.fabric.setup_dataloaders(self.train_loader)
        self.val_loader = self.fabric.setup_dataloaders(self.val_loader)
        for key in self.metrics.keys():
            self.metrics[key] = self.fabric.setup_module(self.metrics[key])

    def start_training(self, cfg: dict, tags: dict) -> str:
        with mlflow.start_run() as run:
            # Logs for current and all epochs
            logs = {}
            history = {}

            # Log things that won't change
            mlflow.set_tags(tags)
            mlflow.log_params(cfg)
            print(f'Started run {run.info.run_name} ({run.info.run_id})')

            # Reset early stop state
            self.cb['early_stop'].on_train_begin()

            # ----------

            for epoch in range(1, cfg['epochs'] + 1):
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

                self.cb['early_stop'].on_epoch_end(epoch, logs)

                # ----------
                # Misc at the end of each epoch

                logs['epoch'] = epoch
                print(f'End of epoch {epoch}: {logs}')

                # Append current epoch logs to history
                for name in logs.keys():
                    result = history.get(name, [])
                    history[name] = result + [ logs[name] ]

                # Export best model and history
                if self.cb['early_stop'].best_epoch == epoch:
                    print('Saving best model so far...')

                    mlflow.pytorch.log_model(
                        self.model,
                        # MLFlow artifact path is not local folder
                        artifact_path = 'model',
                        conda_env = 'conda.yaml',
                        signature = mlflow.models.infer_signature(
                            model_input = {
                                'img_inputs': ds['image'].numpy(force = True),
                                'feat_inputs': ds['features'].numpy(force = True)
                            },
                            model_output = preds.numpy(force = True)
                        )
                    )

                    # Local folder for temporarily storing other file artifacts
                    # The files will be copied to MLFlow artifact path after calling "log_artifacts"
                    shutil.rmtree(cfg['model_dir'], ignore_errors = True)
                    os.makedirs(cfg['model_dir'], exist_ok = True)

                    torch.save(self.optimizer.state_dict(), cfg['model_dir'] + '/optimizer.pth')
                    QSave.save(self.model_str, cfg['model_dir'] + '/model.txt')
                    QSave.save(history, cfg['model_dir'] + '/history.json')
                    mlflow.log_artifacts(cfg['model_dir'])

                    self.fabric.barrier()

                # Log things that may change on each epoch
                mlflow.log_metrics(logs, epoch)

                # Stop training if signaled by early stop
                if self.cb['early_stop'].stop_training:
                    print(f'Early stopping...')
                    # Append best metrics at the end of log
                    mlflow.log_metrics(
                        self.cb['early_stop'].best_logs,
                        epoch + 1
                    )
                    break

        print(f'Stopped run {run.info.run_name} ({run.info.run_id})')
        return run.info.run_id

def run(cfg: TrainParams, tags: TrainTags, mlf: MLFlowSettings):
    # Validate all input parameters
    cfg = TrainParams.model_validate(cfg)
    tags = TrainTags.model_validate(tags)
    mlf = MLFlowSettings.model_validate(mlf)

    print(f'{tags.author} is starting a new run for {mlf.exp_name}...')

    # Set MLFlow to track current experiment
    mlflow.set_tracking_uri(mlf.tracking_uri)
    mlflow.set_experiment(mlf.exp_name)
    mlflow.enable_system_metrics_logging()

    # Set seed for reproducible experiment
    Fabric.seed_everything(cfg.seed)

    df_train, df_val = read_data(cfg)
    train_loader = preprocess_data(df_train, cfg, is_train_data = True)
    val_loader = preprocess_data(df_val, cfg, is_train_data = False)

    model = PawModel()
    trainer = Trainer(model, train_loader, val_loader, cfg)
    run_id = trainer.start_training(cfg, tags, mlf.reg_model_name)

    # Register model from the last run
    mlflow.register_model(
        f'runs:/{run_id}/model',
        mlf.reg_model_name
    )

if __name__ == '__main__':
    # It will read parameters passed from CLI
    args = TrainSettings()

    # Only override tags if not passed from CLI
    if not args.tags:
        args.tags = TrainTags(
            author = 'Andhika',
            lib = 'PyTorch',
            model = 'CNN',
            ext = 'py'
        )

    # Use default MLFlow settings
    mlf = MLFlowSettings()

    run(args.params, args.tags, mlf)