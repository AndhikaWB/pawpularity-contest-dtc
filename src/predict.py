import polars as pl
import argparse
import mlflow

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from lightning.fabric import Fabric

from _model import PawDataset

def get_model_uri(tracking_uri: str, reg_model_name: str, alias: str) -> str:
    client = mlflow.MlflowClient(tracking_uri)

    # Get model info by version alias
    model = client.get_model_version_by_alias(reg_model_name, alias)
    model_uri = model.source

    # Return the model URI to load later
    if not model_uri:
        raise ValueError('Model URI is empty, can\'t load as proper model!')
    return model_uri

def read_dataframe(cfg: dict) -> tuple:
    df = pl.read_csv(cfg['csv_path'])
    df = df.sample(cfg['sample_size'], shuffle = True, seed = cfg['seed'])

    return df

def preprocess_data(df: pl.DataFrame, cfg: dict) -> DataLoader:
    # Resize all images to have the same size
    img_transform = [ v2.Resize(cfg['img_size']) ]
    # Convert all data types to have the same type
    transform = [ v2.ToDtype(torch.float32) ]

    # Pass the dataset to the dataloader
    loader = DataLoader(
        PawDataset(
            df,
            img_dir = cfg['img_dir'],
            img_transform = v2.Compose(img_transform),
            transform = v2.Compose(transform)
        ),
        batch_size = cfg['batch_size'],
        shuffle = False
    )

    return loader

def predict():
    pass

def run(cfg: dict, tags: dict):
    reg_model_name = 'dev.pawpaw-model'

    # Set MLFlow to track current experiment
    mlflow.set_tracking_uri('http://localhost:5000')

    df = read_dataframe(cfg)
    loader = preprocess_data(df, cfg)

    model_uri = get_model(reg_model_name, 'best')
    model = mlflow.pytorch.load_model(model_uri)

    model.eval()

    fabric = Fabric(accelerator = 'gpu')
    loader = fabric.setup_dataloaders(loader)

    preds_list = []

    for step, ds in enumerate(loader):
        preds = model(ds['image'], ds['features'])
        preds_list.append(preds)

    preds = torch.cat(preds_list)
    print(preds.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Train model to predict pet popularity'
    )

    parser.add_argument('--csv-path', type = str, default = '../data/train.csv')
    parser.add_argument('--img-dir', type = str, default = '../data/train')
    parser.add_argument('--sample-size', type = int, default = 1000)
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--seed', type = int, default = 444)
    parser.add_argument('--save-model-dir', type = str, default = 'model')
    parser.add_argument('--developer-name', type = str, default = 'YourName')
    args = parser.parse_args()

    cfg = {
        'csv_path': args.csv_path,
        'img_dir': args.img_dir,
        'model_dir': args.save_model_dir,
        'sample_size': args.sample_size,
        'img_size': (128, 128),
        'seed': args.seed,
        'lr': 0.001,
        'batch_size': 64,
        'epochs': args.epochs,
        'monitor': 'val_bce',
        'patience': 5
    }

    tags = {
        'developer': args.developer_name,
        'model': 'PyTorch',
        'format': 'ipynb',
        'type': 'CNN'
    }

    run(cfg, tags)