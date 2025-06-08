import os
import polars as pl

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image


class PawDataset(Dataset):
    def __init__(self, df: pl.DataFrame, img_dir: str, img_transform = None, transform = None):
        self.df = df
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Image data
        img_col = self.df['Id'][index]
        img_path = os.path.join(self.img_dir, img_col)
        img_path = os.path.abspath(img_path + '.jpg')
        image = decode_image(img_path)

        # Target (must be 2D even if there's only 1 column)
        target = None
        if 'Pawpularity' in self.df.columns:
            target = self.df.select(pl.col('Pawpularity') / 100)
            target = target.to_torch()[index]

        # Tabular data (the rest of the columns)
        features = self.df.select(pl.exclude('Id', 'Pawpularity'))
        features = features.to_torch()[index]

        if self.img_transform:
            image = self.img_transform(image)

        if self.transform:
            image = self.transform(image)
            features = self.transform(features)
            target = None if not target else self.transform(target)

        # Return dict instead of tuple for clarity
        return {
            'image': image,
            'features': features,
            'target': target
        }


class PawModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.img_input = nn.Sequential(
            nn.LazyBatchNorm2d(),

            nn.LazyConv2d(16, 3, padding = 'same'),
            nn.MaxPool2d(2),

            nn.LazyConv2d(32, 3, padding = 'same'),
            nn.MaxPool2d(2),

            nn.LazyConv2d(64, 3, padding = 'same'),
            nn.MaxPool2d(2),

            nn.LazyConv2d(128, 3, padding = 'same'),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.LazyLinear(128)
        )

        self.feat_input = nn.LazyLinear(128)

        self.comb_input = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.LazyLinear(1),
            nn.Sigmoid()
        )

    def forward(self, img_inputs: Tensor, feat_inputs: Tensor) -> Tensor:
        out1 = self.img_input(img_inputs)
        out2 = self.feat_input(feat_inputs)

        # Combine the previous layer output
        out3 = torch.cat([out1, out2], dim = 1)
        out3 = self.comb_input(out3)

        return out3