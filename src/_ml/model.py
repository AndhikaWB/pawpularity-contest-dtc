import io
import polars as pl
from PIL import Image
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor

import boto3
from _pydantic.common import S3Conf
from _s3.common import get_bucket_key


class PawDataset(Dataset):
    """The supported `Dataset` format for `PawModel`. For simplified preprocessing, just
    use `PawDataLoader` directly.
    """

    def __init__(
        self, df: pl.DataFrame, img_dir: str, img_transform = None, transform = None,
        s3_cfg: S3Conf | None = None, cache_dir: str | None = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.transform = transform

        if s3_cfg and self.img_dir.startswith('s3://'):
            # Extract the bucket and key from the full S3 path
            bucket, img_dir = get_bucket_key(self.img_dir)

            s3 = boto3.resource('s3', **s3_cfg.model_dump())
            self.bucket = s3.Bucket(bucket)
            # Image dir now becomes path after the bucket
            self.img_dir = img_dir
            # Cache dir for storing downloaded files
            self.cache_dir = cache_dir

    def __len__(self):
        return self.df.shape[0]

    def __decode_image(self, img_path: str) -> Tensor:
        # Not using S3 bucket
        if not hasattr(self, 'bucket'):
            return decode_image(img_path)

        # Using S3 bucket with no cache dir
        if not self.cache_dir:
            with io.BytesIO() as img_buff:
                self.bucket.download_fileobj(
                    Key = Path(img_path).as_posix(),
                    Fileobj = img_buff
                )

                with Image.open(img_buff) as buff:
                    return pil_to_tensor(buff)

        # Using S3 bucket with cache dir
        local_path = Path(self.cache_dir, img_path)
        if not local_path.exists():
            local_path.parent.mkdir(parents = True, exist_ok = True)

            self.bucket.download_file(
                Key = Path(img_path).as_posix(),
                Filename = str(local_path)
            )

            return decode_image(local_path)

    def __getitem__(self, index) -> dict[str, Tensor | None]:
        # Image data
        img_col = self.df['Id'][index]
        img_path = Path(self.img_dir, img_col + '.jpg')
        image = self.__decode_image(img_path)

        # Target (must be 2D even if only 1 column)
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


class PawDataLoader:
    """PyTorch `DataLoader` which already includes the recommended transformation
    functions and other necessities for `PawDataset`.

    Args:
        csv_path (str): The CSV file containing metadata of the images.
        img_dir (str): Directory to look up for the image files.
        is_train_data (bool): Whether to apply extra transformations intended for train
            data only.
        batch_size (int, optional): Batch size per iteration. Defaults to 64.
        img_size (tuple[int, int], optional): Resize all images to this size. Defaults
            to (128, 128).
        s3_cfg (S3Conf | None, optional): S3 credentials. Required if either the CSV
            path or the image directory is located on S3. Defaults to None.
        cache_dir (str | None, optional): Cache directory for storing downloaded files
            from S3. Can also work without it. Defaults to None.

    Returns:
        DataLoader: PyTorch `DataLoader` object.
    """

    def __new__(
        cls, csv_path: str | pl.DataFrame, img_dir: str, is_train_data: bool,
        batch_size: int = 64, img_size: tuple[int, int] = (128, 128),
        s3_cfg: S3Conf | None = None, cache_dir: str | None = None
    ) -> DataLoader:
        cls.csv_path = csv_path
        cls.img_dir = img_dir
        cls.is_train_data = is_train_data
        cls.batch_size = batch_size
        cls.img_size = img_size
        cls.s3_cfg = s3_cfg
        cls.cache_dir = cache_dir

        # This is just the DataLoader with preconfigured preprocessing
        # There's no need to inherit the DataLoader class directly
        # So we use __new__ instead of the __init__ function
        return cls.__get_dataloader()

    @classmethod
    def __prep_transformers(cls) -> tuple[nn.Module, nn.Module]:
        # Resize all images to have the same size
        img_transform = [ v2.Resize(cls.img_size) ]
        # Convert all data types to the same type
        transform = [ v2.ToDtype(torch.float32) ]

        # When training, apply random transformations
        # Otherwise, leave the image untouched
        if cls.is_train_data:
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

        return img_transform, transform

    @classmethod
    def __get_dataframe(cls) -> pl.DataFrame:
        # If already a dataframe, no need to do anything
        if type(cls.csv_path) == pl.DataFrame:
            return cls.csv_path

        # Source the CSV file from S3 when provided
        s3_cfg = cls.s3_cfg.model_dump() if cls.s3_cfg else None
        return pl.scan_csv(cls.csv_path, storage_options = s3_cfg).collect()
    
    @classmethod
    def __get_dataloader(cls) -> DataLoader:
        img_transform, transform = cls.__prep_transformers()

        dataset = PawDataset(
            df = cls.__get_dataframe(),
            img_dir = cls.img_dir,
            img_transform = v2.Compose(img_transform),
            transform = v2.Compose(transform),
            # Source the image files from S3 when provided
            s3_cfg = cls.s3_cfg,
            cache_dir = cls.cache_dir
        )

        return DataLoader(
            dataset,
            batch_size = cls.batch_size,
            shuffle = cls.is_train_data
        )


class PawModel(nn.Module):
    """PyTorch model for predicting pet pawpularity."""

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

    def forward(self, image: Tensor, features: Tensor) -> Tensor:
        out1 = self.img_input(image)
        out2 = self.feat_input(features)

        # Combine the previous layer output
        out3 = torch.cat([out1, out2], dim = 1)
        out3 = self.comb_input(out3)

        return out3
