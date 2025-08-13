import uuid
import tempfile
import polars as pl

import torch
from lightning import Fabric
from pawpaw.ml.model import PawDataLoader

import mlflow
from pawpaw.pydantic.common import MLFlowConf
from pawpaw.pydantic.train_test import TestParams, MLFlowModel

from pawpaw.pydantic.serve import ServeRequest, ServeResponse, ModelInfo


class Server:
    """Helper class to easily load the best model from MLFlow, which then can be used
    for predicting pet pawpularity. The loaded model can be served via FastAPI or
    similar libraries.
    """

    def __init__(self, params: TestParams, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf):
        # To configure image size and batch size
        self.params = params
        # Directory for storing uploaded images
        self.img_dir = tempfile.TemporaryDirectory()
        # Load the model automatically
        self.reload(mlf_model, mlf_cfg)

    def __get_best_model_version(
        self, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf
    ) -> ModelInfo:
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
        client = mlflow.MlflowClient()

        print(f'Getting the best model version from "{mlf_model.model_registry_name}"')
        alias = mlf_model.best_version_alias

        # Try getting the best model using a version alias
        # If we don't have it yet, an error will be raised
        version = client.get_model_version_by_alias(
            mlf_model.model_registry_name,
            alias = alias
        )

        print(f'Alias "{alias}" is tied to model version "{version.version}"')

        return ModelInfo(
            source = version.source,
            version = version.version,
            variant = version.tags['variant']
        )
    
    def __load_model(self, model_uri: str, mlf_cfg: MLFlowConf):
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)

        fabric = Fabric(accelerator = 'gpu')
        model = mlflow.pytorch.load_model(model_uri)
        model = fabric.setup_module(model)
        model.eval()

        return model

    def reload(self, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf):
        """Reload the served model using an optimistic approach (by assuming no error
        will ever occur).
        """

        # There might be a very little downtime when switching the model here
        self.model_info = self.__get_best_model_version(mlf_model, mlf_cfg)
        self.model = self.__load_model(self.model_info.source, mlf_cfg)

    def predict(self, req: ServeRequest) -> ServeResponse:
        # Save the uploaded image in a directory
        file_name = uuid.uuid4().hex
        with open(f'{self.img_dir.name}/{file_name}.jpg', 'wb') as f:
            f.write(req.image)

        # Dump the image features as dataframe
        df = pl.DataFrame([ req.dump_form() ])
        df.insert_column(0, pl.Series('Id', [file_name]))

        loader = PawDataLoader(
            df,
            img_dir = self.img_dir.name,
            is_train_data = False,
            batch_size = self.params.batch_size,
            img_size = self.params.img_size
        )

        fabric = Fabric(accelerator = 'gpu')
        loader = fabric.setup_dataloaders(loader)

        preds = torch.cat([
            # Prediction result only has range between 0 and 1
            # We multiply by 100 to get the real pawpularity
            self.model(ds['image'], ds['features']) * 100
            for ds in loader
        ])

        # Convert to Python native type instead of Numpy
        # So that we can serialize it as JSON response
        preds = preds.view(-1).tolist()
        # There can only one row/image for now
        preds = preds[0] if len(preds) == 1 else preds

        return ServeResponse(
            model = self.model_info,
            result = preds
        )