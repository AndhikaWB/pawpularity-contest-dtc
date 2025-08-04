import mlflow
import dotenv
import uvicorn
import tempfile
import polars as pl
from pathlib import Path

from _pydantic.common import MLFlowConf
from _pydantic.predict import PredictPost
from _pydantic.train_test import TestParams, MLFlowModel

from _ml.model import PawDataLoader

from fastapi import UploadFile, FastAPI, File, Form
from fastapi.responses import HTMLResponse

import torch
from lightning import Fabric


dotenv.load_dotenv(
    '.env.prod' if Path('.env.prod').exists()
    else '.env.dev'
)

class Prediction:
    def __init__(self, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf):
        model_uri = self.get_best_model_version(mlf_model, mlf_cfg)
        self.model = self.load_model(model_uri, mlf_cfg)
        self.temp_dir = tempfile.TemporaryDirectory()

    def get_best_model_version(
        self, mlf_model: MLFlowModel, mlf_cfg: MLFlowConf
    ) -> str:
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
        client = mlflow.MlflowClient()

        # If found, this model will be tested with the latest data later
        print(f'Getting the best model version from "{mlf_model.model_registry_name}"')
        alias = mlf_model.best_version_alias

        # Try getting the best model using a version alias
        version = client.get_model_version_by_alias(
            mlf_model.model_registry_name,
            alias = mlf_model.best_version_alias
        )

        print(f'Alias "{alias}" is tied to model version "{version.version}"')
        return version.source
    
    def load_model(self, model_uri: str, mlf_cfg: MLFlowConf):
        mlf_cfg.expose_auth_to_env()
        mlflow.set_tracking_uri(mlf_cfg.tracking_uri)

        fabric = Fabric(accelerator = 'gpu')
        model = mlflow.pytorch.load_model(model_uri)
        model = fabric.setup_module(model)

        return model

    def predict(self, df: pl.DataFrame, image: UploadFile, params: TestParams):
        self.model.eval()

        with open(f'{self.temp_dir.name}/image.jpg', 'wb') as f:
            f.write(image.file.read())

        loader = PawDataLoader(
            df,
            img_dir = self.temp_dir.name,
            is_train_data = False,
            batch_size = params.batch_size,
            img_size = params.img_size
        )

        fabric = Fabric(accelerator = 'gpu')
        loader = fabric.setup_dataloaders(loader)

        preds = torch.cat([
            self.model(ds['image'], ds['features'])
            for ds in loader
        ])

        # Can only be one data for now
        preds = preds.view(-1).numpy(force = True)[0]
        return preds


pred_params = TestParams()
model_registry = MLFlowModel()
mlflow_creds = MLFlowConf()

app = FastAPI()
model = Prediction(model_registry, mlflow_creds)


@app.post('/predict')
def predict(
    subject_focus: bool = Form(False),
    eyes: bool = Form(False),
    face: bool = Form(False),
    near: bool = Form(False),
    action: bool = Form(False),
    accessory: bool = Form(False),
    group: bool = Form(False),
    collage: bool = Form(False),
    human: bool = Form(False),
    occlusion: bool = Form(False),
    info: bool = Form(False),
    blur: bool = Form(False),
    image: UploadFile = File(...)
):

    df = pl.DataFrame([ {
        'Subject Focus': subject_focus,
        'Eyes': eyes,
        'Face': face,
        'Near': near,
        'Action': action,
        'Accessory': accessory,
        'Group': group,
        'Collage': collage,
        'Human': human,
        'Occlusion': occlusion,
        'Info': info,
        'Blur': blur
    } ])

    df.insert_column(1, pl.Series('Id', ['image']))
    result = model.predict(df, image, pred_params)
    return {'result': result}

@app.get('/')
def main():
    html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <form method="post" action="predict" enctype="multipart/form-data">
                Image Metadata<br><br>
                Focus: <input type="checkbox" name="subject_focus" value="True"><br>
                Eyes: <input type="checkbox" name="eyes" value="True"><br>
                Face: <input type="checkbox" name="face" value="True"><br>
                Near: <input type="checkbox" name="near" value="True"><br>
                Action: <input type="checkbox" name="action" value="True"><br>
                Accessory: <input type="checkbox" name="accessory" value="True"><br>
                Group: <input type="checkbox" name="group" value="True"><br>
                Collage: <input type="checkbox" name="collage" value="True"><br>
                Human: <input type="checkbox" name="human" value="True"><br>
                Occlusion: <input type="checkbox" name="occlusion" value="True"><br>
                Info: <input type="checkbox" name="info" value="True"><br>
                Blur: <input type="checkbox" name="blur" value="True"><br><br>

                <label for="image">Choose a JPG file to upload</label><br>
                <input type="file" id="image" name="image" required><br><br>
                <input type="submit" value="Predict">
            </form>
        </body>
        </html>
    """

    return HTMLResponse(html_content)

if __name__ == '__main__':
    uvicorn.run(app)