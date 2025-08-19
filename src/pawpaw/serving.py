import dotenv
import uvicorn
import contextlib
from pathlib import Path
from pawpaw import logger
from typing import Annotated

from pawpaw.ml.server import Server
from pawpaw.pydantic_.common import MLFlowConf
from pawpaw.pydantic_.train_test import TestParams, MLFlowModel

from fastapi import FastAPI, Form, File
from fastapi.responses import HTMLResponse
from pawpaw.pydantic_.serve import ServeRequest, ServeResponse


class Runtime:
    params: TestParams = None
    model_registry: MLFlowModel = None
    mlflow_creds: MLFlowConf = None
    model: Server = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Load environment variable before starting
    dotenv.load_dotenv(
        '.env.prod' if Path('.env.prod').exists() else '.env.dev',
        override = False
    )

    # Other things to do before starting
    rt.params = TestParams()
    rt.model_registry = MLFlowModel()
    rt.mlflow_creds = MLFlowConf()

    logger.debug('Initiating and getting the best model version')
    rt.model = Server(rt.params, rt.model_registry, rt.mlflow_creds)
    logger.info(f'Loaded model version "{rt.model.model_info.version}"')

    yield

    # Before shutdown
    rt.model.cleanup()

rt = Runtime()
app = FastAPI(lifespan = lifespan)


@app.post('/predict')
async def predict(req: Annotated[ServeRequest, Form(), File()]) -> ServeResponse:
    return rt.model.predict(req)

@app.get('/reload')
async def reload():
    logger.debug('Refreshing and getting the best model version')
    rt.model.reload(rt.model_registry, rt.mlflow_creds)
    logger.info(f'Loaded model version "{rt.model.model_info.version}"')

    return {'detail': 'OK'}

@app.get('/')
async def main() -> HTMLResponse:
    html_content = """
        <!DOCTYPE html>
        <html>
        <body>
            <form action="predict" enctype="multipart/form-data" method="post">
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
    uvicorn.run(app, port = 8765)