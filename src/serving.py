import dotenv
import uvicorn
from pathlib import Path

from _ml.server import Server
from _pydantic.common import MLFlowConf
from _pydantic.serve import ServeRequest, ServeResponse
from _pydantic.train_test import TestParams, MLFlowModel

from typing import Annotated
from fastapi import FastAPI, Form, File
from fastapi.responses import HTMLResponse


dotenv.load_dotenv(
    '.env.prod' if Path('.env.prod').exists()
    else '.env.dev'
)

params = TestParams()
model_registry = MLFlowModel()
mlflow_creds = MLFlowConf()

model = Server(params, model_registry, mlflow_creds)
app = FastAPI()

@app.post('/predict')
async def predict(req: Annotated[ServeRequest, Form(), File()]) -> ServeResponse:
    return model.predict(req)

@app.get('/reload')
async def reload():
    model.reload(model_registry, mlflow_creds)
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
    model.img_dir.cleanup()