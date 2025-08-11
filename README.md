# Pet Pawpularity Prediction

Ever wondered how cute your pet is compared to other people's pet? Say no more! Just upload the best/cutest photo of your pet and my model will predict the pawpularity score for you.

The original goal if this project is to ease the process of pet adoption, by using the ML model, volunteers can take the best possible photo of their pet, and receive feedback from the model so they can change the pose, etc. to increase the pawpularity score. 

However, this feedback based solution is currently not completed yet (especially the embedding part), so only the pawpularity score will be outputted for now.

## Data Availability

The data is part of this [Kaggle competition](www.kaggle.com/c/petfinder-pawpularity-score/data), and I already received permission from PetFinder.my to use it in this project (I sent them an email). Still, you need to download the image files from Kaggle by yourself. I'm not including the images because the repo size will get considerably bigger, and for the privacy of the people/pet included in the images.

However, the original CSV is already included in this repo, and has been renamed from `train.csv` to `data.csv` to avoid confusion with data from the preprocessing step later.

## Architecture

![](notes/stack.png)

## Stack

|Name|Description|
|-|-|
|JuiceFS|The default S3 object storage server for storing the preprocessed data, can be switched to MinIO or Amazon if you want. I use `boto3` in the code, so any server will work|
|lakeFS|Data versioning on the S3, more flexible than DVC in my opinion. My data is not really time series based so I use commit id as the unique identifier|
|PyTorch|The ML library used to train the model|
|MLFlow|Experiment tracking and model registry, and to store the model evaluation result, which can be used by NannyML to calculate the drift report|
|NannyML|The drift report calculator, mainly because I find the Evidently `dict` report has no standardized structure/type hint, and can be a pain to upload to database|
|PostgreSQL|Database for storing the drift report info and other services data (e.g. lakeFS)|
|Grafana|The dashboard to show visualization from PostgreSQL, only shows a simple drift and alert table for now (WIP)|
|Prefect|The workflow orchestrator, but can also work without it (you can just use `make` if you want)|
|Pydantic|Input validation, will get the required input values from the CLI args or environment variables|
|FastAPI|To serve the model and show a simple form to submit the prediction request|
|Streamlit|The alternative (and prettier) web app that can also be used to submit the prediction request|
|Ruff|Code linter and formatter|

## Usage

### First Time Setup

1. Install [Conda](https://anaconda.org/anaconda/conda) dependencies by running `make setup`
2. Prepare and run Docker compose using `make compose`
    - If it's your first time, Docker will download the required image files first (this may take a while)
    - Once everything is run, wait a bit until all the services are truly ready
3. The Docker compose already has everything pre-configured except for lakeFS
    - Visit [lakeFS setup](http://localhost:8000/setup) URL to generate username and password
    - Copy the username and password to `.env.dev` file
    - [Login](http://localhost:8000/auth/login) to lakeFS with that username and password
    - Create a new repo with this settings:
        - Repository id: `pawpaw-repo`
        - Default branch: `main` (default)
        - Storage namespace: `s3://lakefs/pawpaw-repo`

### Preprocess Data

1. Run the preprocess script (`make preprocess`) to upload the data to lakeFS repo
    - This will create a new commit to the lakeFS repo (containing random sample of the raw data)
    - Each time this is run, a new commit with randomized sample data will be created on lakeFS
    - We can treat these commits as monthly data, or data that are taken at different times from a streaming source

### Model Training and Evaluation

1. Run the evaluation script (`make evaluation`)
    - This will automatically run the training workflow (`make training`) in case there's no model yet, or if the current model test result (tested with data from the newest commit) is below the metric threshold. The threshold is set via environment variable (e.g. from `.env.dev` file)
    - At the end of training, the models will be evaluated and compared automatically. The evaluation is tied to the model and commit id, so if there are 2 models, there will be 2 evaluation results. Model with the best evaluation result will be marked with a version alias so we can easily load and serve it later
    - Because the evaluation process can be expensive, the evaluation result will be saved as MLFlow artifact. If the same model and commit id is detected, we will load the existing evaluation data instead. These data can also be loaded for drift monitoring purpose
    - If no drift report is generated after an evaluation, this is normal because we only have 1 commit so far and no reference/previous data yet. To generate a drift report, we need a minimum of 2 commits (excluding the initial dummy commit), so you may want to re-run the preprocess step once again
    - Make sure you also run at least one evaluation for each commit because the evaluation data is tied to the commit id. If a commit has no evaluation, it will be skipped because we can't find the evaluation data tied to that commit

### Model Serving and Web App

1. To serve the best model, simply run the predict script (`make serving`)
    - It will load the best model from MLFlow registry, via the model version alias we set during evaluation earlier
    - Currently it doesn't auto refresh the best model once you run it, I may improve it later
    - The port has been changed from 8000 to 8765 to avoid conflict with lakeFS
2. To send request to the served model, you can visit the built-in (and ugly) [test form](http://localhost:8765), or run a separate Streamlit server (`make webapp`)

## Learning Notes

Some of my learning notes, which contains the reasons why I choose some tools/softwares over the other are located on the "notes" folder. However, those notes may be outdated as I don't have time to revise them yet.