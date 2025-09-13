from prefect import task, flow
import pawpaw.training as t
from pawpaw_prefect.utils.assets import lazy_materialize as lazym


# Monkey patch everything to work with Prefect
t.get_data_commit_id = task(t.get_data_commit_id)
t.model_training = lazym(
    # mlflow:// is not an actual URI, it's the MLFlow server location
    'mlflow://_/experiments/{{mlf_cfg.experiment_name}}/models/{{model.model_id}}',
    asset_deps = ['{{params.csv_dir}}'], output_as = 'model', _f = t.model_training
)
t.register_model = lazym(
    'mlflow://_/models/{{mlf_model.model_registry_name}}/versions/{{version}}',
    output_as = 'version', _f = t.register_model
)


# Will be called from the evaluation script
@flow(name = 'Model Training')
def run(*args, **kwargs):
    # Don't call this function directly, use an outer function instead
    # Due to Python quirks, calling directly won't apply previous patches
    return t.run(*args, **kwargs)