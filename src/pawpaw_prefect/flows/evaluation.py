from prefect import task, flow
from pawpaw_prefect.utils.assets import lazy_materialize as lazym

import pawpaw.evaluation as e
from pawpaw_prefect.flows.training import t
from pawpaw_prefect.pydantic_.common import DeployConf


# Monkey patch everything to work with Prefect
e.get_data_commit_id = task(e.get_data_commit_id)
e.get_best_model_version = lazym(
    # mlflow:// is not an actual URI, it's the MLFlow server location
    'mlflow://_/models/{{mlf_model.model_registry_name}}/versions/{{version}}',
    output_as = 'version', _f = e.get_best_model_version
)
e.evaluate_model = lazym(
    'mlflow://_/experiments/{{mlf_cfg.experiment_name}}/runs/{{summary.run_id}}',
    asset_deps = ['{{params.csv_dir}}'], output_as = 'summary', _f = e.evaluate_model
)
e.set_best_model_version = lazym(
    'mlflow://_/models/{{mlf_model.model_registry_name}}/versions/{{version}}',
    output_as = 'version', _f = e.set_best_model_version
)
e.generate_report = lazym(
    # Not a postgresql:// connection URI, the credentials are stripped here
    'postgresql://_/databases/{{report_cfg.database}}/tables/{{table_name}}',
    _f = e.generate_report
)

# Must be patched too as a subflow
e.training_run = t.run

@flow(name = 'Model Evaluation')
def main():
    # Don't call this function directly, use an outer function instead
    # Due to Python quirks, calling directly won't apply previous patches
    return e.main()


if __name__ == '__main__':
    # Serve flow to remote server without deployment
    # Without deployment, this machine will be the worker
    deploy_cfg = DeployConf()
    main.serve(deploy_cfg.deployment_name)