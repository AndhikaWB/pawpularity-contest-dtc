from prefect import flow
from pawpaw_prefect.utils.assets import lazy_materialize as lazym
from pawpaw_prefect.pydantic_.common import DeployConf


@lazym('{{out_dir}}', asset_deps = ['{{data_dir}}/raw'], output_as = 'out_dir')
def preprocess_data(data_dir: str, preproc_folder: str):
    print(f'Using data from "{data_dir}/raw"')
    return f'{data_dir}/{preproc_folder}'


@lazym('{{remote_dir}}', output_as = 'remote_dir')
def upload_data(local_dir: str):
    print(f'Uploading data in "{local_dir}"')
    return 's3://bucket/dir'


@flow(name = 'Dummy Test')
def main():
    # Simple flow for testing Prefect connection
    data_dir = preprocess_data('data', 'processed')
    upload_data(data_dir)


if __name__ == '__main__':
    # Serve flow to remote server without deployment
    # Without deployment, this machine will be the worker
    deploy_cfg = DeployConf()
    main.serve(deploy_cfg.deployment_name)