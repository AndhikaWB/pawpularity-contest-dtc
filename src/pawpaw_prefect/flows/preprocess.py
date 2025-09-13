from prefect import task, flow
from pawpaw_prefect.utils.assets import lazy_materialize as lazym

import pawpaw.preprocess as p
from pawpaw_prefect.pydantic_.common import DeployConf


# Monkey patch everything to work with Prefect
p.pull_data = lazym('{{local_dir}}', asset_deps = ['{{remote_dir}}'], _f = p.pull_data)
p.preproc_data = lazym('{{out_dir}}', _f = p.preproc_data)
p.purge_remote_data = task(p.purge_remote_data)
p.upload_data = task(p.upload_data)
p.commit_data = lazym(
    'lakefs://{{repo_id}}/{{commit_id}}', output_as = 'commit_id', _f = p.commit_data
)


@flow(name = 'Data Preprocessing')
def main():
    # Don't call this function directly, use an outer function instead
    # Due to Python quirks, calling directly won't apply previous patches
    return p.main()


if __name__ == '__main__':
    # Serve flow to remote server without deployment
    # Without deployment, this machine will be the worker
    deploy_cfg = DeployConf()
    main.serve(deploy_cfg.deployment_name)