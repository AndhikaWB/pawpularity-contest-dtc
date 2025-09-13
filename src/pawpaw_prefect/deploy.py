import dotenv

from prefect import deploy
from prefect.docker import DockerImage

import pawpaw_prefect.flows.dummy as dummy
import pawpaw_prefect.flows.evaluation as evaluation
import pawpaw_prefect.flows.preprocess as preprocess
from pawpaw_prefect.pydantic_.common import DeployConf


if __name__ == '__main__':
    deploy_cfg = DeployConf()

    # Override the host names to match the service names in Docker compose
    env = dotenv.dotenv_values('infra/docker-deployment/override.env')

    deploy_params = dict(
        name = deploy_cfg.deployment_name,
        # The worker network also need to be configured (see README.md)
        # If the network isn't configured, the hosts may be unreachable
        work_pool_name = 'pool-1',
        job_variables = {
            'env': env
        }
    )

    deploy(
        dummy.main.to_deployment(deploy_params),
        evaluation.main.to_deployment(deploy_params),
        preprocess.main.to_deployment(deploy_params),
        image = DockerImage(
            name = f'{deploy_cfg.raw_deployment_name}/workflow',
            tag = 'latest',
            dockerfile = 'infra/docker-deployment/workflow.Dockerfile'
        ),
        # Prefect doesn't show any output when building
        # It's better to build the image by ourselves
        build = False
    )