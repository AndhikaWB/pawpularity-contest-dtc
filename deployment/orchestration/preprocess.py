import pawpaw.preprocess as preprocess
from prefect import task, flow


# Monkey patch everything to work with Prefect
preprocess.pull_data = task(preprocess.pull_data)
preprocess.preproc_data = task(preprocess.preproc_data)
preprocess.purge_remote_data = task(preprocess.purge_remote_data)
preprocess.upload_data = task(preprocess.upload_data)
preprocess.commit_data = task(preprocess.commit_data)

preprocess.run = flow(
    preprocess.run,
    name = 'Preprocess Data',
    log_prints = True
)


if __name__ == '__main__':
    preprocess.main()