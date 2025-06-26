import os
import polars as pl
import lakefs

from _dvc import DvcExec
from _field import LakeFSSettings

def pull_data(source_uri: str, local_dir: str) -> str:
    # Pretend that we're pulling data from a streaming source
    # This should be done monthly to monitor the model performance
    print(f'Pulling streaming data from {source_uri}...')

    # Save the data to local system to preprocess later   
    # This should be the CSV file plus the image files
    print(f'Saving pulled data to {local_dir}...')
    os.makedirs(local_dir, exist_ok = True)

    return local_dir

def preprocess(local_dir: str, preproc_dir: str, lakefs_opt = LakeFSSettings) -> str:
    df = pl.read_csv(f'{local_dir}/data.csv')

    # Take only 2000 random samples
    df = df.sample(2000, shuffle = True, seed = 1337)

    # The model is tested with this data monthly
    df_test = df.head(1000)

    # Only used if we get a bad score on test data
    df_tmp = df.tail(1000)
    df_val = df_tmp.tail(int(0.2 * len(df_tmp)))
    df_train = df_tmp.head(len(df_tmp) - len(df_val))

    print(f'Saving preprocessed data to {preproc_dir}...')
    os.makedirs(preproc_dir, exist_ok = True)

    df_train.write_csv(f'{preproc_dir}/train.csv')
    df_val.write_csv(f'{preproc_dir}/val.csv')
    df_test.write_csv(f'{preproc_dir}/test.csv')

    return preproc_dir

def commit_data(preproc_dir: str, remote_dir: str) -> str:
    dvc = DvcExec(cwd = preproc_dir)
    dvc.exec('init --no-scm -f')
    dvc.exec(f'remote add pawpaw {remote_dir} -f')
    dvc.exec('add *.csv --glob', error = True)

    print(f'Commiting local changes in {preproc_dir}...')
    dvc.exec('commit')

    print(f'Pushing new data to {preproc_dir}...')
    dvc.exec('push -r pawpaw')

def run():
    local_dir = pull_data('s3://whatever', 'data/raw')
    preproc_dir = preprocess(local_dir, 'data/preprocessed')

    remote_dir = commit_data(preproc_dir, 's3://dvc/pawpaw')


if __name__ == '__main__':
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'admin123'
    os.environ['AWS_ENDPOINT_URL'] = 'http://localhost:9000'

    run()