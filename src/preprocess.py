import polars as pl
from tqdm import tqdm
from pathlib import Path
from random import randint
from shutil import copyfile
from datetime import datetime
from concurrent import futures

import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from _pydantic.common import S3Conf, LakeFSConf, ValidOrNone

import boto3
from _s3.lakefs import get_repo_branch, commit_branch
from _s3.common import download_dir, get_bucket_key, upload_dir


def pull_data(remote_dir: str, local_dir: str, s3_cfg: S3Conf | None = None) -> str:
    # If not an S3 URI, assume it's a local path
    if not remote_dir.startswith('s3://'):
        print(f'Using data from "{local_dir}"')
        return local_dir
    elif not s3_cfg:
        raise ValueError('S3 credentials can\'t be empty if using S3')

    print(f'Pulling data from "{remote_dir}" to "{local_dir}"')
    download_dir(remote_dir, local_dir, s3_cfg, replace = True)

    return local_dir


def preproc_data(
    in_dir: str, out_dir: str, sample: int = 2000, seed: int = 1337
) -> str:
    # TODO: Use stratified sampling if possible
    df = pl.read_csv(Path(in_dir, 'data.csv'))
    df = df.sample(sample, shuffle = True, seed = seed)

    # Use half of the samples as test data
    df_test = df.head(int(0.5 * sample))

    # The other half as train and val data (80/20)
    df_tmp = df.tail(len(df) - len(df_test))
    df_val = df_tmp.tail(int(0.2 * len(df_tmp)))
    df_train = df_tmp.head(len(df_tmp) - len(df_val))

    print(f'Saving preprocessed data to "{out_dir}"')
    Path(out_dir, 'images').mkdir(parents = True, exist_ok = True)

    img_files = [
        # Image files (source and destination)
        f'{dir}/images/' + df['Id'] + '.jpg'
        for dir in [in_dir, out_dir]
    ]

    with tqdm(total = len(img_files[0]) + 3) as prog_bar:
        # Write the CSV files first
        df_train.write_csv(Path(out_dir, 'train.csv'))
        df_val.write_csv(Path(out_dir, 'val.csv'))
        df_test.write_csv(Path(out_dir, 'test.csv'))

        prog_bar.update(3)

        with futures.ThreadPoolExecutor() as exec:
            # Loop through the copyfile function output (i)
            # Without looping the output, any error won't be raised
            # And besides, we need the loop to update the progress bar
            for i in exec.map(copyfile, img_files[0], img_files[1]):
                prog_bar.update(1)

    return out_dir


def purge_remote_data(remote_dir: str, s3_cfg: S3Conf):
    bucket, target_dir = get_bucket_key(remote_dir)
    s3 = boto3.resource('s3', **dict(s3_cfg))
    bucket = s3.Bucket(bucket)

    print(f'Purging data on "{remote_dir}"')

    # Prevent catching long file/folder name with the same prefix
    # E.g. this/example/dir/ and this/example/dirislonger/
    if not target_dir.endswith('/'):
        target_dir += '/'

    # This operation won't directly delete the files if you use lakeFS
    # This will only be treated as unsaved commit which delete files
    # To save it as real changes, we need to commit it later
    bucket.objects.filter(Prefix = target_dir).delete()


def upload_data(local_dir: str, remote_dir: str, s3_cfg: S3Conf):
    print(f'Uploading "{local_dir}" to "{remote_dir}"')
    upload_dir(local_dir, remote_dir, s3_cfg, replace = True)


def commit_data(repo_id: str, branch: str, lfs_cfg: LakeFSConf) -> str:
    print(f'Commiting changes to "{repo_id}/{branch}"')
    commit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    commit_id = commit_branch(
        repo_id,
        branch = branch,
        message = f'Auto-commit at {commit_time}',
        lfs_cfg = lfs_cfg
    )

    print(f'Saved changes as commit id "{commit_id}"')
    return commit_id


def run(
    source_dir: str, target_dir: str, source_creds: S3Conf | None,
    target_creds: LakeFSConf, seed: int
):
    # Download and preprocess data from a local or S3 storage
    local_dir = pull_data(source_dir, 'data/raw', source_creds)
    preproc_dir = preproc_data(local_dir, 'data/preprocessed', seed = seed)

    # Purge the existing remote data before uploading again
    purge_remote_data(target_dir, target_creds.as_s3())
    upload_data(preproc_dir, target_dir, target_creds.as_s3())

    # Save the changes by commiting it
    repo_id, branch = get_repo_branch(target_dir)
    commit_data(repo_id, branch, target_creds)

    return target_dir


if __name__ == '__main__':
    dotenv.load_dotenv(
        '.env.prod' if Path('.env.prod').exists()
        else '.env.dev'
    )

    class ParseArgs(BaseSettings):
        """Preprocess data to be used for model training or testing."""

        model_config = SettingsConfigDict(
            cli_parse_args = True,
            cli_kebab_case = True,
            validate_by_name = True
        )

        source_dir: str = Field(alias = 'RAW_DATA_SOURCE')
        target_dir: str = Field(alias = 'TRAIN_DATA_SOURCE')
        source_creds: S3Conf | None = Field(default_factory = ValidOrNone(S3Conf))
        target_creds: LakeFSConf = Field(default_factory = LakeFSConf)

        # We can simulate streaming/monthly data by using random seed
        # However, complete uniqueness can't guaranteed from each seed
        seed: int = Field(default_factory = lambda: randint(1, 9999))

    args = ParseArgs()

    run(
        args.source_dir, args.target_dir,
        args.source_creds, args.target_creds,
        args.seed
    )