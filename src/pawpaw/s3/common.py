from tqdm import tqdm
from concurrent import futures
from urllib.parse import urlparse
from pathlib import Path, PurePosixPath

import boto3
from botocore.exceptions import ClientError
from pawpaw.pydantic_.common import S3Conf


def download_dir(remote_dir: str, local_dir: str, s3_cfg: S3Conf, replace: bool = True):
    """Download directory contents from an S3 bucket (not including the directory
    itself).

    Args:
        remote_dir (str): The full S3 path (e.g. `s3://bucket/dir`).
        local_dir (str): Local directory to store the downloaded files.
        s3_cfg (S3Conf): The S3 credentials (secret key, etc.).
        replace (bool, optional): Replace existing file. Defaults to True.
    """

    # Get the bucket name and the directory after the bucket
    bucket, remote_dir = get_bucket_key(remote_dir)
    s3 = boto3.resource('s3', **s3_cfg.model_dump())
    bucket = s3.Bucket(bucket)

    with futures.ThreadPoolExecutor() as exec:
        results = []

        # Prevent catching long file/folder name with the same prefix
        # E.g. this/example/dir/ and this/example/dirislonger/
        remote_dir = Path(remote_dir).as_posix() + '/'

        for obj in bucket.objects.filter(Prefix = remote_dir):
            # Get relative path from the prefix, not the bucket
            remote_path = PurePosixPath(obj.key).relative_to(remote_dir)

            local_path = Path(local_dir, remote_path)
            local_path.parent.mkdir(parents = True, exist_ok = True)

            # If file exists, don't redownload
            if not replace and local_path.exists():
                continue

            results.append(
                exec.submit(
                    bucket.download_file,
                    Key = obj.key,
                    Filename = str(local_path)
                )
            )

        for i in tqdm(futures.as_completed(results), total = len(results)):
            # Check for error by accessing the result
            # Without this, any error will not be raised
            _ = i.result()


def upload_dir(local_dir: str, remote_dir: str, s3_cfg: S3Conf, replace: bool = True):
    """Upload directory contents to an S3 bucket (not including the directory itself).

    Args:
        local_dir (str): Local directory containing the files to upload.
        remote_dir (str): The full S3 path (e.g. `s3://bucket/dir`).
        s3_cfg (S3Conf): The required S3 credentials (secret key, etc.).
        replace (bool, optional): Replace existing file. Defaults to True. If set to
            False, it will check by making an extra API call per file.
    """

    # Get the bucket name and the directory after the bucket
    bucket, remote_dir = get_bucket_key(remote_dir)
    s3 = boto3.resource('s3', **s3_cfg.model_dump())
    bucket = s3.Bucket(bucket)

    # Nested function (closure) for faster operation later
    # Faster because we don't need to recreate the S3 object everytime
    def __upload_file(local_path: str, remote_path):
        if not replace:
            try:
                # Check file existence by accessing common property
                bucket.Object(remote_path).content_length
            except ClientError as err:
                # If the file doesn't exist, upload it after this
                if err.response['Error']['Code'] == '404':
                    pass
                # If unrelated client error, raise it
                else:
                    raise err
            # For any other error, also raise it
            except Exception as exc:
                raise exc
            # If the file exists, don't reupload
            else:
                return

        # Upload the file if it doesn't exist or if replace is True
        bucket.upload_file(
            Filename = local_path,
            Key = remote_path
        )

    with futures.ThreadPoolExecutor() as exec:
        results = []

        for dir, folders, files in Path(local_dir).walk():
            for file_name in files:
                local_path = Path(dir, file_name)

                remote_path = local_path.relative_to(local_dir)
                remote_path = PurePosixPath(remote_dir, remote_path)

                results.append(
                    exec.submit(
                        __upload_file,
                        local_path = str(local_path),
                        remote_path = str(remote_path)
                    )
                )

        for i in tqdm(futures.as_completed(results), total = len(results)):
            # Check for error by accessing the result
            # Without this, any error will not be raised
            _ = i.result()


def get_bucket_key(s3_path: str) -> tuple[str, str]:
    """Get the bucket name and key (path after bucket) from the full S3 path.

    Args:
        s3_path (str): S3 path (e.g. `s3://bucket/dir/file`).

    Raises:
        ValueError: If the bucket name can't be extracted.

    Returns:
        tuple[str, str]: Bucket name and key (key can be an empty string).
    """

    path = urlparse(s3_path, 's3')
    if not path.netloc:
        raise ValueError(f'Can\'t get bucket name from "{s3_path}"')

    # Remove the leading slash (root dir) from the path
    return path.netloc, path.path[1:]