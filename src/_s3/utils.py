from tqdm import tqdm
from concurrent import futures
from urllib.parse import urlparse
from pathlib import Path, PurePosixPath

import boto3
from _pydantic.shared import S3Conf
from botocore.exceptions import ClientError


def download_dir(remote_dir: str, local_dir: str, s3_cfg: S3Conf, replace: bool = True):
    """Download directory contents from an S3 bucket (not including the directory
    itself).

    Args:
        remote_dir (str): The full S3 path (e.g. `s3://bucket/dir`).
        local_dir (str): Local directory to save the download.
        s3_cfg (S3Conf): The required S3 credentials (secret key, etc.).
        replace (bool, optional): Replace existing file. Defaults to True.
    """

    # Treat remote directory as path after the bucket name
    bucket, remote_dir = get_bucket_key(remote_dir)
    s3 = boto3.resource('s3', **dict(s3_cfg))
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
        local_dir (str): Local directory to upload.
        remote_dir (str): The full S3 path (e.g. `s3://bucket/dir`).
        s3_cfg (S3Conf): The required S3 credentials (secret key, etc.).
        replace (bool, optional): Replace existing file. Defaults to True. If set to
            False, it will check by making an extra API call per file.
    """

    # Treat remote directory as path after the bucket name
    bucket, remote_dir = get_bucket_key(remote_dir)
    s3 = boto3.resource('s3', **dict(s3_cfg))
    bucket = s3.Bucket(bucket)

    # Create a closure for faster operation later
    # Rather than recreating the S3 resource everytime
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

        # Upload if replace is true or file doesn't exist
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
    """Get the bucket name and key (path after bucket) from a full S3 path.

    Args:
        s3_path (str): S3 path (e.g. `s3://bucket/dir/file`).

    Raises:
        ValueError: If it fails to get the bucket name.

    Returns:
        tuple[str,str]: Bucket name and key (key can be an empty string).
    """

    path = urlparse(s3_path, 's3')
    if not path.netloc:
        raise ValueError(f'Can\'t get bucket name from "{s3_path}"')

    return path.netloc, path.path[1:]


def get_repo_branch(s3_path: str) -> tuple[str, str]:
    """Get the repo id and branch name from a full S3 path. Should only be used if
    you're using lakeFS.

    Args:
        s3_path (str): S3 path (e.g. `s3://repo_id/branch/dir`).

    Raises:
        ValueError: If it fails to get the repo id or the branch name.

    Returns:
        tuple[str,str]: Repo id and branch name.
    """

    path = urlparse(s3_path, 's3')
    if not path.netloc:
        raise ValueError(f'Can\'t get repo id from "{s3_path}"')
    
    branch = PurePosixPath(path.path[1:]).parts
    if len(branch) == 0 or not branch[0]:
        raise ValueError(f'Can\'t get branch name from "{s3_path}"')

    return path.netloc, branch[0]
