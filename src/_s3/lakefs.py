from urllib.parse import urlparse
from pathlib import PurePosixPath

import lakefs
from lakefs.models import Commit
from lakefs.exceptions import NotFoundException
from _pydantic.common import LakeFSConf


def get_repo_branch(s3_path: str) -> tuple[str, str]:
    """Get the repo id and branch name from the full S3 path. Should only be used if
    you're using lakeFS.

    Args:
        s3_path (str): The full S3 path (e.g. `s3://repo_id/branch/dir`).

    Raises:
        ValueError: If the repo id or the branch name can't be extracted.

    Returns:
        tuple[str,str]: Repo id and branch name (or commit id).
    """

    path = urlparse(s3_path, 's3')
    if not path.netloc:
        raise ValueError(f'Can\'t get repo id from "{s3_path}"')
    
    # Remove the leading slash (root dir) from the path
    key = PurePosixPath(path.path[1:]).parts

    if len(key) == 0 or not key[0]:
        raise ValueError(f'Can\'t get branch name from "{s3_path}"')

    return path.netloc, key[0]


def replace_branch(s3_path: str, new_branch: str) -> str:
    """Replace the branch part of an S3 path with other branch or commit id. Should only
    be used if you're using lakeFS.

    Args:
        s3_path (str): S3 path (e.g. `s3://repo_id/branch/dir`).
        new_branch (str): String to replace the current branch, can be another branch
            name or an exact commit id.

    Raises:
        ValueError: If the repo id or the branch name can't be extracted, or if the new
            branch name is None/empty string.

    Returns:
        str: A new S3 path with the replaced branch.
    """

    if not new_branch:
        raise ValueError('The new branch name can\'t be empty.')

    path = urlparse(s3_path, 's3')
    if not path.netloc:
        raise ValueError(f'Can\'t get repo id from "{s3_path}"')
    
    # Remove the leading slash (root dir) from the path
    key = PurePosixPath(path.path[1:]).parts

    if len(key) == 0 or not key[0]:
        raise ValueError(f'Can\'t get branch name from "{s3_path}"')

    new_path = PurePosixPath('/', new_branch, *key[1:])
    return 's3://' + path.netloc + str(new_path)


def get_exact_commit(
    s3_path: str, lfs_cfg: LakeFSConf, relative: str | None = None,
    return_id: bool = True
) -> str | Commit | None:
    """Get the exact commit (defaults to latest commit) from a lakeFS branch.

    Args:
        s3_path (str): S3 path (e.g. `s3://repo_id/branch/dir`). Path after the branch
            will always be ignored.
        lfs_cfg (LakeFSConf): lakeFS credentials (secret key, etc.).
        relative (str | None, optional): Relative to the current one (just like Git).
            For example, `~3` means 3 commits before the current one. Defaults to None.
        return_id (bool, optional): Whether to return the whole `Commit` object or just
            the commit id. Defaults to True (commit id only).

    Returns:
        Any: The commit id, or `Commit` object, or None if the commit can't be found
    """

    client = lakefs.Client(**lfs_cfg.model_dump())

    # Extract the branch name or commit id from the full S3 path
    # E.g. s3://my-repo/main or s3://my-repo/64675c312d48be7e
    repo_id, branch = get_repo_branch(s3_path)
    branch = branch + relative if relative else branch
    ref = lakefs.Reference(repo_id, branch, client = client)

    try:
        commit = ref.get_commit()
    except NotFoundException:
        # Possible if we're using ~ or ^ as the branch relative
        # E.g. s3://my-repo/main~3 (3 commits before main)
        return None

    return commit.id if return_id else commit


def commit_branch(
    repo_id: str, branch: str, message: str, lfs_cfg: LakeFSConf
) -> str:
    """Commit all unsaved changes in a lakeFS branch.

    Args:
        repo_id (str): lakeFS repo id (may not be the same as the repo name).
        branch (str): Branch of the repo to commit the changes.
        message (str): Message to be associated with the commit.
        lfs_cfg (LakeFSConf): lakeFS credentials (secret key, etc.).

    Returns:
        str: Commit id of the pushed commit.
    """

    client = lakefs.Client(**lfs_cfg.model_dump())
    repo = lakefs.Repository(repo_id, client = client)

    commit = repo.branch(branch).commit(message = message)
    return commit.id