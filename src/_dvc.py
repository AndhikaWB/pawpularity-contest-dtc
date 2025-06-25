import os
import shlex
import contextlib

from dvc.cli import parse_args
from dvc.exceptions import DvcException


@contextlib.contextmanager
def change_dir(new_cwd: str = None):
    """Change directory temporarily."""

    old_cwd = os.getcwd()
    if new_cwd: os.chdir(new_cwd)

    try: yield
    finally: os.chdir(old_cwd)


def dvc_exec(argv: str, cwd: str = None, error: bool = False) -> int:
    """Python wrapper for `dvc` CLI app.

    Args:
        argv (str): Arguments to be passed to DVC.
        cwd (str, optional): Directory to execute DVC.
            Defaults to current directory.
        error (bool, optional): Raises DVC exception error.
            Defaults to False.

    Returns:
        int: Exit code raised by DVC.
    
    Raises:
        DvcException: When DVC exited with non-zero exit code.
    """

    with change_dir(cwd):
        args = parse_args(shlex.split(argv))
        stderr = args.func(args).do_run()

    if error and stderr != 0:
        raise DvcException(
            f'"dvc {argv}" returned with exit code {stderr}'
        )

    return stderr