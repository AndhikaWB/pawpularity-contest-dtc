import os
import shlex
import contextlib

from dvc.cli import parse_args
from dvc.exceptions import DvcException


@contextlib.contextmanager
def change_dir(new_cwd: str = None):
    """Change directory temporarily."""

    old_cwd = os.getcwd()
    if new_cwd:
        os.chdir(new_cwd)

    try:
        yield
    finally:
        os.chdir(old_cwd)


class DvcExec():
    """Python wrapper for DVC CLI tool.

    Args:
        cwd (str, optional): Working directory for DVC.
            Defaults to None (current directory).
        quiet (bool, optional): Set log level to critical.
            Defaults to True.
    """

    def __init__(self, cwd: str = None, quiet: bool = True):
        self.cwd = cwd
        self.quiet = quiet

    def exec(self, argv: str, error: bool = False) -> int:
        """Execute DVC with specified arguments.

        Args:
            argv (str): Arguments to be passed to DVC.
            error (bool, optional): Raises DVC exception error.
                Defaults to False.

        Returns:
            int: Exit code raised by DVC.

        Raises:
            DvcException: When DVC exited with non-zero exit code.
                Only raised if `error` is set to True.
        """

        if self.cwd:
            argv = f'--cd "{self.cwd}" {argv}'
        if self.quiet:
            argv = f'-q {argv}'

        print(f'Executing "dvc {argv}"')

        args = parse_args(shlex.split(argv))
        try:
            # DVC may change cwd permanently after execution
            # The main Python script will also be affected
            # This fix will force to return to original cwd
            with change_dir(os.getcwd()):
                # Error can happen before returning stderr
                stderr = args.func(args).do_run()

            # In case no error but stderr is non-zero
            if stderr != 0:
                raise DvcException(
                    f'"dvc {argv}" returned with exit code {stderr}'
                )
        except Exception as e:
            stderr = 255
            # Prevent UnboundLocalError 
            exc = e

        if error and stderr != 0:
            raise exc

        return stderr