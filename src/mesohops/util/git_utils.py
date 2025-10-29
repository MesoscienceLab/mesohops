import os
import subprocess

__title__ = "Git Utilities"
__author__ = "A. Hartzell, Z. W. Freeman"
__version__ = "1.6"


def get_git_commit_hash():
    """
    Retrieves the current git commit hash of the repository.

    Returns
    -------
    1. commit_hash : str
                     The current git commit hash, or the git error message if there was
                     an error retrieving the hash.
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Run git command to get the current commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )

        if result.returncode == 0:
            # If the command was successful, return the commit hash
            return result.stdout.strip()
        else:
            # If the command failed, return the error message
            return result.stderr.strip()
    except FileNotFoundError:
        return "Git is not installed or not found in PATH."
