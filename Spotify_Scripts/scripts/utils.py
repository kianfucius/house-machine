"""
Random util functions
"""
import os
import shutil

from dotenv import load_dotenv


def prep_env(dev=False):
    """
    Basic validation of environment

    Parameters
    ----------
    dev : boolean
        True if we don't want to remove stuff

    Returns
    -------
    None
    """
    load_dotenv()
    if dev:
        return
    temp_directory = "tmp"
    shutil.rmtree(temp_directory, ignore_errors=True)
    os.mkdir(temp_directory)
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    # Verify .env values exist
    env_vars = [
        # Spotify credentials
        "USERNAME",
        "CLIENT_IDS",
        "CLIENT_SECRETS",
        # Temp playlist storage and output
        "TEMP_PLAYLISTS",
        "NEW_PLAYLISTS",
        # Seeds
        "SEED_PLAYLISTS",
        # AWS credentials/locations
        "AWS_ID",
        "AWS_KEY",
        "BUCKET_NAME",
    ]
    for var in env_vars:
        buffer = os.getenv(var, default="")
        if not buffer:
            raise EnvironmentError(f"You need to specify {var} in .env")


def ceildiv(numerator: int, denominator: int) -> int:
    """
    Ceiling division, useful for indexing.

    Parameters
    ----------
    numerator : int
        Number to divide by
    denominator : int
        Number to divide with

    Returns
    -------
    int : the result
    """
    return -(numerator // -denominator)
