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
    folder = 'tmp'
    shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)

    # Verify .env values exist
    env_vars = ['USERNAME', 'CLIENT_ID', 'CLIENT_SECRET', # Login stuff
                'TEMP_PLAYLISTS', 'NEW_PLAYLISTS', # Temp playlist storage and output
                'SEED_PLAYLISTS', 'SEED_ARTISTS', 'SEED_GENRES', # Seeds
                'AWS_ID', 'AWS_KEY', 'BUCKET_NAME'] # AWS credentials/locations
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
