"""
This is the main file to run Spotify recommendation generation.
"""

import argparse
import logging
import os
from typing import List

import pandas as pd
import spotdl
from scripts.hm_spot_dl import HmSpotDl
from scripts.utils import prep_env
from spotdl.types.options import DownloaderOptionalOptions
from spotdl.types.song import Song

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_all_songs_from_csv(input_file: str) -> List[str]:
    """
    Gets all track id's from our checkpoint csv's.

    Parameters
    ----------
    input_file : str
        input file to upload from

    Returns
    -------
    List[str] :
        list of the spotify track id's to upload
    """
    song_df = pd.read_csv(input_file, index_col=0)
    logger.info("Found %s songs", len(song_df))
    return song_df["track_id"].tolist()


def download_songs(
    track_ids: List[str], checkpoint: int = 0, credential_index: int = 0
):
    """
    Uses Spotipy to get songs to upload

    Parameters
    ----------
    track_ids : List[str]
        list of the spotify track id's to download and upload
    checkpoint: int
        index where we got rate limited

    Returns
    -------
    List[str] :
        list of the spotify track id's to upload
    """
    logger.info("Using id %s", credential_index)
    downloader_settings: DownloaderOptionalOptions = {
        "output": "tmp/tracks",
        "bitrate": "128k",
        "format": "flac",
    }
    hm_spot_dl = HmSpotDl(
        os.getenv("CLIENT_IDS", default="").split(",")[credential_index],
        os.getenv("CLIENT_SECRETS", default="").split(",")[credential_index],
        downloader_settings=downloader_settings,
    )

    num_tracks = len(track_ids)
    for index in range(checkpoint, len(track_ids)):
        logger.info(
            "Downloading song #%s of %s, id %s", index, num_tracks, track_ids[index]
        )
        try:
            song = Song.from_url(f"https://open.spotify.com/track/{track_ids[index]}")
            hm_spot_dl.download(song)
        except spotdl.types.song.SongError:
            logger.info("Song not found, skipping")


def main(checkpoint: int, input_file: str, credential_index: int = 0) -> None:
    """
    Run main function

    Parameters
    ----------
    checkpoint : int
        checkpoint to start at
    checkpoint_file : str
        checkpoint file to start at

    Returns
    -------
    None
    """
    prep_env(dev=False)

    # Get all songs from csv checkpoints
    track_ids: List[str] = get_all_songs_from_csv(input_file)
    # Begin downloading songs
    download_songs(track_ids, checkpoint=checkpoint, credential_index=credential_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="upload.py",
        description="Upload spotify songs to S3",
    )
    parser.add_argument(
        dest="credential_index",
        type=int,
        help="number of credentials to use",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="checkpoints/all_songs.csv",
        help="input file to upload from",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=0, help="checkpoint to start at"
    )
    args = parser.parse_args()
    main(args.checkpoint, args.input_file, args.credential_index)
