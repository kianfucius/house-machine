"""
This is the main file to run Spotify recommendation generation.
"""
import argparse
import logging
import os
from typing import List

import pandas as pd
import utils
from scripts.hm_spot_dl import HmSpotDl
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


def download_songs(track_ids: List[str], checkpoint: int = 0):
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
    downloader_settings: DownloaderOptionalOptions = {
        "output": "tmp/tracks",
        "bitrate": "128k",
        "format": "flac",
    }
    hm_spot_dl = HmSpotDl(
        os.getenv("CLIENT_ID", default=""),
        os.getenv("CLIENT_SECRET", default=""),
        downloader_settings=downloader_settings,
    )

    num_tracks = len(track_ids)
    logger.info("Made songs")
    for index in range(checkpoint, len(track_ids)):
        logger.info(
            "Downloading song #%s of %s, id %s", index, num_tracks, track_ids[index]
        )
        hm_spot_dl.download(
            Song.from_url(f"https://open.spotify.com/track/{track_ids[index]}")
        )


def main(input_file: str, checkpoint: int) -> None:
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
    utils.prep_env(dev=False)

    # Get all songs from csv checkpoints
    track_ids: List[str] = get_all_songs_from_csv(input_file)
    # Begin downloading songs
    download_songs(track_ids, checkpoint=checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="upload.py",
        description="Upload spotify songs to S3",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="checkpoints/data/all_songs.csv",
        help="input file to upload from",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=0, help="checkpoint to start at"
    )
    args = parser.parse_args()
    main(args.checkpoint, args.checkpoint_file)
