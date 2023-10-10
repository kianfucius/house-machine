"""
This is the main file to run Spotify recommendation generation.
"""
import argparse
import logging
import os
from typing import List

import pandas as pd
import utils
from scripts.spotify import Spotify

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def populate_playlist(checkpoint: int, checkpoint_file: str, upload: bool) -> List[str]:
    """
    Uses Spotipy to get songs to upload. Uses tracks in playlists as seeds,
    then generates recommendations per track.

    Parameters
    ----------
    upload : bool
        Whether we should upload to spotify

    Returns
    -------
    List[str] :
        list of the spotify track id's to upload
    """
    spotify = Spotify()

    # Get Playlist df
    if checkpoint is None:
        playlist_ids: List[str] = os.getenv("SEED_PLAYLISTS", default="").split(",")
        playlist_df: pd.DataFrame = spotify.get_recommendation_playlist_df(playlist_ids)
        playlist_df.to_csv("tmp/playlist_seed_original.csv")
    else:
        playlist_df = pd.read_csv(checkpoint_file, index_col=0)

    # Filter by length
    # playlist_df = Spotify.filter_playlist(playlist_df, 2, 4)

    # Get list of seed tracks for recommendations
    seed_ids: List = playlist_df["track_id"].tolist()

    # Create recommendation list
    recommendation_df: pd.DataFrame = spotify.get_recommendations_from_seeds(
        seed_ids, batch_size=5, limit=100, checkpoint=checkpoint
    )
    recommendation_df.to_csv("tmp/recommendations.csv")
    logger.info("Recommendation df length: %s", len(recommendation_df))

    # Append recommendations to original playlist
    new_playlist = (
        pd.concat([playlist_df, recommendation_df])
        .drop_duplicates("track_id")
        .reset_index(drop=True)
    )
    track_ids = new_playlist["track_id"].tolist()

    if upload:
        # Set up our output id's
        new_playlist_ids: List[str] = os.getenv("NEW_PLAYLISTS", default="").split(",")

        # Add songs to playlist
        spotify.append_to_playlists(new_playlist_ids, track_ids)

    return track_ids


def populate_from_csv(upload: bool = False) -> List[str]:
    """
    Uses Spotipy to upload songs from our checkpoint csv's.
    Not currently in use.

    Parameters
    ----------
    upload : bool
        Whether we should upload to spotify

    Returns
    -------
    List[str] :
        list of the spotify track id's to upload
    """

    # Get df containg seed and recommendation songs
    seed_df = pd.read_csv("checkpoints/data/playlist_seed_original.csv", index_col=0)
    recommendation_df = pd.read_csv("checkpoints/data/recommendations.csv", index_col=0)
    upload_df = (
        pd.concat([seed_df, recommendation_df])
        .drop_duplicates("track_id")
        .reset_index(drop=True)
    )
    upload_df = Spotify.filter_playlist(upload_df)
    upload_ids = upload_df["track_id"].tolist()

    # Upload to spotify
    if upload:
        spotify = Spotify()
        # Set up our output id's
        new_playlist_ids: List[str] = os.getenv("NEW_PLAYLISTS", default="").split(",")
        # Upload songs
        spotify.append_to_playlists(new_playlist_ids, upload_ids)

    logger.info("Found %s songs", len(upload_ids))
    return upload_ids


def main(checkpoint: int, checkpoint_file: str, upload: bool) -> None:
    """
    Run main function

    Parameters
    ----------
    checkpoint : int
        checkpoint to start at
    checkpoint_file : str
        checkpoint file to start at
    upload : bool
        whether to upload to spotify

    Returns
    -------
    None
    """
    utils.prep_env(dev=False)

    # Get tracks from seed playlist, then generate recommendations using tracks
    populate_playlist(checkpoint, checkpoint_file, upload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="recommend.py",
        description="Get spotify recommendations based on seed playlists",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=None, help="checkpoint to start at"
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="checkpoints/data/recommendation_checkpoint.csv",
        help="checkpoint file to start at",
    )
    parser.add_argument(
        "--upload",
        type=bool,
        default=True,
        help="whether to upload to spotify",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.checkpoint_file, args.upload)
