import argparse
import os
from typing import List

import boto3
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SONG_EXTENSION = ".wav"


def download_file(
    song_list: List[str],
    index: int,
    # track_id: str,
    dest: str | os.PathLike,
    s3_client: boto3.client,
    bucket_name: str,
):
    """
    Downloads a song by its track_id if it is not already downloaded.
    Logs failures without halting the program.
    """
    track_id = song_list[index]
    dest_path = os.path.join(dest, f"{track_id}{SONG_EXTENSION}")
    if os.path.exists(dest_path):
        print(f"{index} Found local: {dest_path}")
        return

    try:
        s3_client.download_file(bucket_name, f"{track_id}{SONG_EXTENSION}", dest_path)
        print(f"{index} Downloaded {track_id}")
    except Exception as e:
        print(f"{index} Failed to download {track_id}: {e}")


def main(
    song_list_file: str | os.PathLike,
    num_songs: int,
    bucket_name: str,
    dest: str | os.PathLike,
):
    """
    Downloads n songs from a given S3 bucket to a given local directory.
    """
    if not os.path.exists(dest):
        os.makedirs(dest)

    # Read song list
    song_info = pd.read_csv(song_list_file, nrows=num_songs)
    song_list: List[str] = song_info["track_id"].tolist()

    start_idx = 13058
    song_list = song_list[start_idx:]
    num_songs = num_songs - start_idx

    aws_access_key_id = os.getenv("AWS_ID")
    aws_secret_access_key = os.getenv("AWS_KEY")
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    for song_index in tqdm(range(num_songs)):
        download_file(song_list, song_index, dest, s3_client, bucket_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download n songs from a given S3 bucket to a given local directory."
    )
    parser.add_argument(
        "song_list_file", type=str, help="Path to the CSV file of song IDs."
    )
    parser.add_argument("num_songs", type=int, help="Number of songs to download.")
    parser.add_argument("bucket_name", type=str, help="Name of the S3 bucket.")
    parser.add_argument(
        "dest", type=str, help="Destination directory for downloaded songs."
    )
    args = parser.parse_args()

    main(**vars(args))
