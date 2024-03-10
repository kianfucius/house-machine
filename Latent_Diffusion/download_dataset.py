"""
Downloads n songs from a given S3 bucket to a given local directory.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
from dotenv import load_dotenv

load_dotenv()


def download_file(bucket, file, directory):
    """
    Downloads a single file from an S3 bucket to a given local directory.

    Parameters
    ----------
    bucket : s3.Bucket
        S3 bucket object.
    file : dict
        Dictionary containing metadata about the file to download.
    directory : str
        Path to local directory where file will be downloaded
    """
    # Get the file name
    file_name = file["Key"].split("/")[-1]
    # Download the file
    bucket.download_file(file["Key"], os.path.join(directory, file_name))
    print(f"Downloaded {file_name}")


def main(num_songs: int, directory: str):
    """
    Downloads n songs from a given S3 bucket to a given local directory.

    Parameters
    ----------
    num_songs : int
        Number of songs to download.
    directory : str
        Path to local directory where files will be downloaded
    """
    # If directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # set up s3 client
    aws_id = os.getenv("AWS_ID", default="")
    aws_key = os.getenv("AWS_KEY", default="")
    bucket_name = os.getenv("SOURCE_BUCKET", default="")
    # Create an S3 resource object using the temporary credentials
    s3_client = boto3.resource(
        "s3",
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_key,
    )

    # List objects in the bucket. There are over 1k files, so we need to paginate
    # through the results.
    bucket = s3_client.Bucket(bucket_name)
    paginator = s3_client.meta.client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name)
    files_to_download = []
    for page in pages:
        files_to_download.extend(page["Contents"])

    # Download the files
    with ThreadPoolExecutor() as executor:
        for i in range(num_songs):
            executor.submit(download_file, bucket, files_to_download[i], directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download n songs from a given S3 bucket to a given local directory."
    )
    parser.add_argument("n", type=int, help="number of songs to download")
    parser.add_argument("local_dir", type=str, help="path to local directory")
    args = parser.parse_args()
    main(args.n, args.local_dir)
