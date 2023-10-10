import asyncio
import datetime
import json
import logging
import os
import traceback
from pathlib import Path
from typing import List, Optional, Tuple
from typing_extensions import override

import boto3
from botocore.exceptions import ClientError
from spotdl._version import __version__
from spotdl.download.downloader import (
    SPONSOR_BLOCK_CATEGORIES,
    Downloader,
    DownloaderError,
)
from spotdl.providers.audio import AudioProvider
from spotdl.types.song import Song
from spotdl.utils.config import get_errors_path, get_temp_path
from spotdl.utils.ffmpeg import FFmpegError, convert
from spotdl.utils.formatter import create_file_name
from spotdl.utils.m3u import gen_m3u_files
from spotdl.utils.metadata import MetadataError, embed_metadata
from spotdl.utils.search import reinit_song, songs_from_albums
from syncedlyrics import search as syncedlyrics_search
from syncedlyrics.utils import is_lrc_valid, save_lrc_file
from yt_dlp.postprocessor.modify_chapters import ModifyChaptersPP
from yt_dlp.postprocessor.sponsorblock import SponsorBlockPP

logger = logging.getLogger(__name__)


class HMDownloader(Downloader):
    """
    House-Machine's version of SpotDL's Downloader. We had to replace a few functions to
    enable pushing to S3.
    Downloader class, this is where all the downloading pre/post processing happens etc.
    It handles the downloading/moving songs, multithreading, metadata embedding etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_s3_client()

    def init_s3_client(self) -> None:
        """
        Initializes the Boto3 S3 client

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        aws_id = os.getenv("AWS_ID", default="")
        aws_key = os.getenv("AWS_KEY", default="")
        self.bucket_name = os.getenv("BUCKET_NAME")
        # Create an S3 resource object using the temporary credentials
        self.s3_client = boto3.resource(
            "s3", aws_access_key_id=aws_id, aws_secret_access_key=aws_key
        )

    def check_if_song_exists(self, youtube_id) -> bool:
        """ "
        Checks if a song already exists in our S3 bucket

        Parameters
        ----------
        youtube_id : string
            The Youtube ID the song. NOT the Spotify ID

        Returns
        -------
        bool :
            True if exists, False if does not exist
        """
        try:
            self.s3_client.Object(self.bucket_name, youtube_id).load()
            return True
        except ClientError as error:
            if error.response["Error"]["Code"] == "404":
                return False
            raise

    def upload_song(self, file_path, file_name):
        """ "
        Upload song to our S3 bucket

        Parameters
        ----------
        file_path : Path
            The path to the song to be uploaded
        file_name : string
            The name of the song. Could be youtube ID or spotify ID.
            Currently, is `{artist_name} - {song_name}`.

        Returns
        -------
        None
        """
        # self.s3_client.upload_file(file_path, self.bucket_name, youtube_id)
        self.s3_client.Object(self.bucket_name, file_name).upload_file(file_path)
        file_path.unlink()

    # All of the following  code is modified from spotdl
    # https://github.com/spotDL/spotify-downloader
    @override
    def download_multiple_songs(
        self, songs: List[Song]
    ) -> List[Tuple[Song, Optional[Path]]]:
        """
        Download multiple songs to the temp directory.

        ### Arguments
        - songs: The songs to download.

        ### Returns
        - list of tuples with the song and the path to the downloaded file if successful.
        """
        if self.settings["fetch_albums"]:
            albums = set(song.album_id for song in songs if song.album_id is not None)
            logger.info(
                "Fetching %d album%s", len(albums), "s" if len(albums) > 1 else ""
            )

            songs.extend(songs_from_albums(list(albums)))

            # Remove duplicates
            return_obj = {}
            for song in songs:
                return_obj[song.url] = song

            songs = list(return_obj.values())

        logger.debug("Downloading %d songs", len(songs))

        if self.settings["archive"]:
            songs = [song for song in songs if song.url not in self.url_archive]
            logger.debug("Filtered %d songs with archive", len(songs))

        self.progress_handler.set_song_count(len(songs))

        # Create tasks list
        tasks = [self.pool_download(song) for song in songs]

        # Call all task asynchronously, and wait until all are finished
        results = list(self.loop.run_until_complete(asyncio.gather(*tasks)))

        # Print errors
        if self.settings["print_errors"]:
            for error in self.errors:
                logger.error(error)

        # Save archive
        if self.settings["archive"]:
            for result in results:
                if result[1] or self.settings["add_unavailable"]:
                    self.url_archive.add(result[0].url)

            self.url_archive.save(self.settings["archive"])
            logger.info(
                "Saved archive with %d urls to %s",
                len(self.url_archive),
                self.settings["archive"],
            )

        # Create m3u playlist
        if self.settings["m3u"]:
            song_list = [
                song
                for song, path in results
                if path or self.settings["add_unavailable"]
            ]

            gen_m3u_files(
                song_list,
                self.settings["m3u"],
                self.settings["output"],
                self.settings["format"],
                self.settings["restrict"],
                False,
            )

        # Save results to a file
        if self.settings["save_file"]:
            with open(self.settings["save_file"], "w", encoding="utf-8") as save_file:
                json.dump([song.json for song, _ in results], save_file, indent=4)

            logger.info("Saved results to %s", self.settings["save_file"])

        return results

    @override
    def search_and_download(self, song: Song) -> Tuple[Song, Optional[Path]]:
        """
        Search for the song and download it.

        ### Arguments
        - song: The song to download.

        ### Returns
        - tuple with the song and the path to the downloaded file if successful.

        ### Notes
        - This function is synchronous.
        """
        reinitialized = False
        try:
            # Create the output file path
            output_file = create_file_name(
                song,
                self.settings["output"],
                self.settings["format"],
                self.settings["restrict"],
            )
        except Exception:
            song = reinit_song(song)
            output_file = create_file_name(
                song,
                self.settings["output"],
                self.settings["format"],
                self.settings["restrict"],
            )

            reinitialized = True

        # Initalize the progress tracker
        display_progress_tracker = self.progress_handler.get_new_tracker(song)

        # Create the temp folder path
        temp_folder = get_temp_path()

        # Check if there is an already existing song file, with the same spotify URL in its
        # metadata, but saved under a different name. If so, save its path.
        dup_song_paths: List[Path] = self.known_songs.get(song.url, [])

        # Remove files from the list that have the same path as the output file
        dup_song_paths = [
            dup_song_path
            for dup_song_path in dup_song_paths
            if (dup_song_path.absolute() != output_file.absolute())
            and dup_song_path.exists()
        ]

        file_exists = output_file.exists() or dup_song_paths
        if dup_song_paths:
            logger.debug(
                "Found duplicate songs for %s at %s", song.display_name, dup_song_paths
            )

        # If the file already exists and we don't want to overwrite it,
        # we can skip the download
        if file_exists and self.settings["overwrite"] == "skip":
            logger.info(
                "Skipping %s (file already exists) %s",
                song.display_name,
                "(duplicate)" if dup_song_paths else "",
            )

            display_progress_tracker.notify_download_skip()
            return song, output_file

        # Check if we have all the metadata
        # and that the song object is not a placeholder
        # If it's None extract the current metadata
        # And reinitialize the song object
        # Force song reinitialization if we are fetching albums
        # they have most metadata but not all
        if (
            (song.name is None and song.url)
            or (self.settings["fetch_albums"] and reinitialized is False)
            or None
            in [
                song.genres,
                song.disc_count,
                song.tracks_count,
                song.track_number,
                song.album_id,
                song.album_artist,
            ]
        ):
            song = reinit_song(song)
            reinitialized = True

        # Don't skip if the file exists and overwrite is set to force
        if file_exists and self.settings["overwrite"] == "force":
            logger.info(
                "Overwriting %s %s",
                song.display_name,
                " (duplicate)" if dup_song_paths else "",
            )

            # If the duplicate song path is not None, we can delete the old file
            for dup_song_path in dup_song_paths:
                try:
                    logger.info("Removing duplicate file: %s", dup_song_path)

                    dup_song_path.unlink()
                except (PermissionError, OSError) as exc:
                    logger.debug(
                        "Could not remove duplicate file: %s, error: %s",
                        dup_song_path,
                        exc,
                    )

        ## Currently, we're saving the song name as the spotify id so we can check here.
        s3_file_name = song.song_id
        if self.check_if_song_exists(s3_file_name):
            display_progress_tracker.notify_download_skip()
            return song, None

        # Find song lyrics and add them to the song object
        lyrics = self.search_lyrics(song)
        if lyrics is None:
            logger.debug(
                "No lyrics found for %s, lyrics providers: %s",
                song.display_name,
                ", ".join([lprovider.name for lprovider in self.lyrics_providers]),
            )
        else:
            song.lyrics = lyrics

        # If the file already exists and we want to overwrite the metadata,
        # we can skip the download
        if file_exists and self.settings["overwrite"] == "metadata":
            most_recent_duplicate: Optional[Path] = None
            if dup_song_paths:
                # Get the most recent duplicate song path and remove the rest
                most_recent_duplicate = max(
                    dup_song_paths,
                    key=lambda dup_song_path: dup_song_path.stat().st_mtime,
                )

                # Remove the rest of the duplicate song paths
                for old_song_path in dup_song_paths:
                    if most_recent_duplicate == old_song_path:
                        continue

                    try:
                        logger.info("Removing duplicate file: %s", old_song_path)
                        old_song_path.unlink()
                    except (PermissionError, OSError) as exc:
                        logger.debug(
                            "Could not remove duplicate file: %s, error: %s",
                            old_song_path,
                            exc,
                        )

                # Move the old file to the new location
                if most_recent_duplicate:
                    most_recent_duplicate.replace(
                        output_file.with_suffix(f".{self.settings['format']}")
                    )

            # Update the metadata
            embed_metadata(output_file=output_file, song=song)

            logger.info(
                f"Updated metadata for {song.display_name}"
                f", moved to new location: {output_file}"
                if most_recent_duplicate
                else ""
            )

            display_progress_tracker.notify_complete()

            return song, output_file

        # Create the output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if song.download_url is None:
                download_url = self.search(song)
            else:
                download_url = song.download_url

            ## We would run the download check here if we were storing based on yt data
            # You would just have to extract the yt id from download_url
            # s3_file_name = output_file.name # If we want to use the song name
            # if self.check_if_song_exists(s3_file_name):
            #     display_progress_tracker.notify_download_skip()
            #     return song, None

            # Initialize audio downloader
            audio_downloader = AudioProvider(
                output_format=self.settings["format"],
                cookie_file=self.settings["cookie_file"],
                search_query=self.settings["search_query"],
                filter_results=self.settings["filter_results"],
                geo_bypass=self.settings["geo_bypass"],
            )

            logger.debug("Downloading %s using %s", song.display_name, download_url)

            # Add progress hook to the audio provider
            audio_downloader.audio_handler.add_progress_hook(
                display_progress_tracker.yt_dlp_progress_hook
            )

            # Download the song using yt-dlp
            download_info = audio_downloader.get_download_metadata(
                download_url, download=True
            )

            temp_file = Path(
                temp_folder / f"{download_info['id']}.{download_info['ext']}"
            )

            if download_info is None:
                logger.debug(
                    "No download info found for %s, url: %s",
                    song.display_name,
                    download_url,
                )

                raise DownloaderError(
                    f"yt-dlp failed to get metadata for: {song.name} - {song.artist}"
                )

            display_progress_tracker.notify_download_complete()

            # Ignore the bitrate if the bitrate is set to auto for m4a/opus
            # or if bitrate is set to disabled
            if self.settings["bitrate"] == "disable":
                bitrate = None
            elif self.settings["bitrate"] == "auto" or self.settings["bitrate"] is None:
                # Ignore the bitrate if the input and output formats are the same
                # and the input format is m4a/opus
                if (temp_file.suffix == ".m4a" and output_file.suffix == ".m4a") or (
                    temp_file.suffix == ".opus" and output_file.suffix == ".opus"
                ):
                    bitrate = None
                else:
                    bitrate = f"{int(download_info['abr'])}k"
            else:
                bitrate = str(self.settings["bitrate"])

            success, result = convert(
                input_file=temp_file,
                output_file=output_file,
                ffmpeg=self.ffmpeg,
                output_format=self.settings["format"],
                bitrate=bitrate,
                ffmpeg_args=self.settings["ffmpeg_args"],
                progress_handler=display_progress_tracker.ffmpeg_progress_hook,
            )

            # Remove the temp file
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except (PermissionError, OSError) as exc:
                    logger.debug(
                        "Could not remove temp file: %s, error: %s", temp_file, exc
                    )

                    raise DownloaderError(
                        f"Could not remove temp file: {temp_file}, possible duplicate song"
                    ) from exc

            if not success and result:
                # If the conversion failed and there is an error message
                # create a file with the error message
                # and save it in the errors directory
                # raise an exception with file path
                file_name = (
                    get_errors_path()
                    / f"ffmpeg_error_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
                )

                error_message = ""
                for key, value in result.items():
                    error_message += f"### {key}:\n{str(value).strip()}\n\n"

                with open(file_name, "w", encoding="utf-8") as error_path:
                    error_path.write(error_message)

                # Remove the file that failed to convert
                if output_file.exists():
                    output_file.unlink()

                raise FFmpegError(
                    f"Failed to convert {song.display_name}, "
                    f"you can find error here: {str(file_name.absolute())}"
                )

            download_info["filepath"] = str(output_file)

            # Set the song's download url
            if song.download_url is None:
                song.download_url = download_url

            display_progress_tracker.notify_conversion_complete()

            # SponsorBlock post processor
            if self.settings["sponsor_block"]:
                # Initialize the sponsorblock post processor
                post_processor = SponsorBlockPP(
                    audio_downloader.audio_handler, SPONSOR_BLOCK_CATEGORIES
                )

                # Run the post processor to get the sponsor segments
                _, download_info = post_processor.run(download_info)
                chapters = download_info["sponsorblock_chapters"]

                # If there are sponsor segments, remove them
                if len(chapters) > 0:
                    logger.info(
                        "Removing %s sponsor segments for %s",
                        len(chapters),
                        song.display_name,
                    )

                    # Initialize the modify chapters post processor
                    modify_chapters = ModifyChaptersPP(
                        audio_downloader.audio_handler,
                        remove_sponsor_segments=SPONSOR_BLOCK_CATEGORIES,
                    )

                    # Run the post processor to remove the sponsor segments
                    # this returns a list of files to delete
                    files_to_delete, download_info = modify_chapters.run(download_info)

                    # Delete the files that were created by the post processor
                    for file_to_delete in files_to_delete:
                        Path(file_to_delete).unlink()

            try:
                embed_metadata(output_file, song, self.settings["id3_separator"])
            except Exception as exception:
                raise MetadataError(
                    "Failed to embed metadata to the song"
                ) from exception

            if self.settings["generate_lrc"]:
                if song.lyrics and is_lrc_valid(song.lyrics):
                    lrc_data = song.lyrics
                else:
                    try:
                        lrc_data = syncedlyrics_search(song.display_name)
                    except Exception:
                        lrc_data = None

                if lrc_data:
                    save_lrc_file(str(output_file.with_suffix(".lrc")), lrc_data)
                    logger.debug("Saved lrc file for %s", song.display_name)
                else:
                    logger.debug("No lrc file found for %s", song.display_name)

            display_progress_tracker.notify_complete()

            # Add the song to the known songs
            self.known_songs.get(song.url, []).append(output_file)

            logger.info('Downloaded "%s": %s', song.display_name, song.download_url)

            ## Upload to S3
            self.upload_song(output_file, s3_file_name)

            return song, output_file
        except (Exception, UnicodeEncodeError) as exception:
            if isinstance(exception, UnicodeEncodeError):
                exception_cause = exception
                exception = DownloaderError(
                    "You may need to add PYTHONIOENCODING=utf-8 to your environment"
                )

                exception.__cause__ = exception_cause

            display_progress_tracker.notify_error(
                traceback.format_exc(), exception, True
            )
            self.errors.append(
                f"{song.url} - {exception.__class__.__name__}: {exception}"
            )
            return song, None
