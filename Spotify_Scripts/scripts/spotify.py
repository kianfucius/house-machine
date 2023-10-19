"""
Spotify class, authenticates and holds client. All our spotify helpers live here.
"""
import logging
import os
from typing import List

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from scripts.utils import ceildiv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Spotify:
    """
    Spotify class, authenticates and holds client. All our spotify helpers live here.
    """

    def __init__(self, cred_num: int = 0) -> None:
        """
        Initializes the Spotipy library with .env values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # credNum indicates the current index of the credentials we're using,
        # since we're using a whole bunch of accounts
        self.cred_num = cred_num
        logger.info("Using cred #%s", self.cred_num)
        self.temp_playlists = os.getenv("TEMP_PLAYLISTS", default="").split(",")
        self.user_name = os.getenv("USERNAME", default="")
        # client_id = os.getenv("CLIENT_ID", default="")
        # client_secret = os.getenv("CLIENT_SECRET", default="")
        self.client_ids = os.getenv("CLIENT_IDS", default="").split(",")
        self.client_secrets = os.getenv("CLIENT_SECRETS", default="").split(",")
        self.authenticate(self.cred_num)

    def authenticate(self, account_index: int = 0):
        """
        Authenticates the Spotipy library with .env values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.cred_num > len(self.client_ids):
            raise EnvironmentError("You need to specify more credentials in .env.")
        # remove .cache-house-machine from root
        # if os.path.exists(f".cache-{self.user_name}"):
        #     os.remove(f".cache-{self.user_name}")
        scope = [
            "user-library-read",
            "playlist-read-private",
            "playlist-read-collaborative",
            "playlist-modify-private",
            "playlist-modify-public",
        ]
        self.client = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                username=self.user_name,
                client_id=self.client_ids[account_index],
                client_secret=self.client_secrets[account_index],
                redirect_uri="http://localhost",
                scope=" ".join(scope),
            ),
            requests_timeout=5,
            retries=0,
        )
        # user_info = self.client.current_user()
        # assert user_info
        # logger.info(
        #     "Logged in as %s w/ key %s",
        #     user_info["display_name"],
        #     self.client_ids[account_index],
        # )
        self.cred_num += 1

    def get_recommendation_playlist_df(self, playlist_ids: List[str]) -> pd.DataFrame:
        """
        Hits Spotipy to get a df of our playlist tracks

        Parameters
        ----------
        playlist_ids : List[string]
            The ID's for our seed Spotify playlists

        Returns
        -------
        df :
            Dataframe of all tracks from our playlists
        """
        if len(playlist_ids) == 0:
            raise EnvironmentError("No playlist id's provided")

        logger.info("Getting playlists: %s", playlist_ids)
        final_df = pd.DataFrame()
        for playlist_id in playlist_ids:
            # Get the number of tracks
            num_tracks: int = self.__get_playlist_length(playlist_id)

            # Get the playlist songs from API
            playlist_tracks: List = self.__get_playlist_results(playlist_id, num_tracks)

            # Convert to df
            playlist_df: pd.DataFrame = self.__create_df_from_playlist_results(
                playlist_tracks
            )

            # Add to total df
            final_df = (
                pd.concat([final_df, playlist_df])
                .drop_duplicates("track_id")
                .reset_index(drop=True)
            )

        return final_df

    def __get_playlist_length(self, playlist_id: str) -> int:
        """
        Hits Spotipy to get the length of a spotify playlist

        Parameters
        ----------
        playlist_id : string
            The ID for a Spotify playlist

        Returns
        -------
        int :
            Length of Spotify playlist
        """
        # Get the total number of tracks in the playlist
        playlist = self.client.playlist(playlist_id)
        assert playlist
        num_tracks = playlist["tracks"]["total"]
        logger.info("Playlist %s has %s songs", playlist["name"], num_tracks)

        return num_tracks

    def __get_playlist_results(self, playlist_id: str, num_tracks: int) -> List[object]:
        """
        Hits Spotipy to get a list of all the tracks in a playlist, as objects

        Parameters
        ----------
        playlist_id : string
            The ID for a Spotify playlist
        num_tracks: int
            The number of tracks in the playlist

        Returns
        -------
        list[object] :
            List of objects containing the track objects from Spotipy

        """
        # Set up an empty list to store the tracks
        tracks = []

        # Set up an offset counter to handle pagination
        offset = 0

        # Loop through all pages of tracks in the playlist
        while offset < num_tracks:
            logger.info("Fetching from offset: %s", offset)
            # Get a page of tracks from the playlist
            results = self.client.playlist_tracks(
                playlist_id, offset=offset, fields="items(track)"
            )
            assert results
            tracks.extend(results["items"])

            # Increment the offset to get the next page of tracks
            offset += len(results["items"])

        logger.info("Fetched %s songs from playlist %s", len(tracks), playlist_id)
        return tracks

    def get_recommendations_from_seeds(
        self, seed_tracks: list[str], batch_size: int, limit: int, checkpoint: int = 0
    ) -> pd.DataFrame:
        """
        Hits Spotipy to get unique recommendations based on a list of seed tracks.

        Parameters
        ----------
        seed_tracks : list[str]
            The list of track id's to use as a seed
        batch_size : int
            The amount of seeds to pass in, incrementally
        limit : int
            The amount of recommendations to generate per query.
            Spotify's limit is 5 tracks.
        checkpoint : int
            The checkpoint from which to start (where we last failed).
            This will probably be a multiple of 100.

        Returns
        -------
        pd.Dataframe :
            DF of objects containing the track objects from Spotipy
        """
        # If we provided a checkpoint we should use it.
        if checkpoint > 0:
            logger.info("Starting from checkpoint %s", checkpoint)
            recommendation_df = pd.read_csv(
                "checkpoints/recommendation_checkpoint.csv", index_col=0
            )
        else:
            checkpoint = 0
            recommendation_df = pd.DataFrame()

        for i in range(checkpoint, len(seed_tracks), batch_size):
            logger.info(
                "Getting recommendations from songs %s to %s", i, i + batch_size
            )
            try:
                # Hit API to get recommendations
                recommendations = self.client.recommendations(
                    seed_tracks=seed_tracks[i : i + batch_size],
                    limit=limit,
                    fields="tracks",
                )
            except Exception:
                logger.info(
                    "Failed to get recommendations, reauthing at index %s",
                    self.cred_num,
                )
                self.authenticate(self.cred_num)
                # Hit API to get recommendations
                recommendations = self.client.recommendations(
                    seed_tracks=seed_tracks[i : i + batch_size],
                    limit=limit,
                    fields="tracks",
                )
            assert recommendations
            # Add to recomendation df
            temp_df = self.create_df_from_recommendation_results(
                recommendations["tracks"]
            )
            recommendation_df = (
                pd.concat([recommendation_df, temp_df])
                .drop_duplicates("track_id")
                .reset_index(drop=True)
            )
            recommendation_df.to_csv("tmp/recommendation_checkpoint.csv")
        return recommendation_df

    def add_all_songs_to_playlists(
        self, playlist_ids: List[str], tracks: List[str]
    ) -> None:
        """
        Replaces the contents of a playlist

        Parameters
        ----------
        playlist_ids : list[string]
            The ID of the Spotify playlists
        tracks : list[string]
            The list of tracks to add

        Returns
        -------
        None
        """
        # Add all songs to playlist.
        # Spotipy limits us to 100 at a time, and playlists cannot exceed 10k songs
        total_tracks = len(tracks)
        if total_tracks > len(playlist_ids) * 10000:
            raise EnvironmentError(
                f"""We have {total_tracks} tracks to upload, \
                            but only {len(playlist_ids)} playlists to hold them."""
            )

        # Ceiling division determines which playlist id to upload to
        for playlist_index in range(0, ceildiv(total_tracks, 10000)):
            # Remove all songs from playlist
            self.client.playlist_replace_items(playlist_ids[playlist_index], [])
            # Upload to the correct playlist in batches of 100
            for i in range(
                0,
                len(tracks[playlist_index * 10000 : (playlist_index + 1) * 10000]),
                100,
            ):
                logger.info("Uploading tracks %s to %s", i, i + 100)
                self.client.playlist_add_items(
                    playlist_ids[playlist_index], tracks[i : i + 100]
                )

    def append_to_playlists(
        self, playlist_ids: List[str], tracks: List[str], checkpoint: int = 0
    ) -> None:
        """
        Appends to a playlist

        Parameters
        ----------
        playlist_ids : list[string]
            The ID of the Spotify playlists
        tracks : list[string]
            The list of tracks to add
        checkpoint : int
            The checkpoint from which to start (where we last failed).
            This will probably be a multiple of 100.

        Returns
        -------
        None
        """
        # Add all songs to playlist.
        # Spotipy limits us to 100 at a time, and playlists cannot exceed 10k songs
        total_tracks = len(tracks)
        if total_tracks > len(playlist_ids) * 10000:
            raise EnvironmentError(
                f"""We have {total_tracks} tracks to upload, \
                            but only {len(playlist_ids)} playlists to hold them."""
            )

        # Based on checkpoint, we can determine which playlists to wipe.
        # We should only wipe playlists higher than the checkpoint value.
        # e.g. if checkpoint = 15000, then wipe playlist[2:]
        for playlist_index in range(checkpoint // 10000, len(playlist_ids)):
            logger.info(
                "Emptying playlist #%s with id %s",
                playlist_index,
                playlist_ids[playlist_index],
            )
            # Remove all songs from playlist
            self.client.playlist_replace_items(playlist_ids[playlist_index], [])

        for song_batch in range(checkpoint, len(tracks), 100):
            logger.info(
                "Uploading tracks %s to %s of %s to playlist #%s",
                song_batch,
                song_batch + 100,
                total_tracks,
                song_batch // 10000,
            )
            self.client.playlist_add_items(
                playlist_ids[song_batch // 10000], tracks[song_batch : song_batch + 100]
            )

    @staticmethod
    def filter_playlist(
        song_df: pd.DataFrame, min_length_mins: float = 2, max_length_mins: float = 4
    ) -> pd.DataFrame:
        """
        Filters a playlist by duration.

        Parameters
        ----------
        song_df : DataFrame
            The dataframe to filter
        min_length_mins : float
            The minimum amount of minutes for a song duration. e.g. 2 mins
        max_length_mins : float
            The maximum length of a song in minutes e.g. 3.5 is 3min 30sec
        """
        logger.info("Playlist length before duration filtering: %s", len(song_df))

        # Get times in milliseconds
        min_length = 1000 * 60 * min_length_mins
        max_length = 1000 * 60 * max_length_mins

        song_df = song_df[
            (song_df["duration"] > min_length) & (song_df["duration"] < max_length)
        ].reset_index(drop=True)
        logger.info("Playlist length after duration filtering: %s", len(song_df))

        song_df.to_csv("tmp/playlist_seed_filtered_duration.csv")
        return song_df

    @staticmethod
    def __create_df_from_playlist_results(api_results: List) -> pd.DataFrame:
        """
        Reads in the spotipy query results from the playlist API, and returns a
        DataFrame with track_name, track_id, artist, album, duration, popularity

        Parameters
        ----------
        api_results : list[object]
            The results of a query to spotify with .current_user_saved_tracks()

        Returns
        -------
        df :
            DataFrame containing track_name, track_id, artist, album, duration, popularity
        """
        # create lists for df-columns
        track_name = []
        track_id = []
        artist = []
        artist_id = []
        album = []
        duration = []
        popularity = []
        # loop through api_results
        for items in api_results:
            try:
                track_name.append(items["track"]["name"])
                track_id.append(items["track"]["id"])
                artist.append(items["track"]["artists"][0]["name"])
                artist_id.append(items["track"]["artists"][0]["id"])
                duration.append(items["track"]["duration_ms"])
                album.append(items["track"]["album"]["name"])
                popularity.append(items["track"]["popularity"])
            except TypeError:
                pass
        # Create the final df
        playlist_df = pd.DataFrame(
            {
                "track_name": track_name,
                "album": album,
                "track_id": track_id,
                "artist": artist,
                "artist_id": artist_id,
                "duration": duration,
                "popularity": popularity,
            }
        )

        return playlist_df

    @staticmethod
    def create_df_from_recommendation_results(api_results: List) -> pd.DataFrame:
        """
        Reads in the spotipy query results for the recommendation API, and returns
        DataFrame with: track_name, track_id, artist, album, duration, popularity

        Parameters
        ----------
        api_results :
            The results of a query to spotify with .current_user_saved_tracks()

        Returns
        -------
        df :
            DataFrame containing track_name, track_id, artist, album, duration, popularity
        """
        track_name = []
        track_id = []
        artist = []
        album = []
        duration = []
        popularity = []
        for items in api_results:
            try:
                track_name.append(items["name"])
                track_id.append(items["id"])
                artist.append(items["artists"][0]["name"])
                duration.append(items["duration_ms"])
                album.append(items["album"]["name"])
                popularity.append(items["popularity"])
            except TypeError:
                pass
        recommendation_df = pd.DataFrame(
            {
                "track_name": track_name,
                "album": album,
                "track_id": track_id,
                "artist": artist,
                "duration": duration,
                "popularity": popularity,
            }
        )

        return recommendation_df
