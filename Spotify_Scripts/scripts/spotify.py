"""
Spotify class, authenticates and holds client. All our spotify helpers live here.
"""
import logging
import os
import time
from typing import List

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

from utils import ceildiv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Spotify:
    """
    Spotify class, authenticates and holds client. All our spotify helpers live here.
    """

    def __init__(self) -> None:
        """
        Initializes the Spotipy library with .env values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        scope = [
            "user-library-read",
            "playlist-read-private",
            "playlist-read-collaborative",
            "playlist-modify-private",
            "playlist-modify-public",
        ]
        self.temp_playlists = os.getenv("TEMP_PLAYLISTS", default="").split(",")
        client_id = os.getenv("CLIENT_ID", default="")
        client_secret = os.getenv("CLIENT_SECRET", default="")
        self.client = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                username="house-machine",
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri="http://localhost",
                scope=" ".join(scope),
            )
        )
        user_info = self.client.current_user()
        assert user_info
        logger.info("Logged in as %s", user_info["display_name"])

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

    def get_recommendation_artist_df(
        self, artist_ids: List[str], checkpoint: int = 0
    ) -> pd.DataFrame:
        """
        Hits Spotipy to get a df of our artists' tracks

        Parameters
        ----------
        artist_ids : List[string]
            The ID's for our seed Spotify artists
        checkpoint : int
            The checkpoint to start at (last failed artist)

        Returns
        -------
        df :
            Dataframe of all tracks from our artists
        """
        artist_count = len(artist_ids)
        if artist_count == 0:
            raise EnvironmentError("No artist id's provided")
        logger.info("Collecting songs from %s artists", artist_count)

        final_df = pd.DataFrame()
        if checkpoint != 0:
            logger.info("Starting from checkpoint %s", checkpoint)
            final_df = pd.read_csv(
                "checkpoints/data/artist_checkpoint.csv", index_col=0
            )

        for index in range(checkpoint, len(artist_ids)):
            logger.info("Collecting songs from artist #%s/%s", index, artist_count)
            artist_id = artist_ids[index]
            # Get artist's albums
            albums = self.client.artist_albums(artist_id)
            assert albums

            # Make list to hold all track ids for the current artist
            artist_track_ids = []

            for album in albums["items"]:
                # Get list of artist tracks
                track_results = self.client.album_tracks(album["id"])
                assert track_results

                artist_track_ids.extend(
                    [track["id"] for track in track_results["items"]]
                )

            # If for some reason the artist has no tracks, let's move on.
            # This is super weird, but it DOES happen
            if len(artist_track_ids) == 0:
                continue

            # Add to temporary playlist.
            # We do this because the spotipy.album_tracks does not give us popularity
            # or album name, and we'll probably want that for filtering later.
            logger.info("Uploading artist %s to temp playlist", artist_id)
            self.add_all_songs_to_playlists(self.temp_playlists, artist_track_ids)

            # Get songs from playlist as df
            # Weird indexing on temp_playlists is to prevent us from over-reading playlists
            # with nothing in them.
            artist_df = self.get_recommendation_playlist_df(
                self.temp_playlists[0 : ceildiv(len(artist_track_ids), 10000)]
            )

            # Add to total df
            final_df = (
                pd.concat([final_df, artist_df])
                .drop_duplicates("track_id")
                .reset_index(drop=True)
            )

            # Checkpoint just in case we get rate limited or something :)
            final_df.to_csv("tmp/artist_checkpoint.csv")

        final_df.to_csv("tmp/artist_recommendations.csv")
        return final_df

    def get_recommendation_genre_df(
        self, genres: List[str], num_tracks: int = 10000
    ) -> pd.DataFrame:
        """
        Hits Spotipy to get a df of our artists' tracks

        Parameters
        ----------
        genres : List[string]
            The Spotify genres from which to get recommendations
        num_tracks: int
            The total amount of recommendations to get for the genre

        Returns
        -------
        df :
            Dataframe of all tracks from our genre
        """
        valid_genres = self.client.recommendation_genre_seeds()
        assert valid_genres
        for genre in genres:
            if genre not in valid_genres["genres"]:
                raise EnvironmentError(f"Genre '{genre}' does not exist in Spotify")

        # Define the list of recommended tracks
        recommended_tracks = []

        # Retrieve recommendations based on the 'trance' genre and a limit of 100 tracks per request
        while len(recommended_tracks) < num_tracks:
            remaining_tracks = num_tracks - len(recommended_tracks)
            logger.info("Collecting genre tracks. Remaining: %s", remaining_tracks)
            limit = min(remaining_tracks, 100)
            recommendations = self.client.recommendations(
                seed_genres=genres, limit=limit, fields="tracks"
            )
            assert recommendations
            recommended_tracks += recommendations["tracks"]

        # Create df of our genre-recommended songs
        genre_df: pd.DataFrame = self.create_df_from_recommendation_results(
            recommended_tracks
        )
        genre_df.to_csv("tmp/genre_recommendations.csv")
        return genre_df

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

        Returns
        -------
        pd.Dataframe :
            DF of objects containing the track objects from Spotipy
        """
        # If we provided a checkpoint we should use it.
        if checkpoint != 0:
            logger.info("Starting from checkpoint %s", checkpoint)
            recommendation_df = pd.read_csv(
                "checkpoints/data/recommendation_checkpoint.csv", index_col=0
            )

        recommendation_df = pd.DataFrame()
        for i in range(checkpoint, len(seed_tracks), batch_size):
            # Some website said spotify's rate limit is 180rpm
            # So let's just pause 1s per rec to be safe :shrug:
            time.sleep(1)
            logger.info(
                "Getting recommendations from songs %s to %s", i, i + batch_size
            )
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
                song_batch // 1000,
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
