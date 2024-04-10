"""
House-Machine's version of the Spotdl class, which simplifies
the process of downloading songs from Spotify.
"""

import asyncio
from typing import Optional, Union

from scripts.hm_downloader import HMDownloader
from spotdl import Spotdl
from spotdl.types.options import DownloaderOptionalOptions, DownloaderOptions
from spotdl.utils.spotify import SpotifyClient


class HmSpotDl(Spotdl):
    """
    House-Machine's version of the Spotdl class, which simplifies
    the process of downloading songs from Spotify.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_auth: bool = False,
        cache_path: Optional[str] = None,
        no_cache: bool = False,
        headless: bool = False,
        downloader_settings: Optional[
            Union[DownloaderOptionalOptions, DownloaderOptions]
        ] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        # pylint: disable-super-init-not-called
        """
        Initialize the Spotdl class

        ### Arguments
        - client_id: Spotify client id
        - client_secret: Spotify client secret
        - user_auth: If true, user will be prompted to authenticate
        - cache_path: Path to cache directory
        - no_cache: If true, no cache will be used
        - headless: If true, no browser will be opened
        - downloader_settings: Settings for the downloader
        - loop: Event loop to use
        """

        if downloader_settings is None:
            downloader_settings = {}

        # Initialize spotify client
        SpotifyClient.init(
            client_id=client_id,
            client_secret=client_secret,
            user_auth=user_auth,
            cache_path=cache_path,
            no_cache=no_cache,
            headless=headless,
        )

        # Initialize downloader (house-machine's version)
        self.downloader = HMDownloader(
            settings=downloader_settings,
            loop=loop,
        )
