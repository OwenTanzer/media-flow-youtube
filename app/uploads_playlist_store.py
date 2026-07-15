"""Caches each channel's resolved YouTube Data API uploads-playlist ID
(channel_id -> playlist_id) in `_uploads_playlists.json` in the Drive
folder, so discovery.py's API path (app/youtube_data_api.py, issue #24)
only needs one channels.list quota unit per channel ever, instead of one
every discovery run. Deliberately a separate file rather than a field on
channel_store.Channel/channels.json - that file is operator-edited
directly in Drive (see channel_store.py's docstring), and shouldn't also
carry app-derived cache data."""

from __future__ import annotations

import json
import logging

from . import drive

logger = logging.getLogger("media_flow.uploads_playlist_store")

CACHE_FILENAME = "_uploads_playlists.json"


def read_cache(folder_id: str) -> dict[str, str]:
    text = drive.download_text(folder_id, CACHE_FILENAME)
    if text is None:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("%s in folder %s was not valid JSON; treating as empty.", CACHE_FILENAME, folder_id)
        return {}

    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def write_cache(folder_id: str, cache: dict[str, str]) -> None:
    drive.upload_text_file(
        folder_id, CACHE_FILENAME, json.dumps(cache, indent=2, sort_keys=True), mime_type="application/json"
    )
