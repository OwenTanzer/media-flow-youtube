"""Discovers new uploads via the official YouTube Data API v3's
uploads-playlist flow (issue #24) - the primary discovery source once
YOUTUBE_DATA_API_KEY is set, replacing the public RSS feeds discovery.py
otherwise falls back to. See discovery.py for how these two functions are
combined with the existing dedup/queue logic."""

from __future__ import annotations

import requests

from .discovered_video import DiscoveredVideo

CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
PLAYLIST_ITEMS_URL = "https://www.googleapis.com/youtube/v3/playlistItems"
PLAYLIST_ITEMS_PAGE_SIZE = 50


class YouTubeDataApiError(Exception):
    pass


def _get_json(url: str, params: dict, timeout: float) -> dict:
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise YouTubeDataApiError(f"YouTube Data API request to {url} failed: {exc}") from exc


def resolve_uploads_playlist_id(channel_id: str, api_key: str, timeout: float = 10.0) -> str:
    """Looks up a channel's uploads playlist ID via channels.list
    (part=contentDetails), the one-time (then cached, see
    uploads_playlist_store.py) resolution discovery.py needs before it can
    poll playlistItems.list for that channel."""

    data = _get_json(CHANNELS_URL, {"part": "contentDetails", "id": channel_id, "key": api_key}, timeout)

    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise YouTubeDataApiError(f"channels.list returned no items for channel {channel_id!r}: {data!r}")

    try:
        return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except (KeyError, TypeError) as exc:
        raise YouTubeDataApiError(
            f"channels.list response for {channel_id!r} is missing contentDetails.relatedPlaylists.uploads: {items[0]!r}"
        ) from exc


def fetch_uploads_playlist_videos(
    playlist_id: str, api_key: str, known_ids: set[str], channel_id: str, timeout: float = 10.0
) -> list[DiscoveredVideo]:
    """Polls a channel's uploads playlist via playlistItems.list, paginating
    while every video on the page is new. The uploads playlist is ordered
    newest-first, so as soon as an already-known video ID is seen, everything
    after it is known too - stopping there keeps steady-state hourly polling
    to a single request per channel while still fully draining history for a
    brand-new channel (empty known_ids paginates until exhausted)."""

    videos: list[DiscoveredVideo] = []
    page_token: str | None = None

    while True:
        params = {
            "part": "contentDetails,snippet",
            "playlistId": playlist_id,
            "maxResults": PLAYLIST_ITEMS_PAGE_SIZE,
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        data = _get_json(PLAYLIST_ITEMS_URL, params, timeout)

        items = data.get("items")
        if not isinstance(items, list):
            raise YouTubeDataApiError(f"playlistItems.list returned no items list for playlist {playlist_id!r}: {data!r}")

        hit_known = False
        for item in items:
            content_details = item.get("contentDetails") or {}
            video_id = content_details.get("videoId")
            if not video_id:
                continue
            if video_id in known_ids:
                hit_known = True
                break
            snippet = item.get("snippet") or {}
            videos.append(
                DiscoveredVideo(
                    video_id=video_id,
                    channel_id=channel_id,
                    published=content_details.get("videoPublishedAt") or snippet.get("publishedAt"),
                )
            )

        page_token = data.get("nextPageToken")
        if hit_known or not page_token:
            break

    return videos
