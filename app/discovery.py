"""Discovers new uploads from configured YouTube channels and queues
previously-unseen video IDs for the existing batch pipeline. Uses the
YouTube Data API's uploads-playlist flow (app/youtube_data_api.py, issue
#24) when YOUTUBE_DATA_API_KEY is set, otherwise falls back to each
channel's public upload RSS feed. See discover_and_process.py (repo root)
for the serialized entrypoint that combines this with running the queue."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from . import channel_store, drive, queue_store, uploads_playlist_store, youtube, youtube_data_api
from .config import settings
from .discovered_video import DiscoveredVideo
from .youtube_data_api import YouTubeDataApiError

logger = logging.getLogger("media_flow.discovery")

FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
}


@dataclass
class DiscoveryReport:
    channels_configured: int
    channels_enabled: int
    discovered_total: int
    newly_queued: int
    duplicates_skipped: int
    feed_failures: list[tuple[str, str]] = field(default_factory=list)


def fetch_channel_feed(channel_id: str, timeout: float = 10.0) -> list[DiscoveredVideo]:
    """Fetches and parses a channel's public upload RSS feed. Raises on a
    network or parse failure - discover_and_enqueue() isolates one
    channel's failure from the others."""

    proxy_config = youtube.build_proxy_config()
    response = requests.get(
        FEED_URL.format(channel_id=channel_id),
        timeout=timeout,
        proxies=proxy_config.to_requests_dict() if proxy_config else None,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)

    videos = []
    for entry in root.findall("atom:entry", ATOM_NS):
        video_id_el = entry.find("yt:videoId", ATOM_NS)
        if video_id_el is None or not video_id_el.text:
            continue
        published_el = entry.find("atom:published", ATOM_NS)
        videos.append(
            DiscoveredVideo(
                video_id=video_id_el.text.strip(),
                channel_id=channel_id,
                published=published_el.text if published_el is not None else None,
            )
        )
    return videos


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _known_video_ids(folder_id: str, existing_queue: list[str | dict]) -> set[str]:
    known = set(drive.read_index(folder_id).keys())
    for entry in existing_queue:
        try:
            known.add(youtube.extract_video_id(queue_store.entry_url(entry)))
        except youtube.VideoUrlError:
            logger.warning("Could not parse existing queue entry %r while discovering; leaving it as-is.", entry)
    return known


def _fetch_channel_videos_via_api(
    channel: channel_store.Channel, known_ids: set[str], playlist_cache: dict[str, str], api_key: str
) -> list[DiscoveredVideo]:
    """Resolves (and caches, in playlist_cache) the channel's uploads
    playlist ID, then polls it via the Data API. Raises YouTubeDataApiError
    on any failure - the caller isolates one channel's failure from the
    others, same as fetch_channel_feed()."""

    playlist_id = playlist_cache.get(channel.channel_id)
    if playlist_id is None:
        playlist_id = youtube_data_api.resolve_uploads_playlist_id(channel.channel_id, api_key)
        playlist_cache[channel.channel_id] = playlist_id

    return youtube_data_api.fetch_uploads_playlist_videos(playlist_id, api_key, known_ids, channel.channel_id)


def _enqueue_new_videos(folder_id: str, channels: list[channel_store.Channel]) -> tuple[int, int, int, list[tuple[str, str]]]:
    """Fetches each given channel's latest uploads and enqueues any video
    not already known (in _index.json or queue.json), scoped to exactly the
    channels passed in. Shared core for discover_and_enqueue() (every
    enabled channel, the normal recurring poll) and
    backfill_new_channels() (just channels with no known videos yet - see
    that function and backfill_new_channels.py at the repo root).

    Uses the YouTube Data API's uploads-playlist flow (issue #24) as the
    exclusive source when YOUTUBE_DATA_API_KEY is set - RSS is only used as
    a fallback while that key is unset, so there's never a double-fetch per
    channel.

    Returns (discovered_total, newly_queued, duplicates_skipped,
    feed_failures)."""

    existing_queue = queue_store.read_queue(folder_id)
    known_ids = _known_video_ids(folder_id, existing_queue)

    api_key = settings.youtube_data_api_key
    playlist_cache = uploads_playlist_store.read_cache(folder_id) if api_key else {}
    playlist_cache_dirty = False

    new_entries: list[str | dict] = []
    discovered_total = 0
    duplicates_skipped = 0
    feed_failures: list[tuple[str, str]] = []
    seen_this_run: set[str] = set()

    for channel in channels:
        try:
            if api_key:
                had_cached_playlist = channel.channel_id in playlist_cache
                videos = _fetch_channel_videos_via_api(channel, known_ids, playlist_cache, api_key)
                if not had_cached_playlist:
                    playlist_cache_dirty = True
            else:
                videos = fetch_channel_feed(channel.channel_id)
        except (requests.RequestException, ET.ParseError, YouTubeDataApiError) as exc:
            logger.warning("Feed fetch failed for channel %s (%s): %s", channel.channel_id, channel.name, exc)
            feed_failures.append((channel.channel_id, str(exc)))
            continue

        discovered_total += len(videos)
        for video in videos:
            if video.video_id in known_ids or video.video_id in seen_this_run:
                duplicates_skipped += 1
                continue
            seen_this_run.add(video.video_id)
            url = youtube.canonical_url(video.video_id)
            entry: dict = {"url": url, "first_seen_at": _utcnow().isoformat(), "channel_id": channel.channel_id}
            if video.published:
                entry["published_at"] = video.published
            if channel.languages:
                entry["languages"] = channel.languages
            new_entries.append(entry)

    if new_entries:
        queue_store.write_queue(folder_id, existing_queue + new_entries)
    if playlist_cache_dirty:
        uploads_playlist_store.write_cache(folder_id, playlist_cache)

    return discovered_total, len(new_entries), duplicates_skipped, feed_failures


def discover_and_enqueue(folder_id: str) -> DiscoveryReport:
    channels = channel_store.read_channels(folder_id)
    enabled = [channel for channel in channels if channel.enabled]

    discovered_total, newly_queued, duplicates_skipped, feed_failures = _enqueue_new_videos(folder_id, enabled)

    return DiscoveryReport(
        channels_configured=len(channels),
        channels_enabled=len(enabled),
        discovered_total=discovered_total,
        newly_queued=newly_queued,
        duplicates_skipped=duplicates_skipped,
        feed_failures=feed_failures,
    )


def find_unbackfilled_channels(folder_id: str) -> list[channel_store.Channel]:
    """Enabled channels with zero videos anywhere in _index.json or
    queue.json yet - i.e. genuinely never discovered, not merely "no new
    uploads since the last check". A channel just added to channels.json
    has none of its videos in either place, so it shows up here until its
    first backfill (or the next normal discovery run, which would also
    pick it up - see discover_and_enqueue()) runs at least once.

    Used by backfill_new_channels() to scope a one-off backfill to just
    the channels that actually need it."""

    channels = channel_store.read_channels(folder_id)
    index = drive.read_index(folder_id)
    existing_queue = queue_store.read_queue(folder_id)

    known_channel_ids = {entry.get("channel_id") for entry in index.values() if entry.get("channel_id")}
    known_channel_ids |= {
        entry.get("channel_id") for entry in existing_queue if isinstance(entry, dict) and entry.get("channel_id")
    }
    return [c for c in channels if c.enabled and c.channel_id not in known_channel_ids]


def backfill_new_channels(folder_id: str) -> DiscoveryReport:
    """Runs the same fetch-and-enqueue logic as discover_and_enqueue(),
    scoped to only the channels find_unbackfilled_channels() identifies as
    never-discovered - see backfill_new_channels.py (repo root) for the
    standalone entrypoint this backs. That script deliberately does not
    use app/job_lock.py's main discovery lock: this is a single feed fetch
    plus queue append per new channel (seconds, not the potentially
    long-running full discover+batch+summarize cycle), and there's no
    reason it should have to wait for or contend with that lock."""

    channels = find_unbackfilled_channels(folder_id)
    discovered_total, newly_queued, duplicates_skipped, feed_failures = _enqueue_new_videos(folder_id, channels)

    return DiscoveryReport(
        channels_configured=len(channels),
        channels_enabled=len(channels),
        discovered_total=discovered_total,
        newly_queued=newly_queued,
        duplicates_skipped=duplicates_skipped,
        feed_failures=feed_failures,
    )
