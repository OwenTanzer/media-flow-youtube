"""Discovers new uploads from configured YouTube channels via their public
RSS feeds and queues previously-unseen video IDs for the existing batch
pipeline. See discover_and_process.py (repo root) for the serialized
entrypoint that combines this with running the queue."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from . import channel_store, drive, queue_store, youtube

logger = logging.getLogger("media_flow.discovery")

FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
}


@dataclass
class DiscoveredVideo:
    video_id: str
    channel_id: str
    published: str | None


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


def discover_and_enqueue(folder_id: str) -> DiscoveryReport:
    channels = channel_store.read_channels(folder_id)
    enabled = [channel for channel in channels if channel.enabled]

    existing_queue = queue_store.read_queue(folder_id)
    known_ids = _known_video_ids(folder_id, existing_queue)

    new_entries: list[str | dict] = []
    discovered_total = 0
    duplicates_skipped = 0
    feed_failures: list[tuple[str, str]] = []
    seen_this_run: set[str] = set()

    for channel in enabled:
        try:
            videos = fetch_channel_feed(channel.channel_id)
        except (requests.RequestException, ET.ParseError) as exc:
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

    return DiscoveryReport(
        channels_configured=len(channels),
        channels_enabled=len(enabled),
        discovered_total=discovered_total,
        newly_queued=len(new_entries),
        duplicates_skipped=duplicates_skipped,
        feed_failures=feed_failures,
    )
