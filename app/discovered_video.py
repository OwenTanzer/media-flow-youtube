"""Shared result type for discovery.py's two fetch backends (RSS feeds and
the YouTube Data API uploads playlist, app/youtube_data_api.py) - split out
of discovery.py so youtube_data_api.py can return it without an import cycle
(discovery.py imports youtube_data_api.py, not the other way around)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiscoveredVideo:
    video_id: str
    channel_id: str
    published: str | None
