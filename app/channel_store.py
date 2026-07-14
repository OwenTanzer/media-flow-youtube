"""Reads the operator-managed channel registry (`channels.json`) from the
Drive folder: which YouTube channels to poll for new uploads, and any
per-channel overrides. Purely read-only from the app's side - the
registry is meant to be edited directly in Drive, no redeploy required."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from . import drive
from .config import settings

logger = logging.getLogger("media_flow.channels")

CHANNELS_FILENAME = "channels.json"


@dataclass
class Channel:
    channel_id: str
    name: str
    enabled: bool = True
    languages: list[str] | None = None
    # The dashboard's top-level classification (issue #8) - e.g. "Finance"
    # or "Google". Left as None (rather than defaulted here) when absent
    # from the registry, so a caller can distinguish "explicitly set" from
    # "unset, needs a fallback" - the fallback itself belongs at the point
    # group is consumed, not here.
    group: str | None = None


def read_channels(folder_id: str) -> list[Channel]:
    if settings.dry_run:
        return []

    text = drive.download_text(folder_id, CHANNELS_FILENAME)
    if text is None:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("channels.json in folder %s was not valid JSON; treating as empty.", folder_id)
        return []

    raw_channels = data.get("channels") if isinstance(data, dict) else None
    if not isinstance(raw_channels, list):
        logger.warning("channels.json in folder %s has no \"channels\" list; treating as empty.", folder_id)
        return []

    channels: list[Channel] = []
    for entry in raw_channels:
        if not isinstance(entry, dict) or not isinstance(entry.get("channel_id"), str):
            logger.warning("Skipping malformed channels.json entry: %r", entry)
            continue
        languages = entry.get("languages")
        group = entry.get("group")
        channels.append(
            Channel(
                channel_id=entry["channel_id"],
                name=str(entry.get("name") or entry["channel_id"]),
                enabled=bool(entry.get("enabled", True)),
                languages=[str(code) for code in languages] if isinstance(languages, list) and languages else None,
                group=group if isinstance(group, str) and group.strip() else None,
            )
        )
    return channels
