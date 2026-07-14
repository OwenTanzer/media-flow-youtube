#!/usr/bin/env python3
"""One-off/rerunnable maintenance script: backfills the "channel_id" field
(see app/discovery.py, app/queue_store.py, app/pipeline.py,
app/summary_store.py) for already-archived videos that predate that
field's existence.

Matches video IDs against each configured channel's *current* RSS feed -
the only source this app has for which channel a video came from. Unlike
backfill_published_dates.py, this doesn't need a per-video published date
out of the feed: every video returned by fetch_channel_feed(channel_id) is
already known to belong to that one channel, so the map is built for free
while iterating. YouTube's feed only exposes a channel's most recent
uploads, so an older video may no longer be present in it; those are left
untouched (already-absent channel_id, same as before this script ran) and
reported separately rather than guessed at.

Idempotent and safe to rerun: only fills in channel_id where currently
absent on a status: "ok" index entry - never overwrites an existing value,
never touches non-"ok" entries. Updates both _index.json and any existing
summary artifact so they stay consistent. There's no transcript
frontmatter field to patch here (unlike published_at) - the transcript's
own "channel:" line is a separate, free-text field, not derived from this.

Usage: python backfill_channel_ids.py
"""

import logging
import sys

from app import channel_store, discovery, drive, summary_store
from app.config import ConfigError, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media_flow.backfill_channel_ids")


def build_channel_id_map(folder_id: str) -> dict[str, str]:
    """Maps video_id -> channel_id for every video still present in any
    configured channel's current RSS feed. A channel whose feed fetch
    fails is skipped (logged), not fatal to the whole run."""

    channels = channel_store.read_channels(folder_id)
    channel_id_by_video_id: dict[str, str] = {}
    for channel in channels:
        try:
            videos = discovery.fetch_channel_feed(channel.channel_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feed fetch failed for channel %s (%s): %s", channel.channel_id, channel.name, exc)
            continue
        for video in videos:
            channel_id_by_video_id[video.video_id] = channel.channel_id
    return channel_id_by_video_id


def _patch_summary_artifact(folder_id: str, video_id: str, channel_id: str) -> None:
    try:
        existing = summary_store.read_summary(folder_id, video_id)
        if existing is None or existing.get("channel_id"):
            return
        existing["channel_id"] = channel_id
        summary_store.write_summary(folder_id, video_id, existing)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to patch summary artifact for %s", video_id)


def main() -> int:
    try:
        folder_id = settings.require_drive_folder_id()
    except ConfigError as exc:
        logger.error("Run aborted: %s", exc)
        return 1

    channel_id_by_video_id = build_channel_id_map(folder_id)
    logger.info(
        "Found a channel for %d video(s) still present in configured channels' current RSS feeds.",
        len(channel_id_by_video_id),
    )

    index = drive.read_index(folder_id)
    updated = 0
    already_had_it = 0
    not_found = 0

    for video_id, entry in index.items():
        if entry.get("status") != "ok":
            continue
        if entry.get("channel_id"):
            already_had_it += 1
            continue
        channel_id = channel_id_by_video_id.get(video_id)
        if channel_id is None:
            not_found += 1
            continue

        entry["channel_id"] = channel_id
        drive.update_index_entry(folder_id, video_id, entry)
        _patch_summary_artifact(folder_id, video_id, channel_id)

        updated += 1

    logger.info(
        "Backfill complete: %d updated, %d already had a channel_id, %d not found in any "
        "configured channel's current feed (too old for RSS - left untouched).",
        updated,
        already_had_it,
        not_found,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
