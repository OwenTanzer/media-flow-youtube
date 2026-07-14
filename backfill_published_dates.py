#!/usr/bin/env python3
"""One-off/rerunnable maintenance script: backfills the "published_at"
field (see app/discovery.py, app/queue_store.py, app/summary_store.py) for
already-archived videos that predate that field's existence.

Matches video IDs against each configured channel's *current* RSS feed -
the only source this app has for a video's real publish date. YouTube's
feed only exposes a channel's most recent uploads, so an older video may
no longer be present in it; those are left untouched (already-absent
published_at, same as before this script ran) and reported separately
rather than guessed at.

Idempotent and safe to rerun: only fills in published_at where currently
absent on a status: "ok" index entry - never overwrites an existing
value, never touches non-"ok" entries. Updates _index.json, the
transcript file's own frontmatter, and any existing summary artifact, so
all three stay consistent rather than only the index.

Usage: python backfill_published_dates.py
"""

import logging
import re
import sys

from app import channel_store, discovery, drive, summary_store
from app.config import ConfigError, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media_flow.backfill_published_dates")

_FETCHED_AT_LINE_RE = re.compile(r"(?m)^(fetched_at: .+)$")


def build_published_at_map(folder_id: str) -> dict[str, str]:
    """Maps video_id -> published (ISO string) for every video still
    present in any configured channel's current RSS feed. A channel whose
    feed fetch fails is skipped (logged), not fatal to the whole run."""

    channels = channel_store.read_channels(folder_id)
    published_at_by_video_id: dict[str, str] = {}
    for channel in channels:
        try:
            videos = discovery.fetch_channel_feed(channel.channel_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feed fetch failed for channel %s (%s): %s", channel.channel_id, channel.name, exc)
            continue
        for video in videos:
            if video.published:
                published_at_by_video_id[video.video_id] = video.published
    return published_at_by_video_id


def _patch_transcript_frontmatter(folder_id: str, filename: str, published_at: str) -> None:
    """Best-effort: inserts a published_at line right after fetched_at,
    matching youtube.render_transcript_markdown()'s own field order.
    Logs and continues on any failure - the index entry is the durable
    source of truth; the transcript file's frontmatter is a convenience
    that's fine to leave stale if this one video's patch fails."""

    try:
        markdown = drive.download_text(folder_id, filename)
        if markdown is None:
            return
        frontmatter_end = markdown.find("\n---\n")
        if frontmatter_end != -1 and "published_at:" in markdown[: frontmatter_end + 5]:
            return  # already has it somehow - don't duplicate the line
        patched, count = _FETCHED_AT_LINE_RE.subn(rf"\1\npublished_at: {published_at}", markdown, count=1)
        if count == 0:
            logger.warning("Could not find a fetched_at line to patch in %r; leaving it as-is.", filename)
            return
        drive.upload_text_file(folder_id, filename, patched)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to patch transcript frontmatter for %r", filename)


def _patch_summary_artifact(folder_id: str, video_id: str, published_at: str) -> None:
    try:
        existing = summary_store.read_summary(folder_id, video_id)
        if existing is None or existing.get("video_published_at"):
            return
        existing["video_published_at"] = published_at
        summary_store.write_summary(folder_id, video_id, existing)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to patch summary artifact for %s", video_id)


def main() -> int:
    try:
        folder_id = settings.require_drive_folder_id()
    except ConfigError as exc:
        logger.error("Run aborted: %s", exc)
        return 1

    published_at_by_video_id = build_published_at_map(folder_id)
    logger.info(
        "Found publish dates for %d video(s) still present in configured channels' current RSS feeds.",
        len(published_at_by_video_id),
    )

    index = drive.read_index(folder_id)
    updated = 0
    already_had_it = 0
    not_found = 0

    for video_id, entry in index.items():
        if entry.get("status") != "ok":
            continue
        if entry.get("published_at"):
            already_had_it += 1
            continue
        published_at = published_at_by_video_id.get(video_id)
        if published_at is None:
            not_found += 1
            continue

        entry["published_at"] = published_at
        drive.update_index_entry(folder_id, video_id, entry)

        filename = entry.get("filename")
        if filename:
            _patch_transcript_frontmatter(folder_id, filename, published_at)
        _patch_summary_artifact(folder_id, video_id, published_at)

        updated += 1

    logger.info(
        "Backfill complete: %d updated, %d already had a publish date, %d not found in any "
        "configured channel's current feed (too old for RSS - left untouched).",
        updated,
        already_had_it,
        not_found,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
