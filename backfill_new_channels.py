#!/usr/bin/env python3
"""One-off/rerunnable maintenance script: backfills whatever videos are
currently visible in the RSS feed of any channel in channels.json that
has never been discovered at all - i.e. any channel added since the last
time this ran (or since discover_and_process.py's own discovery step
happened to see it first). YouTube's feed only exposes a channel's most
recent uploads (see backfill_channel_ids.py's docstring for the same
limitation), so "backfill" here means "pick up what's there now", not a
full historical archive.

Deliberately separate from discover_and_process.py and its advisory Drive
lock (app/job_lock.py, LOCK_FILENAME): that lock guards the full
discover+batch+summarize run, which can legitimately take a long time
(see BATCH_SIZE_THRESHOLD/BATCH_COOLDOWN_SECONDS), and there's no reason
a newly-added channel's one-off backfill - a single feed fetch and queue
append per new channel, seconds long - should have to wait for or
contend with it. This script acquires its own independent lock
(NEW_CHANNEL_BACKFILL_LOCK_FILENAME) instead, only to stop two concurrent
invocations of *this* script from racing each other.

Idempotent and safe to rerun: a channel already backfilled (has at least
one video in _index.json or queue.json) is skipped, so rerunning after
new channels are added only processes the new ones.

Usage: python backfill_new_channels.py
"""

import logging
import sys

from app import job_lock
from app.config import ConfigError, settings
from app.discovery import backfill_new_channels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media_flow.backfill_new_channels")

# Generous relative to how long this script actually takes (a handful of
# feed fetches), but still bounded - matches the spirit of
# DISCOVERY_LOCK_TTL_SECONDS without needing its own settings knob for a
# script this small and infrequently run.
LOCK_TTL_SECONDS = 300


def main() -> int:
    try:
        folder_id = settings.require_drive_folder_id()
    except ConfigError as exc:
        logger.error("Run aborted: %s", exc)
        return 1

    lock_token = job_lock.acquire_lock(
        folder_id, LOCK_TTL_SECONDS, lock_filename=job_lock.NEW_CHANNEL_BACKFILL_LOCK_FILENAME
    )
    if lock_token is None:
        logger.error(
            "Another backfill_new_channels run appears to be in progress (lock held, "
            "age < %ds), or lost a race to acquire the lock; exiting without acting.",
            LOCK_TTL_SECONDS,
        )
        return 1

    try:
        report = backfill_new_channels(folder_id)
        if report.channels_configured == 0:
            logger.info("No channels need an initial backfill - every enabled channel already has at least one known video.")
            return 0

        logger.info(
            "Backfill: %d never-discovered channel(s), %d upload(s) seen, %d newly queued, "
            "%d duplicate(s) skipped, %d feed failure(s).",
            report.channels_configured,
            report.discovered_total,
            report.newly_queued,
            report.duplicates_skipped,
            len(report.feed_failures),
        )
        for channel_id, message in report.feed_failures:
            logger.warning("  feed failure - channel %s: %s", channel_id, message)

        return 0
    finally:
        job_lock.release_lock(folder_id, lock_token, lock_filename=job_lock.NEW_CHANNEL_BACKFILL_LOCK_FILENAME)


if __name__ == "__main__":
    sys.exit(main())
