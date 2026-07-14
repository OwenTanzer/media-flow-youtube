#!/usr/bin/env python3
"""One-off/rerunnable maintenance script: backfills whatever videos are
currently visible in the RSS feed of any channel in channels.json that
has never been discovered at all - i.e. any channel added since the last
time this ran (or since discover_and_process.py's own discovery step
happened to see it first). YouTube's feed only exposes a channel's most
recent uploads (see backfill_channel_ids.py's docstring for the same
limitation), so "backfill" here means "pick up what's there now", not a
full historical archive.

Shares discover_and_process.py's own advisory Drive lock
(app/job_lock.py, LOCK_FILENAME) rather than a lock of its own: both this
script and discover_and_process.py's batch checkpoint read-modify-write
the *same* queue.json, and a distinct lock would only serialize this
script against itself while doing nothing to stop it from interleaving
with - and silently corrupting - a concurrently-running
discover_and_process.py. See app/job_lock.py's module docstring.

This deliberately does not wait for the lock: if discover_and_process.py
is already running, this exits immediately rather than blocking, so
running this never introduces a waiting period of its own - it just
means this particular invocation did no work, and the channel will still
be picked up by that in-progress run or the next one either way.

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


def main() -> int:
    try:
        folder_id = settings.require_drive_folder_id()
    except ConfigError as exc:
        logger.error("Run aborted: %s", exc)
        return 1

    lock_token = job_lock.acquire_lock(folder_id, settings.discovery_lock_ttl_seconds)
    if lock_token is None:
        logger.error(
            "discover_and_process.py appears to be running right now (lock held); exiting "
            "without acting rather than waiting - that run (or the next one) will pick up "
            "any new channels anyway. Rerun this once it finishes if you don't want to wait."
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
        job_lock.release_lock(folder_id, lock_token)


if __name__ == "__main__":
    sys.exit(main())
