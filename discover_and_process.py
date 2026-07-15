#!/usr/bin/env python3
"""Standalone entrypoint that discovers new uploads from configured YouTube
channels (channels.json in the Drive folder) and processes them through
the existing queue pipeline, in one serialized invocation. Deploy as its
own Railway "Cron Job" service (Custom Start Command:
`python discover_and_process.py`).

Concurrency invariant: do not also enable ENABLE_SCHEDULER or deploy
batch_runner.py on the same schedule - this script already runs the queue
processor as its second half, and queue.json/_index.json aren't safe for
concurrent writers. An advisory Drive-based lock (app/job_lock.py)
prevents two invocations of *this* script from overlapping, but it can't
protect against running this alongside a different queue-writing job.
"""

import logging
import sys

from app import job_lock
from app.batch import run_batch
from app.config import ConfigError, settings
from app.discovery import discover_and_enqueue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media_flow.discover_and_process")


def main() -> int:
    try:
        folder_id = settings.require_drive_folder_id()
    except ConfigError as exc:
        logger.error("Run aborted: %s", exc)
        return 1

    lock_token = job_lock.acquire_lock(folder_id, settings.discovery_lock_ttl_seconds)
    if lock_token is None:
        logger.error(
            "Another discover-and-process run appears to be in progress (lock held, "
            "age < %ds), or lost a race to acquire the lock; exiting without acting.",
            settings.discovery_lock_ttl_seconds,
        )
        return 1

    try:
        report = discover_and_enqueue(folder_id)
        logger.info(
            "Discovery: %d channel(s) configured, %d enabled, %d upload(s) seen, "
            "%d newly queued, %d duplicate(s) skipped, %d feed failure(s).",
            report.channels_configured,
            report.channels_enabled,
            report.discovered_total,
            report.newly_queued,
            report.duplicates_skipped,
            len(report.feed_failures),
        )
        for channel_id, message in report.feed_failures:
            logger.warning("  feed failure - channel %s: %s", channel_id, message)

        def _renew_lock() -> None:
            # A large queue's cooldowns alone can comfortably exceed
            # DISCOVERY_LOCK_TTL_SECONDS, so keep the lease fresh after every
            # entry (not just every chunk - a single entry's own internal
            # retries can themselves run for minutes) rather than letting a
            # healthy run look crashed. If the lock no longer belongs to us
            # (lost to a concurrent run), stop working immediately instead of
            # continuing to write against a lease we no longer hold.
            if not job_lock.renew_lock(folder_id, lock_token):
                raise RuntimeError("Lost the discovery lock mid-run; aborting to avoid racing a new owner.")

        results = run_batch(on_progress=_renew_lock)
        ok = sum(1 for r in results if r.status == "ok")
        logger.info("Processing complete: %d fetched, %d skipped/errored, %d total.", ok, len(results) - ok, len(results))
        for r in results:
            if r.status != "ok":
                logger.info("  %s (%s): %s", r.video_id or r.url, r.status, r.message)

        return 0
    finally:
        job_lock.release_lock(folder_id, lock_token)


if __name__ == "__main__":
    sys.exit(main())
