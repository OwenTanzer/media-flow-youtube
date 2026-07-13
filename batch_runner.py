#!/usr/bin/env python3
"""Standalone entrypoint for processing the Drive-hosted queue without
running the web server. Deploy this as a separate Railway "Cron Job"
service (Custom Start Command: `python batch_runner.py`) if you'd rather
schedule batches at the platform level instead of using the in-process
APScheduler (ENABLE_SCHEDULER) in the web service.
"""

import logging
import sys

from app.batch import run_batch
from app.config import ConfigError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media_flow.batch_runner")


def main() -> int:
    try:
        results = run_batch()
    except ConfigError as exc:
        logger.error("Batch run aborted: %s", exc)
        return 1

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status != "ok")
    logger.info("Batch run complete: %d fetched, %d skipped/errored, %d total.", ok, skipped, len(results))
    for r in results:
        if r.status != "ok":
            logger.info("  %s (%s): %s", r.video_id or r.url, r.status, r.message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
