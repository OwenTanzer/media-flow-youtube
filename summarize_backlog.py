#!/usr/bin/env python3
"""Independent, fast backlog summarizer.

Unlike discover_and_process.py this never mutates queue.json or _index.json,
so it intentionally does not use the Drive discovery lock. It may run while
RSS discovery, transcript retrieval, or channel backfills are in progress.
"""

import logging
import sys

from app.backlog_summarizer import summarize_backlog
from app.config import ConfigError, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media_flow.summarize_backlog")


def main() -> int:
    try:
        folder_id = settings.require_drive_folder_id()
    except ConfigError as exc:
        logger.error("Run aborted: %s", exc)
        return 1
    report = summarize_backlog(folder_id)
    logger.info(
        "Summaries: %d eligible, %d already completed (skipped), %d force-resummarized, %d structured, %d failed "
        "(~%d input / ~%d output tokens, ~$%.4f).",
        report.eligible, report.skipped_current, report.forced, report.summarized, report.failed,
        report.input_tokens, report.output_tokens, report.estimated_cost_usd,
    )
    for video_id, message in report.failures:
        logger.warning("  summary failure - %s: %s", video_id, message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
