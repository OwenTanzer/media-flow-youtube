"""Batch processing: work through either an explicit list of URLs or the
Drive-hosted queue, and (for the queue case) remove entries once handled
so re-running the job doesn't refetch everything."""

from __future__ import annotations

import logging

from . import queue_store
from .config import settings
from .models import VideoResult
from .pipeline import process_video

logger = logging.getLogger("media_flow.batch")


def run_batch(urls: list[str] | None = None, languages: list[str] | None = None) -> list[VideoResult]:
    folder_id = settings.require_drive_folder_id()
    use_queue = urls is None

    if use_queue:
        urls = queue_store.read_queue(folder_id)
        if not urls:
            logger.info("Queue is empty, nothing to process.")
            return []

    results: list[VideoResult] = []
    remaining: list[str] = []
    for url in urls:
        try:
            result = process_video(url, languages)
            results.append(result)
            if result.status not in ("ok", "no_captions", "unavailable", "invalid_url"):
                # Transient failure (e.g. rate limiting) - keep it in the queue for next run.
                remaining.append(url)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error processing %s", url)
            results.append(VideoResult(video_id="", url=url, status="error", message=str(exc)))
            if use_queue:
                remaining.append(url)

    if use_queue:
        queue_store.write_queue(folder_id, remaining)

    return results
