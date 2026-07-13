"""Batch processing: work through either an explicit list of URLs or the
Drive-hosted queue, and (for the queue case) remove entries once handled
so re-running the job doesn't refetch everything."""

from __future__ import annotations

import logging

from . import queue_store
from .config import settings
from .models import VideoResult
from .pipeline import safe_process_video

logger = logging.getLogger("media_flow.batch")

_TERMINAL_STATUSES = ("ok", "no_captions", "unavailable", "invalid_url")


def run_batch(urls: list[str] | None = None, languages: list[str] | None = None) -> list[VideoResult]:
    folder_id = settings.require_drive_folder_id()
    use_queue = urls is None

    if use_queue:
        entries: list[str | dict] = queue_store.read_queue(folder_id)
        if not entries:
            logger.info("Queue is empty, nothing to process.")
            return []
    else:
        entries = urls

    results: list[VideoResult] = []
    remaining: list[str | dict] = []
    for entry in entries:
        video_url = queue_store.entry_url(entry) if use_queue else entry
        video_languages = queue_store.entry_languages(entry, languages) if use_queue else languages
        result = safe_process_video(video_url, video_languages)
        results.append(result)
        if use_queue and result.status not in _TERMINAL_STATUSES:
            # Transient failure (e.g. rate limiting) - keep it in the queue for next
            # run, preserving the original str/dict shape so a language override survives.
            remaining.append(entry)

    if use_queue:
        queue_store.write_queue(folder_id, remaining)

    return results
