"""Batch processing: work through either an explicit list of URLs or the
Drive-hosted queue, and (for the queue case) remove entries once handled
so re-running the job doesn't refetch everything."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from . import queue_store
from .config import settings
from .models import VideoResult
from .pipeline import safe_process_video

logger = logging.getLogger("media_flow.batch")

_TERMINAL_STATUSES = ("ok", "no_captions", "unavailable", "invalid_url")


def _within_no_captions_grace_period(entry: str | dict) -> bool:
    """A video discovered very recently (e.g. an in-progress livestream)
    may not have captions yet purely because it hasn't ended, or YouTube
    hasn't finished processing them. Give entries with a known discovery
    time (see app/discovery.py) a grace period of retries before
    "no_captions" is treated as permanent. Manually-added entries with no
    first_seen_at get no grace period, same as before this existed."""

    first_seen_at = queue_store.entry_first_seen_at(entry)
    if first_seen_at is None:
        return False
    age = datetime.now(timezone.utc) - first_seen_at
    return age < timedelta(hours=settings.no_captions_grace_hours)


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


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

    def _process_one(entry: str | dict) -> None:
        video_url = queue_store.entry_url(entry) if use_queue else entry
        video_languages = queue_store.entry_languages(entry, languages) if use_queue else languages
        result = safe_process_video(video_url, video_languages)
        results.append(result)

        if not use_queue:
            return

        is_terminal = result.status in _TERMINAL_STATUSES
        if is_terminal and result.status == "no_captions" and _within_no_captions_grace_period(entry):
            is_terminal = False
        if not is_terminal:
            # Transient failure (e.g. rate limiting, or a livestream still within
            # its no-captions grace period) - keep it in the queue for next
            # run, preserving the original str/dict shape so overrides survive.
            remaining.append(entry)

    # A long, continuous run of requests through the rotating proxy pool
    # measurably degrades its success rate (observed empirically - see the
    # egress proxy section of the README). Above BATCH_SIZE_THRESHOLD
    # entries, process in smaller chunks with a real cooldown between them
    # so the pool gets a chance to recover, rather than burning through the
    # whole thing in one continuous burst.
    if len(entries) > settings.batch_size_threshold:
        chunks = _chunk(entries, settings.batch_size_threshold)
        logger.info(
            "%d entries exceeds the %d-entry batching threshold; processing in %d chunk(s) of "
            "%d with %.0fs cooldowns between them.",
            len(entries),
            settings.batch_size_threshold,
            len(chunks),
            settings.batch_size_threshold,
            settings.batch_cooldown_seconds,
        )
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(settings.batch_cooldown_seconds)
            for entry in chunk:
                _process_one(entry)
    else:
        for entry in entries:
            _process_one(entry)

    if use_queue:
        queue_store.write_queue(folder_id, remaining)

    return results
