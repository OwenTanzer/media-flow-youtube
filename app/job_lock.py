"""A Drive-based advisory lock preventing two discover-and-process runs
from overlapping. queue.json/_index.json use unlocked read-modify-write
Drive operations that only tolerate one writer at a time - this lock
stops a second concurrent invocation of the same job, not a full
distributed compare-and-swap (out of scope; see README's concurrency
invariant section)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from . import drive
from .config import settings

logger = logging.getLogger("media_flow.job_lock")

LOCK_FILENAME = "_discovery_lock.json"


def acquire_lock(folder_id: str, ttl_seconds: int) -> bool:
    """Returns True if the lock was acquired (either no lock existed, or
    the existing one is older than ttl_seconds and treated as a crashed
    prior run). Returns False without writing anything if a fresh lock is
    already held."""

    if settings.dry_run:
        return True

    text = drive.download_text(folder_id, LOCK_FILENAME)
    if text is not None:
        acquired_at = None
        try:
            acquired_at = datetime.fromisoformat(json.loads(text)["acquired_at"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("Existing lock file in folder %s is unreadable; treating it as stale.", folder_id)

        if acquired_at is not None:
            age_seconds = (datetime.now(timezone.utc) - acquired_at).total_seconds()
            if age_seconds < ttl_seconds:
                return False
            logger.warning(
                "Lock in folder %s is %.0fs old (ttl %ds) - treating the prior run as crashed and proceeding.",
                folder_id,
                age_seconds,
                ttl_seconds,
            )

    payload = json.dumps({"acquired_at": datetime.now(timezone.utc).isoformat()})
    drive.upload_text_file(folder_id, LOCK_FILENAME, payload, mime_type="application/json")
    return True


def release_lock(folder_id: str) -> None:
    if settings.dry_run:
        return
    drive.delete_file(folder_id, LOCK_FILENAME)
