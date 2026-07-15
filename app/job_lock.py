"""A Drive-based advisory lock preventing two discover-and-process runs
from overlapping. queue.json/_index.json use unlocked read-modify-write
Drive operations that only tolerate one writer at a time - this lock
stops a second concurrent invocation of the same job, not a full
distributed compare-and-swap (out of scope; see README's concurrency
invariant section).

Every writer of queue.json/_index.json must acquire this same lock, not
a lock of its own - a distinct lock only serializes a caller against
itself, but does nothing to stop it from racing a *different* writer of
the same underlying file. (An earlier version of backfill_new_channels.py
took out its own independent lock for exactly this reason and was wrong:
it could still interleave with discover_and_process.py's batch checkpoint
on queue.json and silently lose or resurrect entries. See that script and
vidproc/admin.py for how they now share this lock instead.)

acquire_lock()'s check-then-write is not atomic across processes, and
Drive additionally permits duplicate filenames within one folder, so two
near-simultaneous invocations could otherwise both believe they hold the
lock. Two mitigations narrow (without fully eliminating) that window:

- Every lock carries a random ownership token. release_lock() and the
  stale-lock takeover in acquire_lock() only ever act on a lock whose
  token they recognize, so a slow/crashed run can never delete or steal
  a *different* run's active lease.
- After writing the lock, acquire_lock() immediately re-reads it and
  confirms exactly one file with that name exists and it's the one this
  call just wrote. If a concurrent writer created a second file with the
  same name (Drive's duplicate-filename hazard) or overwrote this one,
  both callers back off rather than proceeding as if each holds the lock.

A long batch run (see BATCH_SIZE_THRESHOLD/BATCH_COOLDOWN_SECONDS in
app/batch.py) can legitimately take far longer than ttl_seconds purely
from its own cooldowns - without renewal, acquire_lock()'s stale-lock
takeover would eventually treat a perfectly healthy run as crashed and
let a second invocation start concurrently. renew_lock() lets a
long-running caller refresh its own lease periodically to prevent that.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from . import drive
from .config import settings

logger = logging.getLogger("media_flow.job_lock")

LOCK_FILENAME = "_discovery_lock.json"


def _read_lock(folder_id: str) -> tuple[datetime | None, str | None]:
    """Returns (acquired_at, token) for the current lock file, or (None,
    None) if it doesn't exist or isn't readable."""

    text = drive.download_text(folder_id, LOCK_FILENAME)
    if text is None:
        return None, None
    try:
        payload = json.loads(text)
        return datetime.fromisoformat(payload["acquired_at"]), payload.get("token")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("Existing lock file in folder %s is unreadable; treating it as stale.", folder_id)
        return None, None


def acquire_lock(folder_id: str, ttl_seconds: int) -> str | None:
    """Returns a lock token (an opaque string to pass to release_lock) if
    acquired - either no lock existed, or the existing one is older than
    ttl_seconds and treated as a crashed prior run. Returns None without
    acting if a fresh lock is already held, or if a concurrent writer is
    detected immediately after this call writes its own lock."""

    if settings.dry_run:
        return "dry-run-lock-token"

    acquired_at, _ = _read_lock(folder_id)
    if acquired_at is not None:
        age_seconds = (datetime.now(timezone.utc) - acquired_at).total_seconds()
        if age_seconds < ttl_seconds:
            return None
        logger.warning(
            "Lock in folder %s is %.0fs old (ttl %ds) - treating the prior run as crashed and proceeding.",
            folder_id,
            age_seconds,
            ttl_seconds,
        )

    token = uuid.uuid4().hex
    payload = json.dumps({"acquired_at": datetime.now(timezone.utc).isoformat(), "token": token})
    drive.upload_text_file(folder_id, LOCK_FILENAME, payload, mime_type="application/json")

    file_ids = drive.list_file_ids(folder_id, LOCK_FILENAME)
    if len(file_ids) != 1:
        logger.error(
            "Lock acquisition for folder %s raced with a concurrent writer (%d lock "
            "files present); backing off without proceeding.",
            folder_id,
            len(file_ids),
        )
        return None

    _, confirmed_token = _read_lock(folder_id)
    if confirmed_token != token:
        logger.error("Lost the acquire race for folder %s's lock to a concurrent run; backing off.", folder_id)
        return None

    return token


def renew_lock(folder_id: str, token: str) -> bool:
    """Rewrites the lock with a fresh acquired_at timestamp, keeping the
    same token, so a long-running batch (see BATCH_SIZE_THRESHOLD/
    BATCH_COOLDOWN_SECONDS in app/batch.py) doesn't have its own lease
    time out from ttl_seconds while it's still legitimately working.
    Returns False without writing anything if the lock no longer belongs
    to this token (lost to a concurrent run, or deleted unexpectedly) -
    callers should treat that as a signal to stop, not keep going as if
    still holding the lock.

    Like acquire_lock(), the check-then-write here isn't atomic: a
    takeover can still land in the gap between the ownership check and
    the write. Re-reads after writing (same as acquire_lock()) so that
    race is at least detected - a takeover in that exact window makes this
    call correctly return False rather than incorrectly reporting success,
    even though the write briefly clobbered the new owner's lease."""

    if settings.dry_run:
        return True

    _, current_token = _read_lock(folder_id)
    if current_token != token:
        logger.error(
            "Cannot renew the lock in folder %s: it no longer belongs to this run "
            "(expected token %r, found %r). Another run may already be in progress.",
            folder_id,
            token,
            current_token,
        )
        return False

    payload = json.dumps({"acquired_at": datetime.now(timezone.utc).isoformat(), "token": token})
    drive.upload_text_file(folder_id, LOCK_FILENAME, payload, mime_type="application/json")

    _, confirmed_token = _read_lock(folder_id)
    if confirmed_token != token:
        logger.error(
            "Lost the lock in folder %s to a concurrent run immediately after renewing - a takeover "
            "landed between the ownership check and the write.",
            folder_id,
        )
        return False

    return True


def release_lock(folder_id: str, token: str) -> None:
    """Only deletes the lock file if it still belongs to this token, so a
    run that outlives its own TTL can't delete a different run's lease."""

    if settings.dry_run:
        return

    _, current_token = _read_lock(folder_id)
    if current_token is None:
        return
    if current_token != token:
        logger.warning(
            "Not releasing the lock in folder %s: it belongs to a different run than the one releasing it.",
            folder_id,
        )
        return
    drive.delete_file(folder_id, LOCK_FILENAME)
