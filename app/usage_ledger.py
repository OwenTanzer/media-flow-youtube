"""Append-only ledger of every Claude summarization attempt, in
`_usage_ledger.json` in the Drive folder.

Unlike a summary artifact (summaries/<video_id>.json), which is overwritten
in place every time that video is (re)attempted - a failed attempt followed
by a successful retry replaces the failure and its usage; an explicit
SUMMARY_FORCE_RESUMMARIZE_VIDEO_IDS replacement replaces the previous
successful usage - this ledger is never overwritten, only appended to. One
entry is recorded per attempt, success or failure, so aggregate totals (see
app/insights_store.py's CostUsageSummary) reflect true lifetime spend, not
just whatever happens to be on the currently-live artifacts."""

from __future__ import annotations

import json
import logging

from . import drive, job_lock

logger = logging.getLogger("media_flow.usage_ledger")

LEDGER_FILENAME = "_usage_ledger.json"

# A dedicated lock, independent of job_lock.LOCK_FILENAME (_discovery_lock.json) -
# summarize_backlog.py deliberately runs without the discovery lock (see its
# own module docstring: it never touches queue.json/_index.json, and isn't
# meant to wait on discovery/batch runs). Serializing ledger writes must not
# couple back to that lock, or this worker would start blocking on
# discover_and_process.py runs it was explicitly designed to be independent
# of.
LEDGER_LOCK_FILENAME = "_usage_ledger_lock.json"

# The read-modify-write below is one small JSON file read+write, not a
# multi-minute batch - nowhere near long enough to need discovery's 1800s
# default TTL. A short TTL here means a crashed run's lease is reclaimed
# quickly rather than blocking every subsequent append for half an hour.
_LOCK_TTL_SECONDS = 120


class UsageLedgerCorruptError(RuntimeError):
    """Raised when _usage_ledger.json exists but isn't valid JSON, or isn't
    a list. append_entries() deliberately does not swallow this into an
    empty list the way a read-only display path reasonably could - doing
    so would then overwrite the corrupt file with just the new entries,
    silently discarding whatever history it still held (data that might
    otherwise be recoverable by hand). A display-only reader (see
    app/insights_store.py's load_snapshot()) is expected to catch this
    itself and degrade gracefully, same as its other malformed-data cases."""


def read_ledger(folder_id: str) -> list[dict]:
    """Raises UsageLedgerCorruptError if the file exists but can't be
    parsed as a JSON list - see that class's docstring for why this
    deliberately isn't swallowed here the way a missing file (a totally
    normal, expected state) is."""

    text = drive.download_text(folder_id, LEDGER_FILENAME)
    if text is None:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UsageLedgerCorruptError(f"{LEDGER_FILENAME} in folder {folder_id} is not valid JSON.") from exc
    if not isinstance(data, list):
        raise UsageLedgerCorruptError(f"{LEDGER_FILENAME} in folder {folder_id} does not contain a JSON list.")
    return data


def append_entries(folder_id: str, entries: list[dict]) -> None:
    """Appends every entry in one call under a dedicated advisory lock
    (LEDGER_LOCK_FILENAME), so two concurrent writers (e.g. two overlapping
    summarize_backlog.py invocations - this worker deliberately runs
    without the discovery lock, so overlap is possible) can't race the
    same read-modify-write: without the lock, both could read the same
    existing list and each write back existing+their own entries, with the
    second write silently discarding the first's.

    Idempotent per entry via "attempt_id" (see backlog_summarizer.py,
    which generates a fresh uuid4 per attempt): an entry whose attempt_id
    is already present in the ledger is skipped rather than appended
    again, so a caller that (for whatever reason) submits the same
    already-recorded entry twice can't double-count it.

    Fails closed on a corrupt existing ledger (UsageLedgerCorruptError from
    read_ledger()): logs an error and returns without writing anything,
    rather than overwriting a corrupt file with just this call's entries
    and losing whatever it still held. This call's entries are then not
    durably recorded - callers should surface that loss loudly (see
    backlog_summarizer.py's per-attempt call site) rather than silently
    treating this as a successful append.

    Does nothing at all (no lock, no read, no write) when entries is
    empty."""

    if not entries:
        return

    token = job_lock.acquire_lock(folder_id, _LOCK_TTL_SECONDS, lock_filename=LEDGER_LOCK_FILENAME)
    if token is None:
        logger.error(
            "Could not acquire the usage ledger lock for folder %s - a concurrent writer holds it. "
            "%d usage-ledger entries were NOT durably recorded this call.",
            folder_id, len(entries),
        )
        return

    try:
        try:
            existing = read_ledger(folder_id)
        except UsageLedgerCorruptError:
            logger.error(
                "%s in folder %s is corrupt - refusing to overwrite it. %d usage-ledger entries were NOT "
                "durably recorded this call. Inspect/restore the file manually before further appends can "
                "succeed.",
                LEDGER_FILENAME, folder_id, len(entries),
            )
            return

        existing_ids = {entry.get("attempt_id") for entry in existing if isinstance(entry, dict)}
        new_entries = [entry for entry in entries if entry.get("attempt_id") not in existing_ids]
        if not new_entries:
            return

        payload = json.dumps(existing + new_entries, indent=2)
        drive.upload_text_file(folder_id, LEDGER_FILENAME, payload, mime_type="application/json")
    finally:
        job_lock.release_lock(folder_id, token, lock_filename=LEDGER_LOCK_FILENAME)
