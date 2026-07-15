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

from . import drive

logger = logging.getLogger("media_flow.usage_ledger")

LEDGER_FILENAME = "_usage_ledger.json"


def read_ledger(folder_id: str) -> list[dict]:
    text = drive.download_text(folder_id, LEDGER_FILENAME)
    if text is None:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("%s in folder %s was not valid JSON; treating as empty.", LEDGER_FILENAME, folder_id)
        return []
    return data if isinstance(data, list) else []


def append_entries(folder_id: str, entries: list[dict]) -> None:
    """Appends every entry from one summarize_backlog() run in a single
    read-modify-write, called once after that run's ThreadPoolExecutor has
    already finished draining every job - from that one calling thread, so
    this never races itself the way one write-per-attempt issued directly
    from each worker thread would. Does nothing (no read, no write) when
    entries is empty, so a run with no eligible videos leaves the ledger
    untouched."""

    if not entries:
        return
    existing = read_ledger(folder_id)
    payload = json.dumps(existing + entries, indent=2)
    drive.upload_text_file(folder_id, LEDGER_FILENAME, payload, mime_type="application/json")
