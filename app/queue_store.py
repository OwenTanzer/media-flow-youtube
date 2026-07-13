"""A pending-videos list stored as `queue.json` in the Drive folder itself,
so adding a video to the archive is as simple as editing that file from
anywhere — no redeploy or API call required.

Each entry is either a plain URL/ID string (the original format), or
`{"url": ..., "languages": [...]}` when a discovery source (see
app/discovery.py) wants to override the server's default transcript
languages for that video, e.g. to match the channel it came from."""

from __future__ import annotations

import json
import logging

from . import drive
from .config import settings

logger = logging.getLogger("media_flow.queue")

QUEUE_FILENAME = "queue.json"


def entry_url(entry: str | dict) -> str:
    return entry["url"] if isinstance(entry, dict) else entry


def entry_languages(entry: str | dict, default: list[str] | None) -> list[str] | None:
    if isinstance(entry, dict):
        languages = entry.get("languages")
        if languages:
            return list(languages)
    return default


def read_queue(folder_id: str) -> list[str | dict]:
    if settings.dry_run:
        return []

    text = drive.download_text(folder_id, QUEUE_FILENAME)
    if text is None:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("queue.json in folder %s was not valid JSON; treating as empty.", folder_id)
        return []
    if not isinstance(data, list):
        return []

    entries: list[str | dict] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("url"), str):
            parsed: dict = {"url": item["url"]}
            languages = item.get("languages")
            if isinstance(languages, list) and languages:
                parsed["languages"] = [str(code) for code in languages]
            entries.append(parsed)
        else:
            entries.append(str(item))
    return entries


def write_queue(folder_id: str, entries: list[str | dict]) -> None:
    drive.upload_text_file(folder_id, QUEUE_FILENAME, json.dumps(entries, indent=2), mime_type="application/json")
