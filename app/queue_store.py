"""A pending-videos list stored as `queue.json` in the Drive folder itself,
so adding a video to the archive is as simple as editing that file from
anywhere — no redeploy or API call required.

Each entry is either a plain URL/ID string (the original format), or a
dict with a required "url" plus optional overrides:
- "languages": [...] - overrides the server's default transcript
  languages for that video, e.g. to match the channel it came from.
- "first_seen_at": an ISO timestamp set by discovery.py the first time a
  video was queued. Used to give a video (e.g. an in-progress livestream
  with no captions yet) a grace period of retries before "no_captions" is
  treated as terminal - see NO_CAPTIONS_GRACE_HOURS and app/batch.py.
- "published_at": the video's real publish timestamp, as reported by
  YouTube's own RSS feed (only known for videos discovery.py found - a
  plain URL/ID or a manually-added entry has no such source). Carried
  through to the transcript frontmatter and _index.json so downstream
  consumers (e.g. a future visualizer) can sort by when a video was
  actually published, not merely when this app happened to fetch it.

Plain-string entries (including ones added manually, without
first_seen_at/published_at) get no grace period and no publish date -
"no_captions" is terminal for them immediately, same as before this field
existed."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

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


def entry_published_at(entry: str | dict) -> str | None:
    """Returns the raw ISO string as-is (unlike entry_first_seen_at, nothing
    here needs to do date arithmetic on it - it's just carried through to
    the transcript frontmatter and index entry for downstream consumers)."""

    if not isinstance(entry, dict):
        return None
    value = entry.get("published_at")
    return value if isinstance(value, str) else None


def entry_first_seen_at(entry: str | dict) -> datetime | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("first_seen_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # queue.json is operator-editable, and a timezone-less ISO timestamp
        # (e.g. "2026-07-14T12:00:00") is a realistic manual entry. Assume
        # UTC rather than returning a naive datetime that would crash the
        # aware-vs-naive subtraction in batch.py's grace-period check.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


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
            first_seen_at = item.get("first_seen_at")
            if isinstance(first_seen_at, str):
                parsed["first_seen_at"] = first_seen_at
            published_at = item.get("published_at")
            if isinstance(published_at, str):
                parsed["published_at"] = published_at
            entries.append(parsed)
        else:
            entries.append(str(item))
    return entries


def write_queue(folder_id: str, entries: list[str | dict]) -> None:
    drive.upload_text_file(folder_id, QUEUE_FILENAME, json.dumps(entries, indent=2), mime_type="application/json")
