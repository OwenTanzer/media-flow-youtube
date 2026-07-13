"""Google Drive persistence: transcript files + a small JSON index for
fast lookup, all living inside a single shared folder."""

from __future__ import annotations

import io
import json
import logging
import re
import threading

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from .config import settings

logger = logging.getLogger("media_flow.drive")

SCOPES = ["https://www.googleapis.com/auth/drive"]
INDEX_FILENAME = "_index.json"

_lock = threading.RLock()
_service = None


def _sanitize_filename(name: str, max_length: int = 120) -> str:
    cleaned = re.sub(r"[^\w\s.-]", "", name, flags=re.UNICODE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:max_length] or "untitled"


def transcript_filename(video_id: str, title: str) -> str:
    return f"{_sanitize_filename(title)} [{video_id}].md"


def get_drive_service():
    global _service
    if _service is None:
        creds = service_account.Credentials.from_service_account_info(
            settings.service_account_info, scopes=SCOPES
        )
        _service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return _service


def _find_file(service, folder_id: str, filename: str) -> dict | None:
    escaped = filename.replace("'", "\\'")
    query = f"name = '{escaped}' and '{folder_id}' in parents and trashed = false"
    results = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)", pageSize=1)
        .execute()
    )
    files = results.get("files", [])
    return files[0] if files else None


def upload_text_file(
    folder_id: str, filename: str, content: str, mime_type: str = "text/markdown"
) -> str:
    """Creates the file if it doesn't exist yet, otherwise overwrites the
    existing one in place so re-fetching a video updates its transcript
    rather than duplicating it. Returns the Drive file ID."""

    if settings.dry_run:
        logger.info("[DRY_RUN] would write %r (%d bytes) to folder %s", filename, len(content), folder_id)
        return "dry-run-file-id"

    service = get_drive_service()
    media = MediaIoBaseUpload(io.BytesIO(content.encode("utf-8")), mimetype=mime_type, resumable=False)

    with _lock:
        existing = _find_file(service, folder_id, filename)
        if existing:
            updated = service.files().update(fileId=existing["id"], media_body=media).execute()
            return updated["id"]
        created = (
            service.files()
            .create(body={"name": filename, "parents": [folder_id]}, media_body=media, fields="id")
            .execute()
        )
        return created["id"]


def read_index(folder_id: str) -> dict:
    if settings.dry_run:
        return {}

    service = get_drive_service()
    existing = _find_file(service, folder_id, INDEX_FILENAME)
    if not existing:
        return {}

    buffer = io.BytesIO()
    request = service.files().get_media(fileId=existing["id"])
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    try:
        return json.loads(buffer.getvalue().decode("utf-8"))
    except json.JSONDecodeError:
        logger.warning("Index file in folder %s was not valid JSON; starting fresh.", folder_id)
        return {}


def write_index(folder_id: str, index: dict) -> None:
    payload = json.dumps(index, indent=2, sort_keys=True)
    upload_text_file(folder_id, INDEX_FILENAME, payload, mime_type="application/json")


def update_index_entry(folder_id: str, video_id: str, entry: dict) -> None:
    with _lock:
        index = read_index(folder_id)
        index[video_id] = entry
        write_index(folder_id, index)
