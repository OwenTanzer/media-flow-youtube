"""A pending-videos list stored as `queue.json` in the Drive folder itself,
so adding a video to the archive is as simple as editing that file from
anywhere — no redeploy or API call required."""

from __future__ import annotations

import io
import json
import logging

from googleapiclient.http import MediaIoBaseDownload

from . import drive
from .config import settings

logger = logging.getLogger("media_flow.queue")

QUEUE_FILENAME = "queue.json"


def read_queue(folder_id: str) -> list[str]:
    if settings.dry_run:
        return []

    service = drive.get_drive_service()
    existing = drive._find_file(service, folder_id, QUEUE_FILENAME)  # noqa: SLF001
    if not existing:
        return []

    buffer = io.BytesIO()
    request = service.files().get_media(fileId=existing["id"])
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    try:
        data = json.loads(buffer.getvalue().decode("utf-8"))
    except json.JSONDecodeError:
        logger.warning("queue.json in folder %s was not valid JSON; treating as empty.", folder_id)
        return []
    return [str(item) for item in data] if isinstance(data, list) else []


def write_queue(folder_id: str, urls: list[str]) -> None:
    drive.upload_text_file(folder_id, QUEUE_FILENAME, json.dumps(urls, indent=2), mime_type="application/json")
