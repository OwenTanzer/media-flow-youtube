"""The single code path used by both the ad-hoc endpoint and the batch
job: resolve a URL, fetch the transcript, write it to Drive, update the index."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import drive, youtube
from .config import settings
from .models import VideoResult

logger = logging.getLogger("media_flow.pipeline")


def process_video(url_or_id: str, languages: list[str] | None = None) -> VideoResult:
    languages = languages or settings.languages

    try:
        video_id = youtube.extract_video_id(url_or_id)
    except youtube.VideoUrlError as exc:
        return VideoResult(video_id="", url=url_or_id, status="invalid_url", message=str(exc))

    url = youtube.canonical_url(video_id)
    transcript = youtube.fetch_transcript(video_id, languages)
    metadata = youtube.fetch_video_metadata(video_id)
    fetched_at = datetime.now(timezone.utc).isoformat()

    if transcript.status != "ok":
        logger.info("Skipping %s (%s): %s", video_id, transcript.status, transcript.message)
        result = VideoResult(
            video_id=video_id,
            url=url,
            status=transcript.status,
            title=metadata.title,
            message=transcript.message,
        )
        _record_index_entry(video_id, result, fetched_at)
        return result

    folder_id = settings.require_drive_folder_id()
    filename = drive.transcript_filename(video_id, metadata.title)
    content = youtube.render_transcript_markdown(
        video_id=video_id,
        url=url,
        title=metadata.title,
        author=metadata.author,
        fetched_at=fetched_at,
        language=transcript.language,
        language_code=transcript.language_code,
        is_generated=transcript.is_generated,
        lines=transcript.lines or [],
    )
    file_id = drive.upload_text_file(folder_id, filename, content)

    result = VideoResult(
        video_id=video_id,
        url=url,
        status="ok",
        title=metadata.title,
        filename=filename,
        drive_file_id=file_id,
    )
    _record_index_entry(video_id, result, fetched_at)
    return result


def _record_index_entry(video_id: str, result: VideoResult, fetched_at: str) -> None:
    try:
        folder_id = settings.require_drive_folder_id()
    except Exception:  # noqa: BLE001
        return
    drive.update_index_entry(
        folder_id,
        video_id,
        {
            "video_id": video_id,
            "url": result.url,
            "title": result.title,
            "status": result.status,
            "filename": result.filename,
            "drive_file_id": result.drive_file_id,
            "message": result.message,
            "fetched_at": fetched_at,
        },
    )
