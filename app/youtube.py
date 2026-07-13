"""Helpers for turning a YouTube URL into a transcript + metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

OEMBED_URL = "https://www.youtube.com/oembed"

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class VideoUrlError(ValueError):
    pass


def extract_video_id(url_or_id: str) -> str:
    """Pulls an 11-character YouTube video ID out of any common URL shape,
    or passes through a bare ID if one was given directly."""

    candidate = url_or_id.strip()
    if _VIDEO_ID_RE.match(candidate):
        return candidate

    parsed = urlparse(candidate)
    host = (parsed.netloc or "").lower().removeprefix("www.").removeprefix("m.")

    if host in ("youtu.be",):
        video_id = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(video_id):
            return video_id

    if host in ("youtube.com", "youtube-nocookie.com", "music.youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id and _VIDEO_ID_RE.match(video_id):
                return video_id
        for prefix in ("/embed/", "/shorts/", "/v/", "/live/"):
            if parsed.path.startswith(prefix):
                video_id = parsed.path[len(prefix):].split("/")[0]
                if _VIDEO_ID_RE.match(video_id):
                    return video_id

    raise VideoUrlError(f"Could not extract a video ID from: {url_or_id!r}")


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


@dataclass
class VideoMetadata:
    title: str
    author: str | None


def fetch_video_metadata(video_id: str, timeout: float = 10.0) -> VideoMetadata:
    """Best-effort title/author lookup via YouTube's public oEmbed endpoint.
    Requires no API key. Falls back to the video ID if the lookup fails
    (e.g. the video is private or deleted)."""

    try:
        response = requests.get(
            OEMBED_URL,
            params={"url": canonical_url(video_id), "format": "json"},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return VideoMetadata(title=data.get("title") or video_id, author=data.get("author_name"))
    except requests.RequestException:
        return VideoMetadata(title=video_id, author=None)


@dataclass
class TranscriptResult:
    video_id: str
    status: str  # "ok" | "no_captions" | "unavailable" | "blocked" | "error"
    language: str | None = None
    language_code: str | None = None
    is_generated: bool | None = None
    lines: list[tuple[float, str]] | None = None
    message: str | None = None


def fetch_transcript(video_id: str, languages: list[str]) -> TranscriptResult:
    """Fetches a transcript, translating library exceptions into a status
    the caller can act on without needing to know about this library."""

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    except TranscriptsDisabled:
        return TranscriptResult(video_id, "no_captions", message="Captions are disabled for this video.")
    except NoTranscriptFound:
        return TranscriptResult(
            video_id,
            "no_captions",
            message=f"No transcript available in requested languages: {languages}.",
        )
    except (VideoUnavailable, InvalidVideoId) as exc:
        return TranscriptResult(video_id, "unavailable", message=str(exc))
    except AgeRestricted as exc:
        return TranscriptResult(video_id, "unavailable", message=str(exc))
    except (RequestBlocked, IpBlocked) as exc:
        return TranscriptResult(video_id, "blocked", message=str(exc))
    except CouldNotRetrieveTranscript as exc:
        return TranscriptResult(video_id, "error", message=str(exc))

    lines = [(snippet.start, snippet.text) for snippet in transcript]
    return TranscriptResult(
        video_id,
        "ok",
        language=transcript.language,
        language_code=transcript.language_code,
        is_generated=transcript.is_generated,
        lines=lines,
    )


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_transcript_markdown(
    *,
    video_id: str,
    url: str,
    title: str,
    author: str | None,
    fetched_at: str,
    language: str | None,
    language_code: str | None,
    is_generated: bool | None,
    lines: list[tuple[float, str]],
) -> str:
    frontmatter = [
        "---",
        f"video_id: {video_id}",
        f'title: "{title.replace(chr(34), chr(39))}"',
        f"url: {url}",
        f"channel: {author or 'unknown'}",
        f"fetched_at: {fetched_at}",
        f"language: {language or 'unknown'} ({language_code or '?'})",
        f"auto_generated: {bool(is_generated)}",
        "---",
        "",
    ]
    body = [f"[{_format_timestamp(start)}] {text}" for start, text in lines]
    return "\n".join(frontmatter) + "\n".join(body) + "\n"
