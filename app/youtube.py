"""Helpers for turning a YouTube URL into a transcript + metadata."""

from __future__ import annotations

import json
import logging
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
from youtube_transcript_api.proxies import GenericProxyConfig, ProxyConfig, WebshareProxyConfig

from .config import ConfigError, settings

logger = logging.getLogger("media_flow.youtube")

OEMBED_URL = "https://www.youtube.com/oembed"
OEMBED_ATTEMPTS = 2

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


def build_proxy_config() -> ProxyConfig | None:
    """Builds the egress proxy for outbound YouTube requests, if configured.
    Both a rotating residential provider (Webshare, which the library has
    first-class support for) and any generic HTTP/HTTPS proxy - including a
    self-hosted tunnel - are supported via YOUTUBE_PROXY_TYPE."""

    proxy_type = (settings.youtube_proxy_type or "").strip().lower()
    if not proxy_type or proxy_type == "none":
        return None

    if proxy_type == "webshare":
        if not settings.webshare_proxy_username or not settings.webshare_proxy_password:
            raise ConfigError(
                "YOUTUBE_PROXY_TYPE=webshare requires WEBSHARE_PROXY_USERNAME and "
                "WEBSHARE_PROXY_PASSWORD to be set."
            )
        return WebshareProxyConfig(
            proxy_username=settings.webshare_proxy_username,
            proxy_password=settings.webshare_proxy_password,
            filter_ip_locations=settings.webshare_proxy_locations or None,
            # The library's own retries_when_blocked (default 10) retries
            # blocked/429 responses on the *same* client instance/session.
            # fetch_transcript() is now the single retry authority, each
            # attempt building a brand-new instance for a genuinely fresh
            # connection - stacking the library's internal retries on top
            # would let TRANSCRIPT_FETCH_MAX_ATTEMPTS=3 fan out into 30+
            # requests per video before giving up.
            retries_when_blocked=0,
        )

    if proxy_type == "generic":
        if not settings.youtube_proxy_http_url and not settings.youtube_proxy_https_url:
            raise ConfigError(
                "YOUTUBE_PROXY_TYPE=generic requires YOUTUBE_PROXY_HTTP_URL and/or "
                "YOUTUBE_PROXY_HTTPS_URL to be set."
            )
        return GenericProxyConfig(
            http_url=settings.youtube_proxy_http_url,
            https_url=settings.youtube_proxy_https_url,
        )

    raise ConfigError(f"Unknown YOUTUBE_PROXY_TYPE: {settings.youtube_proxy_type!r}")


@dataclass
class VideoMetadata:
    title: str
    author: str | None


def fetch_video_metadata(video_id: str, timeout: float = 10.0) -> VideoMetadata:
    """Best-effort title/author lookup via YouTube's public oEmbed endpoint.
    Requires no API key. Falls back to the video ID if the lookup fails on
    every attempt (e.g. the video is private/deleted, or a rotating proxy
    keeps landing on a blocked exit IP)."""

    proxy_config = build_proxy_config()
    proxies = proxy_config.to_requests_dict() if proxy_config else None

    last_exc: Exception | None = None
    for attempt in range(1, OEMBED_ATTEMPTS + 1):
        try:
            response = requests.get(
                OEMBED_URL,
                params={"url": canonical_url(video_id), "format": "json"},
                timeout=timeout,
                proxies=proxies,
            )
            response.raise_for_status()
            data = response.json()
            return VideoMetadata(title=data.get("title") or video_id, author=data.get("author_name"))
        except (requests.RequestException, ValueError) as exc:
            # ValueError covers response.json() failing on a malformed/non-JSON body.
            last_exc = exc
            logger.warning("oEmbed lookup failed for %s (attempt %d/%d): %s", video_id, attempt, OEMBED_ATTEMPTS, exc)

    logger.warning("Falling back to video ID as title for %s after %d failed oEmbed attempts: %s", video_id, OEMBED_ATTEMPTS, last_exc)
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
    the caller can act on without needing to know about this library.

    A rotating residential proxy draw has a fixed chance (empirically
    ~15-20%) of landing on an exit IP YouTube has already flagged - this
    doesn't improve by waiting or spacing requests out further (confirmed
    by testing fresh draws against a known-failing video with no delay at
    all: still ~80% success), so a blocked or network-flaky attempt is
    retried a few times with a brand-new client instance - a fresh
    connection, and thus another independent draw from the pool - rather
    than giving up after a single bad draw."""

    last_status = "error"
    last_message = "Unknown failure."
    for attempt in range(1, settings.transcript_fetch_max_attempts + 1):
        try:
            transcript = YouTubeTranscriptApi(proxy_config=build_proxy_config()).fetch(video_id, languages=languages)
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
            last_status, last_message = "blocked", str(exc)
            logger.warning(
                "Blocked fetching %s (attempt %d/%d): %s",
                video_id,
                attempt,
                settings.transcript_fetch_max_attempts,
                exc,
            )
            continue
        except requests.RequestException as exc:
            # Transport-level failures (DNS, TLS, dropped/incomplete
            # connections, proxy errors) rather than an explicit
            # YouTube/library block - kept distinct from "blocked" so a
            # spike in one vs. the other is diagnosable (e.g. YouTube 429s
            # vs. proxy instability), even though both stay queued for
            # retry on the next run either way.
            last_status, last_message = "error", str(exc)
            logger.warning(
                "Transient network failure fetching %s (attempt %d/%d): %s",
                video_id,
                attempt,
                settings.transcript_fetch_max_attempts,
                exc,
            )
            continue
        except CouldNotRetrieveTranscript as exc:
            return TranscriptResult(video_id, "error", message=str(exc))
        else:
            lines = [(snippet.start, snippet.text) for snippet in transcript]
            return TranscriptResult(
                video_id,
                "ok",
                language=transcript.language,
                language_code=transcript.language_code,
                is_generated=transcript.is_generated,
                lines=lines,
            )

    return TranscriptResult(
        video_id,
        last_status,
        message=f"Still failing after {settings.transcript_fetch_max_attempts} attempts ({last_status}): {last_message}",
    )


def format_timestamp(seconds: float) -> str:
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
    # JSON string escaping is a valid subset of YAML double-quoted scalar
    # escaping, so json.dumps() gives us a safe quoted YAML value for any
    # free-form text (titles/channel names with quotes, colons, newlines, etc.)
    # without pulling in a YAML library.
    language_display = f"{language or 'unknown'} ({language_code or '?'})"
    frontmatter = [
        "---",
        f"video_id: {video_id}",
        f"title: {json.dumps(title)}",
        f"url: {url}",
        f"channel: {json.dumps(author or 'unknown')}",
        f"fetched_at: {fetched_at}",
        f"language: {json.dumps(language_display)}",
        f"auto_generated: {bool(is_generated)}",
        "---",
        "",
    ]
    body = [f"[{format_timestamp(start)}] {text}" for start, text in lines]
    return "\n".join(frontmatter) + "\n".join(body) + "\n"
