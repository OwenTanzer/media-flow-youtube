"""Persists and checks idempotency for per-video summary artifacts
(summaries/<video_id>.json in the Drive folder) - the output contract for
issue #7, consumed directly by the future Streamlit interface (#8). Also
hosts summarize_eligible(), the orchestration that reads _index.json,
decides what's eligible, and drives app/summarize.py's model call."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import drive
from .config import settings
from .summarize import (
    PROMPT_VERSION,
    SummarizationError,
    estimate_cost_usd,
    strip_frontmatter,
    summarize_transcript,
    transcript_hash,
)

logger = logging.getLogger("media_flow.summary_store")

SUMMARIES_FOLDER_NAME = "summaries"

_CHANNEL_RE = re.compile(r"^channel: (.+)$", re.MULTILINE)


def _summary_filename(video_id: str) -> str:
    return f"{video_id}.json"


def read_summary(folder_id: str, video_id: str) -> dict | None:
    if settings.dry_run:
        return None
    summaries_folder_id = drive.get_or_create_folder(folder_id, SUMMARIES_FOLDER_NAME)
    text = drive.download_text(summaries_folder_id, _summary_filename(video_id))
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Summary artifact for %s was not valid JSON; treating as absent.", video_id)
        return None


def write_summary(folder_id: str, video_id: str, artifact: dict) -> None:
    summaries_folder_id = drive.get_or_create_folder(folder_id, SUMMARIES_FOLDER_NAME)
    drive.upload_text_file(
        summaries_folder_id,
        _summary_filename(video_id),
        json.dumps(artifact, indent=2, sort_keys=True),
        mime_type="application/json",
    )


def needs_summarization(existing: dict | None, source_hash: str, model: str, prompt_version: str) -> bool:
    """True unless a current, successful summary already exists for this
    exact (transcript hash, model, prompt version) combination. A prior
    failure (status != "ok") stays eligible until it succeeds - that's how
    retries happen across runs, without a separate retry-tracking structure."""

    if existing is None:
        return True
    if existing.get("status") != "ok":
        return True
    return (
        existing.get("source_transcript_hash") != source_hash
        or existing.get("model") != model
        or existing.get("prompt_version") != prompt_version
    )


def _extract_channel(markdown: str) -> str | None:
    match = _CHANNEL_RE.search(markdown)
    if not match:
        return None
    raw = match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


@dataclass
class SummaryReport:
    eligible: int
    skipped_current: int
    summarized: int
    failed: int
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    failures: list[tuple[str, str]] = field(default_factory=list)
    stopped_on_budget: bool = False


def summarize_eligible(folder_id: str, on_progress: Callable[[], None] | None = None) -> SummaryReport:
    """Summarizes every status: "ok" transcript in _index.json that doesn't
    already have a current summary artifact, up to the configured per-run
    budgets. A failure summarizing one video is isolated (recorded as a
    status: "error" artifact) and never aborts the run - discovery and
    transcript archiving have already completed by the time this runs."""

    index = drive.read_index(folder_id)
    ok_entries = [(video_id, entry) for video_id, entry in index.items() if entry.get("status") == "ok"]

    eligible = 0
    skipped_current = 0
    summarized = 0
    failed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    failures: list[tuple[str, str]] = []
    stopped_on_budget = False

    for video_id, entry in ok_entries:
        if (
            summarized + failed >= settings.summary_max_videos_per_run
            or total_input_tokens + total_output_tokens >= settings.summary_max_total_tokens_per_run
            or total_cost >= settings.summary_max_cost_usd_per_run
        ):
            stopped_on_budget = True
            break

        filename = entry.get("filename")
        if not filename:
            continue
        markdown = drive.download_text(folder_id, filename)
        if markdown is None:
            logger.warning("Transcript file for %s (%r) is missing; skipping.", video_id, filename)
            continue

        body = strip_frontmatter(markdown)
        truncated = False
        if len(body) > settings.summary_max_transcript_chars:
            body = body[: settings.summary_max_transcript_chars]
            truncated = True

        source_hash = transcript_hash(body)
        existing = read_summary(folder_id, video_id)
        if not needs_summarization(existing, source_hash, settings.summary_model, PROMPT_VERSION):
            skipped_current += 1
            continue

        eligible += 1
        generated_at = datetime.now(timezone.utc).isoformat()
        base_fields = {
            "video_id": video_id,
            "source_drive_file_id": entry.get("drive_file_id"),
            "source_transcript_hash": source_hash,
            "title": entry.get("title"),
            "author": _extract_channel(markdown),
            "url": entry.get("url"),
            "model": settings.summary_model,
            "prompt_version": PROMPT_VERSION,
            "generated_at": generated_at,
        }

        try:
            model_output, usage = summarize_transcript(
                body, model=settings.summary_model, max_output_tokens=settings.summary_max_output_tokens
            )
        except SummarizationError as exc:
            failed += 1
            failures.append((video_id, str(exc)))
            write_summary(folder_id, video_id, {**base_fields, "status": "error", "message": str(exc)})
            if on_progress is not None:
                on_progress()
            continue

        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        cost = estimate_cost_usd(settings.summary_model, usage.input_tokens, usage.output_tokens)
        if cost is not None:
            total_cost += cost

        artifact = {
            **base_fields,
            "subject": model_output.subject,
            "summary": model_output.summary,
            "points": [point.model_dump() for point in model_output.points],
            "status": "ok",
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "estimated_cost_usd": cost,
            },
        }
        if truncated:
            artifact["transcript_truncated"] = True

        write_summary(folder_id, video_id, artifact)
        summarized += 1

        if on_progress is not None:
            on_progress()

    return SummaryReport(
        eligible=eligible,
        skipped_current=skipped_current,
        summarized=summarized,
        failed=failed,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_estimated_cost_usd=total_cost,
        failures=failures,
        stopped_on_budget=stopped_on_budget,
    )
