"""Minimal, independent summary worker used to clear the insight backlog."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import channel_store, drive, group_store
from .channel_store import DEFAULT_GROUP, resolve_group
from .config import settings
from .summarize import PROMPT_VERSION, SummarizationError, estimate_cost_usd, strip_frontmatter, summarize_transcript, transcript_hash
from .summary_store import SUMMARIES_FOLDER_NAME, _extract_channel
from .youtube import format_timestamp

logger = logging.getLogger("media_flow.backlog_summarizer")


@dataclass
class SummaryReport:
    eligible: int = 0
    skipped_current: int = 0
    forced: int = 0
    summarized: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _Job:
    video_id: str
    body: str
    video_types: list[str]
    descriptions: dict[str, str]
    base: dict
    summaries_folder_id: str


def _read_artifact(folder_id: str, video_id: str) -> dict | None:
    text = drive.download_text(folder_id, f"{video_id}.json")
    if text is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _has_completed_summary(existing: dict | None) -> bool:
    """A completed ("ok") summary is authoritative by default (issue #26):
    once a video has one, a normal backlog run never regenerates or
    overwrites it, even if the source transcript, model, prompt version, or
    group taxonomy has since drifted from what produced it - replacing a
    published result is expensive and risky enough that it must be a
    deliberate choice, not an automatic side effect of this job running.

    A failed/incomplete artifact (status != "ok", including no artifact at
    all) never suppresses work - those videos stay eligible every run.

    The only way to regenerate a completed summary is the explicit,
    targeted opt-in in summarize_backlog() (SUMMARY_FORCE_RESUMMARIZE_VIDEO_IDS)."""
    return bool(existing) and existing.get("status") == "ok"


def _write(folder_id: str, video_id: str, artifact: dict) -> None:
    drive.upload_text_file(folder_id, f"{video_id}.json", json.dumps(artifact, indent=2, sort_keys=True), mime_type="application/json")


def _run_one(job: _Job) -> tuple[str, str, int, int, float, str | None]:
    """Return video id, outcome, actual usage, cost, and an optional error."""
    input_tokens = output_tokens = 0
    try:
        output, usage, points_truncated = summarize_transcript(
            job.body, model=settings.summary_model, max_output_tokens=settings.summary_max_output_tokens,
            video_types=job.video_types, video_type_descriptions=job.descriptions,
        )
    except SummarizationError as exc:
        if exc.usage:
            input_tokens += exc.usage.input_tokens
            output_tokens += exc.usage.output_tokens
        _write(job.summaries_folder_id, job.video_id, {**job.base, "status": "error", "message": str(exc)})
        return job.video_id, "failed", input_tokens, output_tokens, estimate_cost_usd(settings.summary_model, input_tokens, output_tokens) or 0.0, str(exc)

    input_tokens += usage.input_tokens
    output_tokens += usage.output_tokens
    points = []
    for point in output.points:
        item = {"importance": point.importance, "main_point": point.main_point, "explanation": point.explanation}
        if point.timestamp_seconds is not None:
            item["timestamp_seconds"] = point.timestamp_seconds
            item["timestamp"] = format_timestamp(point.timestamp_seconds)
        points.append(item)
    artifact = {**job.base, "status": "ok", "video_type": output.video_type, "summary": output.summary, "points": points}
    if points_truncated:
        artifact["points_truncated"] = True
    _write(job.summaries_folder_id, job.video_id, artifact)
    return job.video_id, "structured", input_tokens, output_tokens, estimate_cost_usd(settings.summary_model, input_tokens, output_tokens) or 0.0, None


def summarize_backlog(folder_id: str) -> SummaryReport:
    """Drain videos with no completed summary yet, without a discovery lock
    or budget gate. A completed ("ok") summary is authoritative and never
    replaced by a normal run (issue #26) - see _has_completed_summary() -
    except for video IDs explicitly listed in
    SUMMARY_FORCE_RESUMMARIZE_VIDEO_IDS, a deliberate, targeted opt-in."""
    report = SummaryReport()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("ANTHROPIC_API_KEY is not set; skipping summarization.")
        return report

    index = drive.read_index(folder_id)
    groups = group_store.read_groups(folder_id)
    group_by_channel = {channel.channel_id: resolve_group(channel) for channel in channel_store.read_channels(folder_id)}
    summaries_folder = drive.get_or_create_folder(folder_id, SUMMARIES_FOLDER_NAME)
    jobs: list[_Job] = []
    for video_id, entry in index.items():
        if entry.get("status") != "ok" or not entry.get("filename"):
            continue
        markdown = drive.download_text(folder_id, entry["filename"])
        if markdown is None:
            continue
        group = group_by_channel.get(entry.get("channel_id"), DEFAULT_GROUP)
        video_types = group_store.get_video_types(groups, group, DEFAULT_GROUP)
        descriptions = group_store.get_video_type_descriptions(groups, group, DEFAULT_GROUP)
        fingerprint = group_store.video_types_fingerprint(video_types, descriptions)
        body = strip_frontmatter(markdown)
        source_hash = transcript_hash(body)
        existing = _read_artifact(summaries_folder, video_id)
        forced = video_id in settings.summary_force_resummarize_video_ids
        if _has_completed_summary(existing) and not forced:
            logger.debug("Skipping %s: a completed summary already exists.", video_id)
            report.skipped_current += 1
            continue
        if forced and _has_completed_summary(existing):
            logger.info("Force-resummarizing %s: explicitly requested via SUMMARY_FORCE_RESUMMARIZE_VIDEO_IDS.", video_id)
            report.forced += 1
        if len(body) > settings.summary_max_transcript_chars:
            body = body[: settings.summary_max_transcript_chars]
        base = {"video_id": video_id, "source_drive_file_id": entry.get("drive_file_id"),
                "source_transcript_hash": source_hash, "title": entry.get("title"), "author": _extract_channel(markdown),
                "url": entry.get("url"), "video_published_at": entry.get("published_at"), "channel_id": entry.get("channel_id"),
                "model": settings.summary_model, "prompt_version": PROMPT_VERSION, "video_types_fingerprint": fingerprint,
                "generated_at": datetime.now(timezone.utc).isoformat()}
        jobs.append(_Job(video_id, body, video_types, descriptions, base, summaries_folder))

    report.eligible = len(jobs)
    with ThreadPoolExecutor(max_workers=settings.summary_worker_concurrency) as pool:
        for video_id, outcome, ins, outs, cost, error in pool.map(_run_one, jobs):
            report.input_tokens += ins
            report.output_tokens += outs
            report.estimated_cost_usd += cost
            if outcome == "structured":
                report.summarized += 1
            else:
                report.failed += 1
                report.failures.append((video_id, error or "summary failed"))
    return report
