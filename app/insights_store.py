"""Read-only data-access layer for the Streamlit insight dashboard
(vidproc_app.py, issue #8). Assembles a single, dashboard-ready snapshot
from the same Drive-hosted sources the pipeline itself writes -
channels.json (app/channel_store.py), _index.json (app/drive.py), and each
video's summaries/<video_id>.json (app/summary_store.py) - without adding
any new Drive capability: _index.json is already the complete enumeration
of every video ever attempted, keyed by video_id, so no folder-listing is
needed.

Deliberately has no Streamlit import - stays independently unit-testable
like every other app/ module, and keeps this module usable from anything
else that might want the same read path later."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from . import channel_store, drive, summary_store
from .channel_store import Channel, DEFAULT_GROUP, resolve_group

logger = logging.getLogger("media_flow.insights_store")

# The Level-2 channel-filter label for a video whose channel_id is missing
# or doesn't match any currently configured channel (predates the
# channel_id field, or its channel was later removed from the registry -
# see backfill_channel_ids.py for the former).
UNASSIGNED_CHANNEL_LABEL = "Unassigned / Other"


@dataclass
class InsightPoint:
    importance: Literal["major", "minor"]
    main_point: str
    explanation: str
    timestamp_seconds: int | None
    timestamp: str | None


@dataclass
class VideoInsight:
    video_id: str
    title: str
    author: str | None  # display fallback only - never used for filtering, see channel_id
    url: str
    channel_id: str | None  # None => Unassigned/Other
    channel_name: str | None  # resolved display name, when channel_id matches a known channel
    group: str  # resolved: matching channel's group, or DEFAULT_GROUP as a fallback
    video_type: str | None
    video_published_at: datetime | None
    generated_at: datetime | None  # the summary artifact's own generated_at - the sort fallback
    summary: str
    points: list[InsightPoint]
    drive_file_id: str | None
    transcript_truncated: bool


@dataclass
class CostUsageSummary:
    """Aggregate Claude spend/usage across every summary artifact this
    snapshot has already fetched (app/backlog_summarizer.py persists a
    "usage" block - input_tokens, output_tokens, estimated_cost_usd - on
    both successful and failed attempts, since a failed one still burns
    real tokens). Powers the admin panel's cost/usage summary without a
    second Drive read pass."""

    total_summarized: int = 0  # artifacts with status "ok"
    total_failed: int = 0  # artifacts with status "error"
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    # Artifacts written before "usage" existed (or by the old, superseded
    # summary_store.summarize_eligible() path, which nested these fields
    # differently) - their tokens/cost simply aren't counted above, so this
    # tells the admin panel its total is a floor, not exact, when nonzero.
    videos_missing_usage_data: int = 0


@dataclass
class InsightsSnapshot:
    videos: list[VideoInsight]
    channels: list[Channel]
    pending_count: int  # status "ok" in the index with no ok summary yet (never summarized, or status: "error")
    cost_usage: CostUsageSummary = field(default_factory=CostUsageSummary)
    load_errors: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # A timezone-less artifact timestamp would otherwise crash
        # feed_sort_key()'s aware-datetime comparisons in vidproc/state.py -
        # assume UTC, same as queue_store's equivalent normalization.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_point(raw: object) -> InsightPoint | None:
    if not isinstance(raw, dict):
        return None
    importance = raw.get("importance")
    main_point = raw.get("main_point")
    explanation = raw.get("explanation")
    if importance not in ("major", "minor") or not isinstance(main_point, str) or not isinstance(explanation, str):
        return None
    timestamp_seconds = raw.get("timestamp_seconds")
    timestamp = raw.get("timestamp")
    return InsightPoint(
        importance=importance,
        main_point=main_point,
        explanation=explanation,
        timestamp_seconds=timestamp_seconds if isinstance(timestamp_seconds, int) else None,
        timestamp=timestamp if isinstance(timestamp, str) else None,
    )


def _build_video_insight(
    video_id: str, artifact: dict, channels_by_id: dict[str, Channel]
) -> VideoInsight | None:
    title = artifact.get("title")
    url = artifact.get("url")
    summary = artifact.get("summary")
    raw_points = artifact.get("points")
    if not isinstance(title, str) or not isinstance(url, str) or not isinstance(summary, str):
        return None
    if not isinstance(raw_points, list):
        return None

    points = [p for p in (_parse_point(raw) for raw in raw_points) if p is not None]

    channel_id = artifact.get("channel_id")
    channel_id = channel_id if isinstance(channel_id, str) else None
    channel = channels_by_id.get(channel_id) if channel_id else None
    group = resolve_group(channel) if channel is not None else DEFAULT_GROUP

    video_type = artifact.get("video_type")
    author = artifact.get("author")

    return VideoInsight(
        video_id=video_id,
        title=title,
        author=author if isinstance(author, str) else None,
        url=url,
        channel_id=channel_id,
        channel_name=channel.name if channel is not None else None,
        group=group,
        video_type=video_type if isinstance(video_type, str) else None,
        video_published_at=_parse_iso(artifact.get("video_published_at")),
        generated_at=_parse_iso(artifact.get("generated_at")),
        summary=summary,
        points=points,
        drive_file_id=artifact.get("source_drive_file_id")
        if isinstance(artifact.get("source_drive_file_id"), str)
        else None,
        transcript_truncated=bool(artifact.get("transcript_truncated")),
    )


def _compute_cost_usage(artifacts: dict[str, dict]) -> CostUsageSummary:
    summary = CostUsageSummary()
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            continue
        status = artifact.get("status")
        if status == "ok":
            summary.total_summarized += 1
        elif status == "error":
            summary.total_failed += 1
        else:
            continue

        usage = artifact.get("usage")
        if not isinstance(usage, dict):
            summary.videos_missing_usage_data += 1
            continue
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cost = usage.get("estimated_cost_usd")
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int) or not isinstance(cost, (int, float)):
            summary.videos_missing_usage_data += 1
            continue
        summary.total_input_tokens += input_tokens
        summary.total_output_tokens += output_tokens
        summary.total_estimated_cost_usd += cost
    return summary


def load_snapshot(folder_id: str) -> InsightsSnapshot:
    """Never raises for expected failure modes - a missing/malformed
    channels.json, a missing/failed/malformed individual summary artifact
    - those degrade into an empty channel list or a skipped video, logged
    and (when unexpected) recorded in load_errors, rather than aborting
    the whole snapshot. Only a genuinely broken Drive/credential
    configuration (a ConfigError from settings, raised by whatever calls
    this with a bad folder_id) is allowed to propagate - the caller (the
    Streamlit app) is expected to catch that and render its own
    public-facing unavailable state."""

    load_errors: list[str] = []

    try:
        channels = channel_store.read_channels(folder_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read channels.json: %s", exc)
        load_errors.append("Channel registry (channels.json) could not be read.")
        channels = []

    channels_by_id = {c.channel_id: c for c in channels}

    index = drive.read_index(folder_id)
    ok_entries = [video_id for video_id, entry in index.items() if entry.get("status") == "ok"]

    # Per-video download failures are already isolated inside
    # read_summaries_bulk() - what can still raise here is the one-time
    # folder resolution/listing, a genuinely broken-Drive-access case this
    # function deliberately lets propagate (see docstring above) rather
    # than silently discarding every artifact already fetched this call.
    artifacts = summary_store.read_summaries_bulk(folder_id, ok_entries)

    videos: list[VideoInsight] = []
    pending_count = 0

    for video_id in ok_entries:
        artifact = artifacts.get(video_id)

        if artifact is None or artifact.get("status") != "ok":
            # Not yet summarized, or a recorded failure (status: "error") -
            # both are an expected, non-broken state, not a data problem.
            pending_count += 1
            continue

        insight = _build_video_insight(video_id, artifact, channels_by_id)
        if insight is None:
            logger.warning("Summary artifact for %s is missing required fields; skipping.", video_id)
            load_errors.append(f"Summary artifact for {video_id} is malformed.")
            continue

        videos.append(insight)

    return InsightsSnapshot(
        videos=videos,
        channels=channels,
        pending_count=pending_count,
        cost_usage=_compute_cost_usage(artifacts),
        load_errors=load_errors,
    )
