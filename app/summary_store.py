"""Persists and checks idempotency for per-video summary artifacts
(summaries/<video_id>.json in the Drive folder) - the output contract for
issue #7, consumed directly by the future Streamlit interface (#8). Also
hosts summarize_eligible(), the orchestration that reads _index.json,
decides what's eligible, and drives app/summarize.py's model call."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from . import channel_store, drive, group_store
from .channel_store import DEFAULT_GROUP, resolve_group
from .config import settings
from .summarize import (
    PROMPT_VERSION,
    SummarizationError,
    count_prompt_tokens,
    estimate_cost_usd,
    strip_frontmatter,
    summarize_fallback,
    summarize_transcript,
    transcript_hash,
)
from .youtube import format_timestamp

logger = logging.getLogger("media_flow.summary_store")

SUMMARIES_FOLDER_NAME = "summaries"

# A single process-wide pool, not one created and torn down per
# read_summaries_bulk() call - a per-call `with ThreadPoolExecutor(...)`
# shuts its threads down on exit, which would rebuild every worker's
# thread-local Drive client (and redo its OAuth refresh) on every single
# snapshot load, defeating the point of thread-local reuse in drive.py.
# Built lazily (not at import) so tests can monkeypatch
# settings.summary_bulk_read_max_workers before first use.
_bulk_read_executor: ThreadPoolExecutor | None = None
_bulk_read_executor_lock = threading.Lock()


def _get_bulk_read_executor() -> ThreadPoolExecutor:
    global _bulk_read_executor
    if _bulk_read_executor is None:
        with _bulk_read_executor_lock:
            if _bulk_read_executor is None:
                _bulk_read_executor = ThreadPoolExecutor(max_workers=settings.summary_bulk_read_max_workers)
    return _bulk_read_executor

# Prefixed onto a fallback summary's text (see summarize_eligible()'s
# last-attempt handling) so it's visually distinguishable from a normal,
# per-point-cited summary anywhere the text is displayed, not just via
# the "fallback_summary" field below.
FALLBACK_SUMMARY_SYMBOL = "⚠️"  # warning sign (U+26A0 U+FE0F)

_CHANNEL_RE = re.compile(r"^channel: (.+)$", re.MULTILINE)

_EMPTY_REPORT_KWARGS = dict(
    eligible=0,
    skipped_current=0,
    summarized=0,
    failed=0,
    retried=0,
    total_input_tokens=0,
    total_output_tokens=0,
    total_estimated_cost_usd=0.0,
)


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
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Summary artifact for %s was not valid JSON; treating as absent.", video_id)
        return None
    if not isinstance(parsed, dict):
        logger.warning("Summary artifact for %s was valid JSON but not an object; treating as absent.", video_id)
        return None
    return parsed


def read_summaries_bulk(folder_id: str, video_ids: list[str]) -> dict[str, dict]:
    """Like read_summary(), but for many videos at once - resolves the
    summaries folder and lists its contents exactly once instead of once
    per video_id, then downloads only the files that are actually present.
    Built for the dashboard's snapshot load (app/insights_store.py), which
    otherwise paid a Drive folder-lookup and a name-lookup query per video
    on top of every download, dominating load time once there are more
    than a handful of videos.

    video_ids with no summary file yet are simply absent from the
    returned dict, same as read_summary() returning None for them.

    Downloads run concurrently on a process-wide pool bounded by
    settings.summary_bulk_read_max_workers - this was the dominant cost
    once there were more than a handful of videos, since each download is
    its own Drive round trip and they don't depend on each other. Safe now
    that drive.get_drive_service() is thread-local (see drive.py) - each
    worker thread gets its own Drive service/http connection, built once
    and reused across every snapshot load, not just within one call.

    A download failure (network error, Drive 5xx, etc. - already retried
    internally by drive.download_text_by_id()) is isolated to that one
    video_id and logged, not allowed to escape and cost every
    already-downloaded artifact in this same call: a single sick artifact
    must not blank out the whole dashboard. Only the one-time folder
    resolution and listing above are allowed to raise - if those fail,
    nothing in this call could have succeeded anyway."""

    if settings.dry_run:
        return {}

    summaries_folder_id = drive.get_or_create_folder(folder_id, SUMMARIES_FOLDER_NAME)
    files_by_name = drive.list_files(summaries_folder_id)

    to_fetch = [
        (video_id, files_by_name[_summary_filename(video_id)])
        for video_id in video_ids
        if _summary_filename(video_id) in files_by_name
    ]
    if not to_fetch:
        return {}

    def _fetch(item: tuple[str, str]) -> tuple[str, dict | None]:
        video_id, file_id = item
        try:
            text = drive.download_text_by_id(file_id)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to download summary artifact for %s; skipping for this load.", video_id, exc_info=True)
            return video_id, None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Summary artifact for %s was not valid JSON; treating as absent.", video_id)
            return video_id, None
        if not isinstance(parsed, dict):
            # Valid JSON (e.g. a list, string, or number) that isn't an
            # object - _build_video_insight()/artifact.get(...) downstream
            # assumes a dict, and this must be isolated exactly like a
            # JSON-decode failure, not allowed to crash the whole snapshot.
            logger.warning("Summary artifact for %s was valid JSON but not an object; treating as absent.", video_id)
            return video_id, None
        return video_id, parsed

    artifacts: dict[str, dict] = {}
    pool = _get_bulk_read_executor()
    for video_id, artifact in pool.map(_fetch, to_fetch):
        if artifact is not None:
            artifacts[video_id] = artifact
    return artifacts


def write_summary(folder_id: str, video_id: str, artifact: dict) -> None:
    summaries_folder_id = drive.get_or_create_folder(folder_id, SUMMARIES_FOLDER_NAME)
    drive.upload_text_file(
        summaries_folder_id,
        _summary_filename(video_id),
        json.dumps(artifact, indent=2, sort_keys=True),
        mime_type="application/json",
    )


def _is_same_work_item(
    existing: dict | None, source_hash: str, model: str, prompt_version: str, taxonomy_fingerprint: str | None = None
) -> bool:
    """True if an existing artifact was produced for this exact (transcript
    hash, model, prompt version, taxonomy fingerprint) combination - i.e.
    retrying/overwriting it represents the same unit of work, not a fresh
    one. A changed hash, model, prompt version, or taxonomy fingerprint
    means prior attempts don't count against this "new" work item's retry
    budget.

    taxonomy_fingerprint (see app/group_store.py's video_types_fingerprint())
    identifies the video's group's video_types/video_type_descriptions at
    the moment of comparison - defaults to None, which skips this
    dimension of the check entirely (existing simpler callers/tests that
    don't care about it keep working unchanged); summarize_eligible()
    always passes the real per-video value."""

    if existing is None:
        return False
    if taxonomy_fingerprint is not None and existing.get("video_types_fingerprint") != taxonomy_fingerprint:
        return False
    return (
        existing.get("source_transcript_hash") == source_hash
        and existing.get("model") == model
        and existing.get("prompt_version") == prompt_version
    )


def needs_summarization(
    existing: dict | None,
    source_hash: str,
    model: str,
    prompt_version: str,
    max_attempts: int | None = None,
    now: datetime | None = None,
    taxonomy_fingerprint: str | None = None,
) -> bool:
    """True unless a current, successful summary already exists for this
    exact (transcript hash, model, prompt version, taxonomy fingerprint)
    combination, or a prior failure for that same combination has already
    exhausted its retry budget, was classified as non-retryable (e.g. a
    safety refusal - deterministic for the same input, so retrying wastes
    budget without changing the outcome), or hasn't reached its recorded
    next_retry_at yet (a retryable failure gets a backoff window, not an
    immediate retry on the very next invocation). A changed hash, model,
    prompt version, or taxonomy fingerprint always makes a video eligible
    again, resetting the attempt count, since that's a new unit of work -
    see _is_same_work_item()'s docstring for what taxonomy_fingerprint is.

    max_attempts, now, and taxonomy_fingerprint are optional only so
    existing simpler call sites keep working; summarize_eligible() always
    passes all three (now defaults to the real current time)."""

    if existing is None:
        return True
    if not _is_same_work_item(existing, source_hash, model, prompt_version, taxonomy_fingerprint):
        return True
    if existing.get("status") == "ok":
        return False
    if existing.get("retryable") is False:
        return False
    if max_attempts is not None and existing.get("attempts", 0) >= max_attempts:
        return False
    next_retry_at = existing.get("next_retry_at")
    if next_retry_at:
        try:
            due_at = datetime.fromisoformat(next_retry_at)
        except ValueError:
            due_at = None
        if due_at is not None and (now or datetime.now(timezone.utc)) < due_at:
            return False
    return True


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
    retried: int
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
    transcript archiving have already completed by the time this runs.

    Two distinct failure handling paths exist, deliberately:

    - Transient, per-video failures (rate limits, connection errors,
      unparseable/invalid output) are recorded on that video's artifact
      and never abort the run.
    - A genuine, still-broken credential problem (detected either by
      ANTHROPIC_API_KEY being entirely unset, checked once up front, or by
      count_prompt_tokens() raising an auth error mid-run) aborts the
      whole run instead. A credential problem isn't a property of any one
      video's content, so recording it as a per-video failure would
      poison that video permanently in a way that fixing the credential
      couldn't undo (nothing about the video's own hash/model/prompt
      changes when only the environment does) - and since summarization is
      an optional stage, an unconfigured deployment should skip it
      cleanly rather than fail discover_and_process.py's whole run.

    A third path handles a video that fails on its *last* allowed attempt
    (this_attempt >= settings.summary_max_attempts_per_video) with a
    retryable error: rather than let it sit permanently as status:
    "error", one extra call is made via summarize_fallback() for a plain
    prose summary with no per-line citations to get wrong - some speakers
    (meandering, conversational, non-linear delivery) make the normal
    source_timestamp/source_anchor grounding hard even when the model
    understood the content fine. A successful fallback writes status:
    "ok" with "fallback_summary": true, an empty points list, and the
    summary text prefixed with FALLBACK_SUMMARY_SYMBOL so it's visually
    distinguishable wherever it's displayed, not just via that field. If
    the fallback call itself fails, the video falls through to the normal
    status: "error" artifact exactly as before - this is a best-effort
    extra attempt, not a guarantee every video eventually gets a summary.

    on_progress is called (a) right before the model call, so a long-running
    lock lease (see discover_and_process.py) is renewed going into a
    potentially slow request, and (b) again right before every write to
    Drive - the second call is the important one: if the lock was lost to a
    concurrent run while the model call was in flight, on_progress raising
    stops this function before it writes anything under a lease it no
    longer holds, rather than writing first and only noticing the loss
    afterward."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("ANTHROPIC_API_KEY is not set; skipping the (optional) summarization stage.")
        return SummaryReport(**_EMPTY_REPORT_KWARGS)

    index = drive.read_index(folder_id)
    ok_entries = [(video_id, entry) for video_id, entry in index.items() if entry.get("status") == "ok"]

    # Resolved once per run, not per video - each video's channel_id maps
    # to its channels.json group, and each group maps to its configured
    # video_types (app/group_store.py), the classification categories
    # used to build that video's system prompt/output schema. A video
    # whose channel_id is missing or doesn't match any configured channel
    # falls back to DEFAULT_GROUP, same as the dashboard (app/insights_store.py)
    # treats an unresolvable channel_id.
    group_by_channel_id = {c.channel_id: resolve_group(c) for c in channel_store.read_channels(folder_id)}
    groups = group_store.read_groups(folder_id)

    eligible = 0
    skipped_current = 0
    summarized = 0
    failed = 0
    retried = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    failures: list[tuple[str, str]] = []
    stopped_on_budget = False

    def _count_usage(input_tokens: int, output_tokens: int) -> None:
        nonlocal total_input_tokens, total_output_tokens, total_cost
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        cost = estimate_cost_usd(settings.summary_model, input_tokens, output_tokens)
        if cost is not None:
            total_cost += cost

    def _next_retry_at(retryable: bool, generated_at_dt: datetime) -> str | None:
        if not retryable:
            return None
        return (generated_at_dt + timedelta(seconds=settings.summary_retry_backoff_seconds)).isoformat()

    for video_id, entry in ok_entries:
        filename = entry.get("filename")
        if not filename:
            continue
        markdown = drive.download_text(folder_id, filename)
        if markdown is None:
            logger.warning("Transcript file for %s (%r) is missing; skipping.", video_id, filename)
            continue

        group_name = group_by_channel_id.get(entry.get("channel_id"), DEFAULT_GROUP)
        video_types = group_store.get_video_types(groups, group_name, DEFAULT_GROUP)
        video_type_descriptions = group_store.get_video_type_descriptions(groups, group_name, DEFAULT_GROUP)
        taxonomy_fingerprint = group_store.video_types_fingerprint(video_types, video_type_descriptions)

        full_body = strip_frontmatter(markdown)
        # Hash the complete transcript, before any truncation - otherwise a
        # real change beyond SUMMARY_MAX_TRANSCRIPT_CHARS would be invisible
        # to the hash and a stale summary would look "current" forever.
        source_hash = transcript_hash(full_body)
        model_input = full_body
        truncated = False
        if len(model_input) > settings.summary_max_transcript_chars:
            model_input = model_input[: settings.summary_max_transcript_chars]
            truncated = True

        existing = read_summary(folder_id, video_id)
        if not needs_summarization(
            existing,
            source_hash,
            settings.summary_model,
            PROMPT_VERSION,
            settings.summary_max_attempts_per_video,
            taxonomy_fingerprint=taxonomy_fingerprint,
        ):
            skipped_current += 1
            continue

        prior_attempts = (
            existing.get("attempts", 0)
            if _is_same_work_item(
                existing, source_hash, settings.summary_model, PROMPT_VERSION, taxonomy_fingerprint
            )
            else 0
        )
        this_attempt = prior_attempts + 1
        if prior_attempts > 0:
            retried += 1

        if summarized + failed >= settings.summary_max_videos_per_run:
            stopped_on_budget = True
            break

        generated_at_dt = datetime.now(timezone.utc)
        base_fields = {
            "video_id": video_id,
            "source_drive_file_id": entry.get("drive_file_id"),
            "source_transcript_hash": source_hash,
            "title": entry.get("title"),
            "author": _extract_channel(markdown),
            "url": entry.get("url"),
            # Only known for RSS-discovered videos (see discovery.py) - the
            # actual YouTube publish time, not when this app happened to
            # fetch it. A future visualizer needs this to sort market/news
            # content chronologically by when it was actually said, not by
            # our own processing order.
            "video_published_at": entry.get("published_at"),
            # Only known for RSS-discovered videos, same as
            # video_published_at above - lets a future consumer (the
            # Streamlit dashboard, issue #8) join a video back to its
            # channels.json entry (and thus its group) reliably, instead
            # of matching on the free-text "author" field above.
            "channel_id": entry.get("channel_id"),
            "model": settings.summary_model,
            "prompt_version": PROMPT_VERSION,
            # Part of needs_summarization()'s idempotency check alongside
            # source_transcript_hash/model/prompt_version - lets a later
            # edit to this video's group's video_types/video_type_descriptions
            # (see app/group_store.py's video_types_fingerprint()) make
            # this video eligible for re-summarization again on its own,
            # without needing a fresh PROMPT_VERSION bump.
            "video_types_fingerprint": taxonomy_fingerprint,
            "generated_at": generated_at_dt.isoformat(),
            "attempts": this_attempt,
        }
        if truncated:
            base_fields["transcript_truncated"] = True

        def _write_failure_artifact(exc: SummarizationError) -> None:
            if on_progress is not None:
                # Re-check lock ownership immediately before writing, not
                # just after - a takeover during the model call must stop
                # this write, not merely be noticed once it's too late.
                on_progress()
            write_summary(
                folder_id,
                video_id,
                {
                    **base_fields,
                    "status": "error",
                    "retryable": exc.retryable,
                    "message": str(exc),
                    "next_retry_at": _next_retry_at(exc.retryable, generated_at_dt),
                },
            )

        # A real pre-flight token count (system prompt + output schema
        # overhead + the transcript itself), not a chars-per-token guess -
        # reserved against the per-run caps before the model call is made,
        # rather than only checking totals accumulated from prior calls.
        # Only a genuine, still-broken credential problem propagates
        # unwrapped here (see count_prompt_tokens()'s docstring); anything
        # recognized as transient comes back as a normal per-video
        # SummarizationError instead, so a blip on this endpoint doesn't
        # abort the whole run and skip every remaining video.
        try:
            input_tokens_estimate = count_prompt_tokens(
                model_input,
                model=settings.summary_model,
                video_types=video_types,
                video_type_descriptions=video_type_descriptions,
            )
        except SummarizationError as exc:
            failed += 1
            failures.append((video_id, str(exc)))
            _write_failure_artifact(exc)
            continue

        reserved_tokens = input_tokens_estimate + settings.summary_max_output_tokens
        reserved_cost = estimate_cost_usd(settings.summary_model, input_tokens_estimate, settings.summary_max_output_tokens)

        if reserved_tokens > settings.summary_max_total_tokens_per_run or (
            reserved_cost is not None and reserved_cost > settings.summary_max_cost_usd_per_run
        ):
            # This video's own worst case exceeds the *entire* configured
            # cap by itself, even from a completely fresh run - not just
            # this run's remaining headroom. Treating that the same as
            # "stopped on budget" would leave it first in line and
            # permanently block every other eligible video behind it,
            # every single run. Skip it instead (it stays eligible next
            # run, in case the caps are raised) rather than starving the
            # whole backlog indefinitely.
            logger.warning(
                "%s's estimated cost/tokens exceed the entire per-run budget by itself "
                "(~%d tokens, ~$%s) - skipping it rather than blocking the rest of the backlog. "
                "Raise SUMMARY_MAX_TOTAL_TOKENS_PER_RUN/SUMMARY_MAX_COST_USD_PER_RUN if this "
                "video should be summarized.",
                video_id,
                reserved_tokens,
                f"{reserved_cost:.4f}" if reserved_cost is not None else "unknown",
            )
            continue

        if total_input_tokens + total_output_tokens + reserved_tokens > settings.summary_max_total_tokens_per_run or (
            reserved_cost is not None and total_cost + reserved_cost > settings.summary_max_cost_usd_per_run
        ):
            stopped_on_budget = True
            break

        eligible += 1

        if on_progress is not None:
            # Renew before the (possibly slow) model call, not just after -
            # a long transcript or a slow provider response can otherwise
            # run past the lock's TTL with no renewal at all in between.
            on_progress()

        try:
            model_output, usage, points_truncated = summarize_transcript(
                model_input,
                model=settings.summary_model,
                max_output_tokens=settings.summary_max_output_tokens,
                video_types=video_types,
                video_type_descriptions=video_type_descriptions,
            )
        except SummarizationError as exc:
            failures.append((video_id, str(exc)))
            if exc.usage is not None:
                # The API still returned (and billed) a response even
                # though it's being treated as a failure - e.g. a safety
                # refusal or an unparseable structured output. Count it,
                # or the budget silently under-tracks real spend.
                _count_usage(exc.usage.input_tokens, exc.usage.output_tokens)
            elif exc.possibly_billed:
                # Usage is unknown but a response plausibly still happened
                # (e.g. the SDK's own schema validation raising before we
                # get access to the raw response). Conservatively charge
                # this call's reserved worst-case estimate rather than
                # contributing zero - otherwise repeated failures like this
                # could let real spend exceed the cap with nothing to show
                # for it in our own tracked totals.
                _count_usage(input_tokens_estimate, settings.summary_max_output_tokens)

            # Last resort: this video has now used up its normal per-point
            # citation attempts and would otherwise sit permanently as
            # status: "error" until something upstream changes. Some
            # speakers (meandering, conversational, non-linear delivery)
            # make source_timestamp/source_anchor grounding hard even when
            # the model clearly understood the content - rather than give
            # up entirely, try once for a plain prose summary instead,
            # which has no per-line citation to get wrong.
            #
            # Gated on fallback_eligible, not just retryable: retryable
            # also covers rate limits, connection errors, and 5xx - an
            # immediate second call right after one of those can't fix
            # anything and, for a rate limit specifically, only makes it
            # worse. fallback_eligible is reserved for content/structured-
            # output problems with this attempt's own response (see
            # SummarizationError's docstring), where an immediate retry
            # with a much simpler schema is actually likely to help.
            # Still only on the video's last attempt - every earlier
            # attempt gets a real shot at the full, timestamped citation
            # format first.
            if exc.fallback_eligible and this_attempt >= settings.summary_max_attempts_per_video:
                try:
                    fallback_summary, fallback_usage = summarize_fallback(
                        model_input, model=settings.summary_model, max_output_tokens=settings.summary_max_output_tokens
                    )
                except SummarizationError as fallback_exc:
                    if fallback_exc.usage is not None:
                        _count_usage(fallback_exc.usage.input_tokens, fallback_exc.usage.output_tokens)
                    elif fallback_exc.possibly_billed:
                        _count_usage(input_tokens_estimate, settings.summary_max_output_tokens)
                    failed += 1
                    _write_failure_artifact(exc)
                    continue

                _count_usage(fallback_usage.input_tokens, fallback_usage.output_tokens)
                fallback_cost = estimate_cost_usd(
                    settings.summary_model, fallback_usage.input_tokens, fallback_usage.output_tokens
                )
                if on_progress is not None:
                    on_progress()
                write_summary(
                    folder_id,
                    video_id,
                    {
                        **base_fields,
                        "video_type": None,
                        "summary": f"{FALLBACK_SUMMARY_SYMBOL} {fallback_summary}",
                        "points": [],
                        "status": "ok",
                        "fallback_summary": True,
                        "usage": {
                            "input_tokens": fallback_usage.input_tokens,
                            "output_tokens": fallback_usage.output_tokens,
                            "estimated_cost_usd": fallback_cost,
                        },
                    },
                )
                summarized += 1
                continue

            failed += 1
            _write_failure_artifact(exc)
            continue

        _count_usage(usage.input_tokens, usage.output_tokens)
        cost = estimate_cost_usd(settings.summary_model, usage.input_tokens, usage.output_tokens)

        artifact = {
            **base_fields,
            "video_type": model_output.video_type,
            "summary": model_output.summary,
            "points": [
                {
                    "importance": point.importance,
                    "main_point": point.main_point,
                    "explanation": point.explanation,
                    "timestamp_seconds": point.timestamp_seconds,
                    # Derived in application code, never trusted from the
                    # model - guarantees it can't disagree with
                    # timestamp_seconds.
                    "timestamp": format_timestamp(point.timestamp_seconds),
                }
                for point in model_output.points
            ],
            "status": "ok",
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "estimated_cost_usd": cost,
            },
        }
        if points_truncated:
            # The model returned more points than _max_points_for_duration()
            # allows for this video's length - only the most significant
            # ones (major over minor) were kept.
            artifact["points_truncated"] = True

        if on_progress is not None:
            on_progress()
        write_summary(folder_id, video_id, artifact)
        summarized += 1

    return SummaryReport(
        eligible=eligible,
        skipped_current=skipped_current,
        summarized=summarized,
        failed=failed,
        retried=retried,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_estimated_cost_usd=total_cost,
        failures=failures,
        stopped_on_budget=stopped_on_budget,
    )
