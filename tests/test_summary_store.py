import json
from datetime import datetime, timezone

import pytest

from app import summary_store
from app.summarize import ResolvedPoint, ResolvedSummary, SummarizationError, Usage

TRANSCRIPT_MARKDOWN = """---
video_id: abc123XYZde
title: "A Title"
url: https://www.youtube.com/watch?v=abc123XYZde
channel: "A Channel"
fetched_at: 2026-07-01T00:00:00+00:00
language: "English (en)"
auto_generated: false
---

[00:00] hello
[00:05] world
"""


def test_needs_summarization_true_when_no_existing_artifact():
    assert summary_store.needs_summarization(None, "sha256:abc", "claude-haiku-4-5", "v1") is True


def test_needs_summarization_false_when_current_and_matching():
    existing = {"status": "ok", "source_transcript_hash": "sha256:abc", "model": "claude-haiku-4-5", "prompt_version": "v1"}
    assert summary_store.needs_summarization(existing, "sha256:abc", "claude-haiku-4-5", "v1") is False


def test_needs_summarization_true_when_hash_changed():
    existing = {"status": "ok", "source_transcript_hash": "sha256:old", "model": "claude-haiku-4-5", "prompt_version": "v1"}
    assert summary_store.needs_summarization(existing, "sha256:new", "claude-haiku-4-5", "v1") is True


def test_needs_summarization_true_when_model_changed():
    existing = {"status": "ok", "source_transcript_hash": "sha256:abc", "model": "old-model", "prompt_version": "v1"}
    assert summary_store.needs_summarization(existing, "sha256:abc", "claude-haiku-4-5", "v1") is True


def test_needs_summarization_true_when_prompt_version_changed():
    existing = {"status": "ok", "source_transcript_hash": "sha256:abc", "model": "claude-haiku-4-5", "prompt_version": "v0"}
    assert summary_store.needs_summarization(existing, "sha256:abc", "claude-haiku-4-5", "v1") is True


def test_needs_summarization_true_when_prior_status_was_error_and_under_attempt_cap():
    existing = {
        "status": "error",
        "source_transcript_hash": "sha256:abc",
        "model": "claude-haiku-4-5",
        "prompt_version": "v1",
        "attempts": 1,
        "retryable": True,
    }
    assert summary_store.needs_summarization(existing, "sha256:abc", "claude-haiku-4-5", "v1", max_attempts=3) is True


def test_needs_summarization_false_when_prior_failure_exhausted_attempt_cap():
    existing = {
        "status": "error",
        "source_transcript_hash": "sha256:abc",
        "model": "claude-haiku-4-5",
        "prompt_version": "v1",
        "attempts": 3,
        "retryable": True,
    }
    assert summary_store.needs_summarization(existing, "sha256:abc", "claude-haiku-4-5", "v1", max_attempts=3) is False


def test_needs_summarization_false_when_prior_failure_was_non_retryable_even_under_cap():
    """A safety refusal or auth failure is deterministic for the same
    input - not worth retrying even on attempt 1 of a generous cap."""
    existing = {
        "status": "error",
        "source_transcript_hash": "sha256:abc",
        "model": "claude-haiku-4-5",
        "prompt_version": "v1",
        "attempts": 1,
        "retryable": False,
    }
    assert summary_store.needs_summarization(existing, "sha256:abc", "claude-haiku-4-5", "v1", max_attempts=5) is False


def test_needs_summarization_true_when_hash_changed_even_if_attempts_exhausted():
    """A changed hash/model/prompt_version is a new unit of work - prior
    attempts against the *old* work item don't count against it."""
    existing = {
        "status": "error",
        "source_transcript_hash": "sha256:old",
        "model": "claude-haiku-4-5",
        "prompt_version": "v1",
        "attempts": 3,
        "retryable": True,
    }
    assert summary_store.needs_summarization(existing, "sha256:new", "claude-haiku-4-5", "v1", max_attempts=3) is True


def test_extract_channel_from_markdown():
    assert summary_store._extract_channel(TRANSCRIPT_MARKDOWN) == "A Channel"


def test_extract_channel_returns_none_when_absent():
    assert summary_store._extract_channel("no frontmatter here") is None


def _stub_drive(monkeypatch, *, index, transcripts, existing_summaries=None):
    monkeypatch.setattr(summary_store.settings, "dry_run", False)
    # summarize_eligible() now checks this directly (not via Settings, per
    # the project's convention of never routing the key through our own
    # config) to decide whether the optional summarization stage runs at all.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(summary_store.drive, "read_index", lambda folder_id: index)
    monkeypatch.setattr(summary_store.drive, "get_or_create_folder", lambda parent, name: "summaries-folder-id")

    written = {}
    existing_summaries = dict(existing_summaries or {})

    def _download_text(folder_id, filename):
        if folder_id == "summaries-folder-id":
            video_id = filename.removesuffix(".json")
            return json.dumps(existing_summaries[video_id]) if video_id in existing_summaries else None
        return transcripts.get(filename)

    def _upload_text_file(folder_id, filename, content, **kwargs):
        written[filename] = json.loads(content)

    monkeypatch.setattr(summary_store.drive, "download_text", _download_text)
    monkeypatch.setattr(summary_store.drive, "upload_text_file", _upload_text_file)
    # A small, fixed, real-looking token count by default - individual
    # tests override this via monkeypatch when they care about the value.
    monkeypatch.setattr(summary_store, "count_prompt_tokens", lambda body, model: 100)
    return written


def test_read_summaries_bulk_lists_folder_once_and_downloads_by_id(monkeypatch):
    monkeypatch.setattr(summary_store.settings, "dry_run", False)
    folder_calls = []
    monkeypatch.setattr(
        summary_store.drive,
        "get_or_create_folder",
        lambda parent, name: folder_calls.append((parent, name)) or "summaries-folder-id",
    )
    list_calls = []
    monkeypatch.setattr(
        summary_store.drive,
        "list_files",
        lambda folder_id: list_calls.append(folder_id)
        or {"vid1.json": "file-1", "vid2.json": "file-2", "unrelated.txt": "file-3"},
    )
    download_calls = []
    artifacts_by_file_id = {
        "file-1": {"status": "ok", "video_id": "vid1"},
        "file-2": {"status": "ok", "video_id": "vid2"},
    }
    monkeypatch.setattr(
        summary_store.drive,
        "download_text_by_id",
        lambda file_id: download_calls.append(file_id) or json.dumps(artifacts_by_file_id[file_id]),
    )

    result = summary_store.read_summaries_bulk("folder-id", ["vid1", "vid2", "vid3-missing"])

    assert folder_calls == [("folder-id", summary_store.SUMMARIES_FOLDER_NAME)]
    assert list_calls == ["summaries-folder-id"]
    assert sorted(download_calls) == ["file-1", "file-2"]
    assert result == {"vid1": {"status": "ok", "video_id": "vid1"}, "vid2": {"status": "ok", "video_id": "vid2"}}


def test_read_summaries_bulk_downloads_concurrently_within_the_worker_cap(monkeypatch):
    """Regression test: downloads must actually overlap in time (not just
    aggregate correctly one-at-a-time), and never exceed
    BULK_READ_MAX_WORKERS concurrent downloads at once - the whole point
    of pooling this is to cut wall-clock time on a large archive without
    unbounded simultaneous Drive connections."""
    import threading
    import time

    monkeypatch.setattr(summary_store.settings, "dry_run", False)
    monkeypatch.setattr(summary_store, "BULK_READ_MAX_WORKERS", 3)
    monkeypatch.setattr(summary_store.drive, "get_or_create_folder", lambda parent, name: "summaries-folder-id")
    video_ids = [f"vid{i}" for i in range(10)]
    monkeypatch.setattr(
        summary_store.drive,
        "list_files",
        lambda folder_id: {f"{vid}.json": f"file-{vid}" for vid in video_ids},
    )

    lock = threading.Lock()
    concurrent_count = 0
    max_concurrent_seen = 0

    def _download_text_by_id(file_id):
        nonlocal concurrent_count, max_concurrent_seen
        with lock:
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
        time.sleep(0.05)
        with lock:
            concurrent_count -= 1
        return json.dumps({"status": "ok"})

    monkeypatch.setattr(summary_store.drive, "download_text_by_id", _download_text_by_id)

    result = summary_store.read_summaries_bulk("folder-id", video_ids)

    assert len(result) == 10
    assert max_concurrent_seen > 1, "downloads never overlapped - not actually concurrent"
    assert max_concurrent_seen <= 3, f"exceeded the configured worker cap: saw {max_concurrent_seen} at once"


def test_read_summaries_bulk_isolates_a_single_download_failure(monkeypatch):
    """Regression test: one video's download raising (network error, Drive
    5xx, etc.) must not discard artifacts already downloaded successfully
    for the other videos in the same call."""
    monkeypatch.setattr(summary_store.settings, "dry_run", False)
    monkeypatch.setattr(summary_store.drive, "get_or_create_folder", lambda parent, name: "summaries-folder-id")
    monkeypatch.setattr(
        summary_store.drive,
        "list_files",
        lambda folder_id: {"vid1.json": "file-1", "vid2.json": "file-2"},
    )

    def _download_text_by_id(file_id):
        if file_id == "file-1":
            raise ConnectionError("broken pipe")
        return json.dumps({"status": "ok", "video_id": "vid2"})

    monkeypatch.setattr(summary_store.drive, "download_text_by_id", _download_text_by_id)

    result = summary_store.read_summaries_bulk("folder-id", ["vid1", "vid2"])

    assert result == {"vid2": {"status": "ok", "video_id": "vid2"}}


def test_read_summaries_bulk_skips_invalid_json(monkeypatch):
    monkeypatch.setattr(summary_store.settings, "dry_run", False)
    monkeypatch.setattr(summary_store.drive, "get_or_create_folder", lambda parent, name: "summaries-folder-id")
    monkeypatch.setattr(summary_store.drive, "list_files", lambda folder_id: {"vid1.json": "file-1"})
    monkeypatch.setattr(summary_store.drive, "download_text_by_id", lambda file_id: "not json")

    result = summary_store.read_summaries_bulk("folder-id", ["vid1"])

    assert result == {}


def test_read_summaries_bulk_returns_empty_in_dry_run(monkeypatch):
    monkeypatch.setattr(summary_store.settings, "dry_run", True)
    result = summary_store.read_summaries_bulk("folder-id", ["vid1"])
    assert result == {}


_INDEX_ONE_VIDEO = {
    "abc123XYZde": {
        "status": "ok",
        "filename": "A Title [abc123XYZde].md",
        "drive_file_id": "file-id-1",
        "title": "A Title",
        "url": "https://www.youtube.com/watch?v=abc123XYZde",
    }
}


def test_summarize_eligible_summarizes_a_newly_eligible_video(monkeypatch):
    written = _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})

    output = ResolvedSummary(
        video_type="Analytic Overview",
        summary="Summary.",
        points=[ResolvedPoint(importance="major", main_point="Point", explanation="Because.", timestamp_seconds=0)],
    )
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=10, output_tokens=5), False)
    )

    report = summary_store.summarize_eligible("folder-id")

    assert report.summarized == 1
    assert report.failed == 0
    assert report.eligible == 1
    assert report.retried == 0
    written_artifact = written["abc123XYZde.json"]
    assert written_artifact["status"] == "ok"
    assert written_artifact["video_type"] == "Analytic Overview"
    assert written_artifact["author"] == "A Channel"
    assert written_artifact["usage"]["input_tokens"] == 10
    assert written_artifact["attempts"] == 1
    # The display timestamp is derived in application code, not trusted
    # from the model - format_timestamp(0) == "00:00".
    assert written_artifact["points"][0]["timestamp"] == "00:00"
    assert written_artifact["points"][0]["timestamp_seconds"] == 0
    # No published_at on this index entry (see _INDEX_ONE_VIDEO) - only
    # known for RSS-discovered videos.
    assert written_artifact["video_published_at"] is None
    # Same for channel_id - not on this index entry either.
    assert written_artifact["channel_id"] is None


def test_summarize_eligible_surfaces_video_published_at_from_the_index(monkeypatch):
    """Regression test: a future visualizer needs to sort market/news
    content by when it was actually published, not by our own processing
    order - this must be carried from _index.json into the artifact."""
    index_with_published_at = {
        "abc123XYZde": {**_INDEX_ONE_VIDEO["abc123XYZde"], "published_at": "2026-07-01T00:00:00+00:00"}
    }
    written = _stub_drive(
        monkeypatch, index=index_with_published_at, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN}
    )
    output = ResolvedSummary(
        video_type="Analytic Overview",
        summary="Summary.",
        points=[ResolvedPoint(importance="major", main_point="Point", explanation="Because.", timestamp_seconds=0)],
    )
    monkeypatch.setattr(
        summary_store,
        "summarize_transcript",
        lambda body, model, max_output_tokens: (output, Usage(input_tokens=10, output_tokens=5), False),
    )

    summary_store.summarize_eligible("folder-id")

    assert written["abc123XYZde.json"]["video_published_at"] == "2026-07-01T00:00:00+00:00"


def test_summarize_eligible_surfaces_channel_id_from_the_index(monkeypatch):
    """Regression test: the dashboard (issue #8) needs to join a summary
    back to its channels.json entry via a stable ID, not the free-text
    "author" field - this must be carried from _index.json into the
    artifact the same way video_published_at is."""
    index_with_channel_id = {"abc123XYZde": {**_INDEX_ONE_VIDEO["abc123XYZde"], "channel_id": "UC_a"}}
    written = _stub_drive(
        monkeypatch, index=index_with_channel_id, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN}
    )
    output = ResolvedSummary(
        video_type="Analytic Overview",
        summary="Summary.",
        points=[ResolvedPoint(importance="major", main_point="Point", explanation="Because.", timestamp_seconds=0)],
    )
    monkeypatch.setattr(
        summary_store,
        "summarize_transcript",
        lambda body, model, max_output_tokens: (output, Usage(input_tokens=10, output_tokens=5), False),
    )

    summary_store.summarize_eligible("folder-id")

    assert written["abc123XYZde.json"]["channel_id"] == "UC_a"


def test_summarize_eligible_skips_already_current_summary(monkeypatch):
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    existing_summaries = {
        "abc123XYZde": {
            "status": "ok",
            "source_transcript_hash": current_hash,
            "model": "claude-haiku-4-5",
            "prompt_version": summary_store.PROMPT_VERSION,
        }
    }
    _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )
    calls = []
    monkeypatch.setattr(summary_store, "summarize_transcript", lambda *a, **k: calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert report.skipped_current == 1
    assert report.summarized == 0
    assert calls == []


def test_summarize_eligible_hash_reflects_content_beyond_the_truncation_cutoff(monkeypatch):
    """Regression test: the idempotency hash must cover the complete
    transcript, not just the (possibly truncated) portion sent to the
    model - otherwise a real change past SUMMARY_MAX_TRANSCRIPT_CHARS is
    invisible and a stale summary looks "current" forever."""
    monkeypatch.setattr(summary_store.settings, "summary_max_transcript_chars", 20)

    long_markdown = TRANSCRIPT_MARKDOWN + "[00:10] and even more content past the truncation cutoff\n"
    full_body = summary_store.strip_frontmatter(long_markdown)
    full_hash = summary_store.transcript_hash(full_body)
    truncated_body = full_body[:20]
    truncated_hash = summary_store.transcript_hash(truncated_body)
    assert full_hash != truncated_hash  # sanity check the test setup is meaningful

    existing_summaries = {
        "abc123XYZde": {
            # An artifact keyed on the truncated hash would look current if
            # summarize_eligible incorrectly hashed post-truncation content.
            "status": "ok",
            "source_transcript_hash": truncated_hash,
            "model": "claude-haiku-4-5",
            "prompt_version": summary_store.PROMPT_VERSION,
        }
    }
    _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": long_markdown},
        existing_summaries=existing_summaries,
    )
    output = ResolvedSummary(video_type="Analytic Overview", summary="S.", points=[ResolvedPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=0)])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1), False)
    )

    report = summary_store.summarize_eligible("folder-id")

    # Hashing the full body means this is correctly seen as changed content,
    # not skipped as "already current".
    assert report.summarized == 1
    assert report.skipped_current == 0


def test_summarize_eligible_ignores_non_ok_index_entries(monkeypatch):
    index = {"abc123XYZde": {"status": "blocked", "filename": "x.md"}}
    _stub_drive(monkeypatch, index=index, transcripts={})
    calls = []
    monkeypatch.setattr(summary_store, "summarize_transcript", lambda *a, **k: calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert report.eligible == 0
    assert calls == []


def test_summarize_eligible_isolates_a_per_video_failure(monkeypatch):
    written = _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})

    def _raise(*a, **k):
        raise SummarizationError("boom", retryable=True)

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)

    report = summary_store.summarize_eligible("folder-id")

    assert report.failed == 1
    assert report.summarized == 0
    assert report.failures == [("abc123XYZde", "boom")]
    assert written["abc123XYZde.json"]["status"] == "error"
    assert written["abc123XYZde.json"]["retryable"] is True
    assert written["abc123XYZde.json"]["attempts"] == 1


def test_summarize_eligible_counts_usage_from_a_billed_failure(monkeypatch):
    """A safety refusal or unparseable output still consumes tokens even
    though it's treated as a failure - the budget must count it, or spend
    tracking silently under-counts real usage."""
    _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})

    def _raise(*a, **k):
        raise SummarizationError("refused", retryable=False, usage=Usage(input_tokens=200, output_tokens=10))

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)

    report = summary_store.summarize_eligible("folder-id")

    assert report.failed == 1
    assert report.total_input_tokens == 200
    assert report.total_output_tokens == 10
    assert report.total_estimated_cost_usd > 0


def test_summarize_eligible_does_not_retry_a_prior_non_retryable_failure(monkeypatch):
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    existing_summaries = {
        "abc123XYZde": {
            "status": "error",
            "source_transcript_hash": current_hash,
            "model": "claude-haiku-4-5",
            "prompt_version": summary_store.PROMPT_VERSION,
            "attempts": 1,
            "retryable": False,
        }
    }
    _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )
    calls = []
    monkeypatch.setattr(summary_store, "summarize_transcript", lambda *a, **k: calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert calls == []
    assert report.eligible == 0


def test_summarize_eligible_stops_retrying_after_max_attempts_per_video(monkeypatch):
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    monkeypatch.setattr(summary_store.settings, "summary_max_attempts_per_video", 2)
    existing_summaries = {
        "abc123XYZde": {
            "status": "error",
            "source_transcript_hash": current_hash,
            "model": "claude-haiku-4-5",
            "prompt_version": summary_store.PROMPT_VERSION,
            "attempts": 2,
            "retryable": True,
        }
    }
    _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )
    calls = []
    monkeypatch.setattr(summary_store, "summarize_transcript", lambda *a, **k: calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert calls == []
    assert report.eligible == 0


def test_summarize_eligible_counts_a_retry_and_increments_attempts(monkeypatch):
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    existing_summaries = {
        "abc123XYZde": {
            "status": "error",
            "source_transcript_hash": current_hash,
            "model": "claude-haiku-4-5",
            "prompt_version": summary_store.PROMPT_VERSION,
            "attempts": 1,
            "retryable": True,
        }
    }
    written = _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )
    output = ResolvedSummary(video_type="Analytic Overview", summary="S.", points=[ResolvedPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=0)])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1), False)
    )

    report = summary_store.summarize_eligible("folder-id")

    assert report.retried == 1
    assert report.summarized == 1
    assert written["abc123XYZde.json"]["attempts"] == 2


def _existing_error(current_hash, *, attempts, retryable):
    return {
        "status": "error",
        "source_transcript_hash": current_hash,
        "model": "claude-haiku-4-5",
        "prompt_version": summary_store.PROMPT_VERSION,
        "attempts": attempts,
        "retryable": retryable,
    }


def test_summarize_eligible_uses_a_fallback_summary_on_the_last_retryable_attempt(monkeypatch):
    """A video that fails on its last allowed attempt gets one extra,
    simpler call for a plain prose summary instead of being left
    permanently as status: 'error' - see summarize_fallback()."""
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    monkeypatch.setattr(summary_store.settings, "summary_max_attempts_per_video", 2)
    existing_summaries = {"abc123XYZde": _existing_error(current_hash, attempts=1, retryable=True)}
    written = _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )

    def _raise(*a, **k):
        raise SummarizationError("grounding failed", retryable=True, fallback_eligible=True)

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)
    monkeypatch.setattr(
        summary_store, "summarize_fallback", lambda body, model, max_output_tokens: ("A plain summary.", Usage(input_tokens=50, output_tokens=20))
    )

    report = summary_store.summarize_eligible("folder-id")

    assert report.summarized == 1
    assert report.failed == 0
    written_artifact = written["abc123XYZde.json"]
    assert written_artifact["status"] == "ok"
    assert written_artifact["fallback_summary"] is True
    assert written_artifact["summary"] == f"{summary_store.FALLBACK_SUMMARY_SYMBOL} A plain summary."
    assert written_artifact["points"] == []
    assert written_artifact["video_type"] is None
    assert written_artifact["usage"]["input_tokens"] == 50
    assert report.total_input_tokens == 50
    assert report.total_output_tokens == 20


def test_summarize_eligible_does_not_use_fallback_before_the_last_attempt(monkeypatch):
    """Every earlier attempt still gets a real shot at the normal,
    per-point-cited format first - fallback is last-resort only. Uses a
    fallback_eligible failure so this test isolates the "not the last
    attempt yet" gate specifically, not the eligibility gate."""
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    monkeypatch.setattr(summary_store.settings, "summary_max_attempts_per_video", 3)
    existing_summaries = {"abc123XYZde": _existing_error(current_hash, attempts=1, retryable=True)}
    written = _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )

    def _raise(*a, **k):
        raise SummarizationError("grounding failed", retryable=True, fallback_eligible=True)

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)
    fallback_calls = []
    monkeypatch.setattr(summary_store, "summarize_fallback", lambda *a, **k: fallback_calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert fallback_calls == []
    assert report.failed == 1
    assert written["abc123XYZde.json"]["status"] == "error"


def test_summarize_eligible_does_not_use_fallback_for_a_non_retryable_failure(monkeypatch):
    """A non-retryable failure (e.g. a safety refusal) is deterministic for
    the same input - the simpler fallback prompt would very likely be
    refused too, so it's not worth the extra call."""
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    monkeypatch.setattr(summary_store.settings, "summary_max_attempts_per_video", 2)
    existing_summaries = {"abc123XYZde": _existing_error(current_hash, attempts=1, retryable=True)}
    written = _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )

    def _raise(*a, **k):
        raise SummarizationError("refused", retryable=False)

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)
    fallback_calls = []
    monkeypatch.setattr(summary_store, "summarize_fallback", lambda *a, **k: fallback_calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert fallback_calls == []
    assert report.failed == 1
    assert written["abc123XYZde.json"]["status"] == "error"


def test_summarize_eligible_does_not_use_fallback_for_a_retryable_but_ineligible_failure(monkeypatch):
    """Regression test: retryable=True also covers provider-level outages
    (rate limits, connection errors, 5xx) via SummarizationError, not just
    content/grounding problems. An immediate fallback call right after one
    of those can't fix anything and, for a rate limit specifically, only
    makes it worse - so the gate must check fallback_eligible, not just
    retryable, even on the video's last attempt."""
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    monkeypatch.setattr(summary_store.settings, "summary_max_attempts_per_video", 2)
    existing_summaries = {"abc123XYZde": _existing_error(current_hash, attempts=1, retryable=True)}
    written = _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )

    def _raise(*a, **k):
        # retryable=True (a later scheduled run may well succeed), but
        # fallback_eligible defaults to False - this is what a real rate
        # limit/connection/5xx failure looks like.
        raise SummarizationError("Rate limited: boom", retryable=True)

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)
    fallback_calls = []
    monkeypatch.setattr(summary_store, "summarize_fallback", lambda *a, **k: fallback_calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert fallback_calls == []
    assert report.failed == 1
    assert written["abc123XYZde.json"]["status"] == "error"
    assert written["abc123XYZde.json"]["retryable"] is True


def test_summarize_eligible_falls_through_to_the_original_error_when_fallback_also_fails(monkeypatch):
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    monkeypatch.setattr(summary_store.settings, "summary_max_attempts_per_video", 2)
    existing_summaries = {"abc123XYZde": _existing_error(current_hash, attempts=1, retryable=True)}
    written = _stub_drive(
        monkeypatch,
        index=_INDEX_ONE_VIDEO,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )

    def _raise_primary(*a, **k):
        raise SummarizationError("grounding failed", retryable=True, fallback_eligible=True)

    def _raise_fallback(*a, **k):
        raise SummarizationError("rate limited", retryable=True)

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise_primary)
    monkeypatch.setattr(summary_store, "summarize_fallback", _raise_fallback)

    report = summary_store.summarize_eligible("folder-id")

    assert report.failed == 1
    assert report.summarized == 0
    written_artifact = written["abc123XYZde.json"]
    assert written_artifact["status"] == "error"
    # The artifact preserves the *original* (primary-call) error message,
    # not the fallback attempt's - the fallback is an implementation
    # detail, not what actually explains why this video has no summary.
    assert written_artifact["message"] == "grounding failed"


def test_summarize_eligible_stops_at_max_videos_per_run(monkeypatch):
    index = {
        f"vid{i}": {
            "status": "ok",
            "filename": f"vid{i}.md",
            "drive_file_id": f"file-{i}",
            "title": f"Title {i}",
            "url": f"https://www.youtube.com/watch?v=vid{i}",
        }
        for i in range(3)
    }
    _stub_drive(monkeypatch, index=index, transcripts={f"vid{i}.md": TRANSCRIPT_MARKDOWN for i in range(3)})
    monkeypatch.setattr(summary_store.settings, "summary_max_videos_per_run", 1)

    output = ResolvedSummary(video_type="Analytic Overview", summary="S.", points=[ResolvedPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=0)])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1), False)
    )

    report = summary_store.summarize_eligible("folder-id")

    assert report.summarized == 1
    assert report.stopped_on_budget is True


def test_summarize_eligible_skips_a_video_whose_own_worst_case_exceeds_the_entire_cap(monkeypatch):
    """A video whose worst-case cost (max_output_tokens fully consumed)
    exceeds the *entire* per-run cap, even from a fresh run, must be
    skipped rather than treated as "stopped on budget" - the latter would
    leave it first in line and permanently block every other eligible
    video behind it, every single run."""
    index = {
        f"vid{i}": {
            "status": "ok",
            "filename": f"vid{i}.md",
            "drive_file_id": f"file-{i}",
            "title": f"Title {i}",
            "url": f"https://www.youtube.com/watch?v=vid{i}",
        }
        for i in range(2)
    }
    _stub_drive(monkeypatch, index=index, transcripts={f"vid{i}.md": TRANSCRIPT_MARKDOWN for i in range(2)})
    # claude-haiku-4-5 output pricing is $5/MTok; a 1,000,000-token ceiling
    # alone reserves $5, comfortably over a tiny cap.
    monkeypatch.setattr(summary_store.settings, "summary_max_output_tokens", 1_000_000)
    monkeypatch.setattr(summary_store.settings, "summary_max_cost_usd_per_run", 0.01)

    calls = []
    monkeypatch.setattr(summary_store, "summarize_transcript", lambda *a, **k: calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert calls == []
    assert report.eligible == 0
    assert report.stopped_on_budget is False


def test_summarize_eligible_stops_on_budget_when_remaining_headroom_runs_out(monkeypatch):
    """Distinct from the "exceeds the entire cap" case above: here each
    individual video's own worst case comfortably fits under the full cap,
    but the *cumulative* total from an already-processed video doesn't
    leave enough headroom for another - this legitimately should stop the
    run (leaving the rest for next time), not skip-and-continue."""
    index = {
        f"vid{i}": {
            "status": "ok",
            "filename": f"vid{i}.md",
            "drive_file_id": f"file-{i}",
            "title": f"Title {i}",
            "url": f"https://www.youtube.com/watch?v=vid{i}",
        }
        for i in range(2)
    }
    _stub_drive(monkeypatch, index=index, transcripts={f"vid{i}.md": TRANSCRIPT_MARKDOWN for i in range(2)})
    monkeypatch.setattr(summary_store, "count_prompt_tokens", lambda body, model: 100)
    monkeypatch.setattr(summary_store.settings, "summary_max_output_tokens", 100)
    # One call's reserved cost (100 input + 100 output tokens) is ~$0.0006 -
    # comfortably under this cap on its own, but two calls' worth isn't.
    monkeypatch.setattr(summary_store.settings, "summary_max_cost_usd_per_run", 0.0009)

    output = ResolvedSummary(video_type="Analytic Overview", summary="S.", points=[ResolvedPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=0)])
    monkeypatch.setattr(
        summary_store,
        "summarize_transcript",
        lambda body, model, max_output_tokens: (output, Usage(input_tokens=100, output_tokens=100), False),
    )

    report = summary_store.summarize_eligible("folder-id")

    assert report.summarized == 1
    assert report.stopped_on_budget is True


def test_summarize_eligible_aborts_without_writing_when_token_counting_fails(monkeypatch):
    """Regression test: count_prompt_tokens() (used to reserve budget)
    runs before a video's attempt count is touched or anything is written
    for it. A failure there - most importantly an auth/credential problem
    - must abort the whole run rather than being recorded as a permanent,
    per-video failure that fixing the credential wouldn't undo."""
    written = _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})

    def _raise(*a, **k):
        raise RuntimeError("credential check failed")

    monkeypatch.setattr(summary_store, "count_prompt_tokens", _raise)
    calls = []
    monkeypatch.setattr(summary_store, "summarize_transcript", lambda *a, **k: calls.append(1))

    with pytest.raises(RuntimeError, match="credential check failed"):
        summary_store.summarize_eligible("folder-id")

    assert calls == []
    assert written == {}


def test_summarize_eligible_calls_on_progress_before_model_call_and_before_write(monkeypatch):
    _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})
    output = ResolvedSummary(video_type="Analytic Overview", summary="S.", points=[ResolvedPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=0)])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1), False)
    )

    calls = []
    summary_store.summarize_eligible("folder-id", on_progress=lambda: calls.append(1))

    # Once before the (possibly slow) model call, once again immediately
    # before the write - not just once after everything completes.
    assert calls == [1, 1]


def test_summarize_eligible_does_not_write_if_lock_is_lost_before_the_write(monkeypatch):
    """Regression test for the review finding: the model call ran without
    any lock renewal, and the summary was written before on_progress()
    verified ownership - a stale worker could write after another run
    acquired the lock. on_progress() raising (simulating a lost lock) must
    stop this function *before* it writes, not merely be observed after."""
    written = _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})
    output = ResolvedSummary(video_type="Analytic Overview", summary="S.", points=[ResolvedPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=0)])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1), False)
    )

    calls = []

    def _on_progress():
        calls.append(1)
        if len(calls) == 2:
            # Mirrors discover_and_process.py's _renew_lock() raising when
            # job_lock.renew_lock() reports the lease no longer belongs to
            # this run.
            raise RuntimeError("Lost the discovery lock mid-run; aborting to avoid racing a new owner.")

    try:
        summary_store.summarize_eligible("folder-id", on_progress=_on_progress)
        raised = False
    except RuntimeError:
        raised = True

    assert raised is True
    assert len(calls) == 2
    assert "abc123XYZde.json" not in written


def test_summarize_eligible_skips_the_stage_when_no_api_key_is_configured(monkeypatch):
    """Regression test: summarization is documented as optional. A
    deployment without ANTHROPIC_API_KEY must skip it cleanly instead of
    crashing discover_and_process.py's whole run every scheduled invocation."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    read_index_calls = []
    monkeypatch.setattr(summary_store.drive, "read_index", lambda folder_id: read_index_calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert report.eligible == 0
    assert report.summarized == 0
    assert read_index_calls == []  # never even looked at the index


def test_summarize_eligible_treats_a_transient_token_count_failure_as_a_per_video_failure(monkeypatch):
    """Regression test: count_prompt_tokens() failures that are recognized
    as transient (rate limit, connection error, 5xx) must not abort the
    whole run and skip every remaining video with no durable retry state -
    only a genuine credential problem should do that."""
    written = _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})

    def _raise(*a, **k):
        raise SummarizationError("token count rate limited", retryable=True)

    monkeypatch.setattr(summary_store, "count_prompt_tokens", _raise)
    calls = []
    monkeypatch.setattr(summary_store, "summarize_transcript", lambda *a, **k: calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert report.failed == 1
    assert calls == []
    assert written["abc123XYZde.json"]["status"] == "error"
    assert written["abc123XYZde.json"]["retryable"] is True
    assert written["abc123XYZde.json"]["next_retry_at"] is not None


def test_summarize_eligible_conservatively_charges_reserved_estimate_for_possibly_billed_failures(monkeypatch):
    """Regression test: a schema-validation failure (real usage
    unavailable, but a response plausibly happened and was billed) must
    not silently contribute zero to the run's tracked spend - repeated
    failures like this could otherwise let real spend exceed the
    configured cap with nothing to show for it in our own totals."""
    _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})
    monkeypatch.setattr(summary_store, "count_prompt_tokens", lambda body, model: 500)
    monkeypatch.setattr(summary_store.settings, "summary_max_output_tokens", 200)

    def _raise(*a, **k):
        raise SummarizationError("schema validation failed", retryable=True, possibly_billed=True)

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)

    report = summary_store.summarize_eligible("folder-id")

    assert report.failed == 1
    # Charged the reserved estimate (500 input + 200 output), not zero.
    assert report.total_input_tokens == 500
    assert report.total_output_tokens == 200
    assert report.total_estimated_cost_usd > 0


def test_needs_summarization_false_before_the_backoff_window_elapses():
    existing = {
        "status": "error",
        "source_transcript_hash": "sha256:abc",
        "model": "claude-haiku-4-5",
        "prompt_version": "v1",
        "attempts": 1,
        "retryable": True,
        "next_retry_at": "2026-07-14T13:00:00+00:00",
    }
    now = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)  # before next_retry_at
    assert (
        summary_store.needs_summarization(existing, "sha256:abc", "claude-haiku-4-5", "v1", max_attempts=3, now=now)
        is False
    )


def test_needs_summarization_true_after_the_backoff_window_elapses():
    existing = {
        "status": "error",
        "source_transcript_hash": "sha256:abc",
        "model": "claude-haiku-4-5",
        "prompt_version": "v1",
        "attempts": 1,
        "retryable": True,
        "next_retry_at": "2026-07-14T13:00:00+00:00",
    }
    now = datetime(2026, 7, 14, 14, 0, 0, tzinfo=timezone.utc)  # after next_retry_at
    assert (
        summary_store.needs_summarization(existing, "sha256:abc", "claude-haiku-4-5", "v1", max_attempts=3, now=now)
        is True
    )


def test_summarize_eligible_records_a_next_retry_at_for_retryable_failures(monkeypatch):
    written = _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})
    monkeypatch.setattr(summary_store.settings, "summary_retry_backoff_seconds", 900)

    def _raise(*a, **k):
        raise SummarizationError("boom", retryable=True)

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)

    summary_store.summarize_eligible("folder-id")

    artifact = written["abc123XYZde.json"]
    generated_at = datetime.fromisoformat(artifact["generated_at"])
    next_retry_at = datetime.fromisoformat(artifact["next_retry_at"])
    assert (next_retry_at - generated_at).total_seconds() == pytest.approx(900)


def test_summarize_eligible_does_not_record_a_next_retry_at_for_non_retryable_failures(monkeypatch):
    written = _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})

    def _raise(*a, **k):
        raise SummarizationError("refused", retryable=False)

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)

    summary_store.summarize_eligible("folder-id")

    assert written["abc123XYZde.json"]["next_retry_at"] is None


def test_summarize_eligible_records_points_truncated_flag(monkeypatch):
    """When summarize_transcript() reports it had to drop points to stay
    under the length-based cap, the artifact should say so."""
    written = _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})
    output = ResolvedSummary(
        video_type="Analytic Overview", summary="S.", points=[ResolvedPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=0)]
    )
    monkeypatch.setattr(
        summary_store,
        "summarize_transcript",
        lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1), True),
    )

    summary_store.summarize_eligible("folder-id")

    assert written["abc123XYZde.json"]["points_truncated"] is True


def test_summarize_eligible_omits_points_truncated_flag_when_not_truncated(monkeypatch):
    written = _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})
    output = ResolvedSummary(
        video_type="Analytic Overview", summary="S.", points=[ResolvedPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=0)]
    )
    monkeypatch.setattr(
        summary_store,
        "summarize_transcript",
        lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1), False),
    )

    summary_store.summarize_eligible("folder-id")

    assert "points_truncated" not in written["abc123XYZde.json"]
