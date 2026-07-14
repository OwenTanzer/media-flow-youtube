import json

from app import summary_store
from app.summarize import ModelSummaryOutput, SummarizationError, SummaryPoint, Usage

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
    return written


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

    output = ModelSummaryOutput(
        subject="Subject",
        summary="Summary.",
        points=[SummaryPoint(importance="major", main_point="Point", explanation="Because.", timestamp_seconds=0)],
    )
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=10, output_tokens=5))
    )

    report = summary_store.summarize_eligible("folder-id")

    assert report.summarized == 1
    assert report.failed == 0
    assert report.eligible == 1
    assert report.retried == 0
    written_artifact = written["abc123XYZde.json"]
    assert written_artifact["status"] == "ok"
    assert written_artifact["subject"] == "Subject"
    assert written_artifact["author"] == "A Channel"
    assert written_artifact["usage"]["input_tokens"] == 10
    assert written_artifact["attempts"] == 1
    # The display timestamp is derived in application code, not trusted
    # from the model - format_timestamp(0) == "00:00".
    assert written_artifact["points"][0]["timestamp"] == "00:00"
    assert written_artifact["points"][0]["timestamp_seconds"] == 0


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
    output = ModelSummaryOutput(subject="S", summary="S.", points=[])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1))
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
    output = ModelSummaryOutput(subject="S", summary="S.", points=[])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1))
    )

    report = summary_store.summarize_eligible("folder-id")

    assert report.retried == 1
    assert report.summarized == 1
    assert written["abc123XYZde.json"]["attempts"] == 2


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

    output = ModelSummaryOutput(subject="S", summary="S.", points=[])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1))
    )

    report = summary_store.summarize_eligible("folder-id")

    assert report.summarized == 1
    assert report.stopped_on_budget is True


def test_summarize_eligible_reserves_worst_case_cost_before_calling(monkeypatch):
    """A video whose worst-case cost (max_output_tokens fully consumed)
    would push the run over SUMMARY_MAX_COST_USD_PER_RUN must not be
    started at all - checking only after-the-fact totals could overshoot
    the cap by a full call's worth of spend."""
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
    assert report.stopped_on_budget is True


def test_summarize_eligible_calls_on_progress_before_model_call_and_before_write(monkeypatch):
    _stub_drive(monkeypatch, index=_INDEX_ONE_VIDEO, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})
    output = ModelSummaryOutput(subject="S", summary="S.", points=[])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1))
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
    output = ModelSummaryOutput(subject="S", summary="S.", points=[])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1))
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
