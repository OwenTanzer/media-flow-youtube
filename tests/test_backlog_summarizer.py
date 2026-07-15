import json

from app import backlog_summarizer as worker
from app.summarize import ResolvedPoint, ResolvedSummary, SummarizationError, Usage

_SUMMARIES_FOLDER = "summaries-folder-id"


def _stub_backlog(monkeypatch, *, index, artifacts=None, markdown_by_filename=None, force_ids=()):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(worker.settings, "summary_force_resummarize_video_ids", frozenset(force_ids))
    monkeypatch.setattr(worker.drive, "read_index", lambda folder_id: index)
    monkeypatch.setattr(worker.group_store, "read_groups", lambda folder_id: [])
    monkeypatch.setattr(worker.channel_store, "read_channels", lambda folder_id: [])
    monkeypatch.setattr(worker.drive, "get_or_create_folder", lambda folder_id, name: _SUMMARIES_FOLDER)

    artifacts = artifacts or {}
    markdown_by_filename = markdown_by_filename or {}

    def _download_text(folder_id, filename):
        if folder_id == _SUMMARIES_FOLDER:
            video_id = filename[: -len(".json")]
            return json.dumps(artifacts[video_id]) if video_id in artifacts else None
        return markdown_by_filename.get(filename)

    monkeypatch.setattr(worker.drive, "download_text", _download_text)


def test_completed_summary_suppresses_work_regardless_of_drift():
    """Issue #26: a completed summary is authoritative even if the fields
    that would have mattered under the old drift-based check (source hash,
    model, prompt version, taxonomy fingerprint) have since changed."""
    stale = {
        "status": "ok",
        "source_transcript_hash": "sha256:old",
        "model": "some-other-model",
        "prompt_version": "old-version",
        "video_types_fingerprint": "old-types",
    }
    assert worker._has_completed_summary(stale) is True


def test_failed_or_missing_summary_does_not_suppress_work():
    assert worker._has_completed_summary(
        {"status": "error", "attempts": 99, "next_retry_at": "2099-01-01T00:00:00+00:00"}
    ) is False
    assert worker._has_completed_summary(None) is False


def test_summarize_backlog_skips_video_with_completed_summary(monkeypatch):
    index = {"video1": {"status": "ok", "filename": "video1.md"}}
    markdown = {"video1.md": "---\nauthor: Someone\n---\n\n[00:00] transcript"}
    artifacts = {"video1": {"status": "ok", "summary": "existing summary"}}
    _stub_backlog(monkeypatch, index=index, artifacts=artifacts, markdown_by_filename=markdown)
    called = []
    monkeypatch.setattr(worker, "summarize_transcript", lambda *a, **k: called.append(1))

    report = worker.summarize_backlog("folder-id")

    assert not called
    assert report.eligible == 0
    assert report.skipped_current == 1
    assert report.forced == 0


def test_summarize_backlog_processes_video_with_failed_summary(monkeypatch):
    index = {"video1": {"status": "ok", "filename": "video1.md"}}
    markdown = {"video1.md": "---\nauthor: Someone\n---\n\n[00:00] transcript"}
    artifacts = {"video1": {"status": "error", "message": "boom"}}
    _stub_backlog(monkeypatch, index=index, artifacts=artifacts, markdown_by_filename=markdown)
    output = ResolvedSummary(video_type="Type", summary="Summary", points=[])
    monkeypatch.setattr(worker, "summarize_transcript", lambda *a, **k: (output, Usage(input_tokens=1, output_tokens=1), False))

    report = worker.summarize_backlog("folder-id")

    assert report.eligible == 1
    assert report.skipped_current == 0
    assert report.summarized == 1


def test_summarize_backlog_processes_video_with_no_summary_yet(monkeypatch):
    index = {"video1": {"status": "ok", "filename": "video1.md"}}
    markdown = {"video1.md": "---\nauthor: Someone\n---\n\n[00:00] transcript"}
    _stub_backlog(monkeypatch, index=index, artifacts={}, markdown_by_filename=markdown)
    output = ResolvedSummary(video_type="Type", summary="Summary", points=[])
    monkeypatch.setattr(worker, "summarize_transcript", lambda *a, **k: (output, Usage(input_tokens=1, output_tokens=1), False))

    report = worker.summarize_backlog("folder-id")

    assert report.eligible == 1
    assert report.skipped_current == 0
    assert report.summarized == 1


def test_summarize_backlog_force_resummarizes_explicitly_listed_video(monkeypatch):
    """Issue #26's deliberate opt-in: a completed summary is only replaced
    when its video ID is explicitly listed via
    SUMMARY_FORCE_RESUMMARIZE_VIDEO_IDS."""
    index = {
        "video1": {"status": "ok", "filename": "video1.md"},
        "video2": {"status": "ok", "filename": "video2.md"},
    }
    markdown = {
        "video1.md": "---\nauthor: Someone\n---\n\n[00:00] transcript one",
        "video2.md": "---\nauthor: Someone\n---\n\n[00:00] transcript two",
    }
    artifacts = {
        "video1": {"status": "ok", "summary": "existing summary one"},
        "video2": {"status": "ok", "summary": "existing summary two"},
    }
    _stub_backlog(
        monkeypatch, index=index, artifacts=artifacts, markdown_by_filename=markdown, force_ids=["video1"]
    )
    output = ResolvedSummary(video_type="Type", summary="New summary", points=[])
    monkeypatch.setattr(worker, "summarize_transcript", lambda *a, **k: (output, Usage(input_tokens=1, output_tokens=1), False))

    report = worker.summarize_backlog("folder-id")

    assert report.eligible == 1
    assert report.forced == 1
    assert report.summarized == 1
    assert report.skipped_current == 1


def test_summarize_backlog_appends_one_ledger_entry_per_attempt(monkeypatch):
    """The usage ledger (app/usage_ledger.py) - not the overwritten-in-place
    summary artifact - is what the admin cost/usage summary sums, since it's
    append-only and survives retries/forced re-summarization."""
    index = {
        "video1": {"status": "ok", "filename": "video1.md"},
        "video2": {"status": "ok", "filename": "video2.md"},
    }
    markdown = {
        "video1.md": "---\nauthor: Someone\n---\n\n[00:00] transcript one",
        "video2.md": "---\nauthor: Someone\n---\n\n[00:00] transcript two",
    }
    _stub_backlog(monkeypatch, index=index, artifacts={}, markdown_by_filename=markdown)

    def _summarize(body, **kwargs):
        if "one" in body:
            return ResolvedSummary(video_type="Type", summary="ok", points=[]), Usage(input_tokens=10, output_tokens=5), False
        raise SummarizationError("boom", usage=Usage(input_tokens=8, output_tokens=0))

    monkeypatch.setattr(worker, "summarize_transcript", _summarize)
    appended = []
    monkeypatch.setattr(worker.usage_ledger, "append_entries", lambda folder_id, entries: appended.append(entries))

    worker.summarize_backlog("folder-id")

    assert len(appended) == 1
    entries = {entry["video_id"]: entry for entry in appended[0]}
    assert entries["video1"]["outcome"] == "ok"
    assert entries["video1"]["input_tokens"] == 10
    assert entries["video2"]["outcome"] == "error"
    assert entries["video2"]["input_tokens"] == 8
    assert "recorded_at" in entries["video1"]


def test_summarize_backlog_does_not_append_to_ledger_when_nothing_ran(monkeypatch):
    index = {"video1": {"status": "ok", "filename": "video1.md"}}
    artifacts = {"video1": {"status": "ok", "summary": "existing"}}
    markdown = {"video1.md": "---\nauthor: Someone\n---\n\n[00:00] transcript"}
    _stub_backlog(monkeypatch, index=index, artifacts=artifacts, markdown_by_filename=markdown)
    appended = []
    monkeypatch.setattr(worker.usage_ledger, "append_entries", lambda folder_id, entries: appended.append(entries))

    worker.summarize_backlog("folder-id")

    assert appended == [[]]


def test_one_structured_attempt_records_a_failure_without_fallback(monkeypatch):
    job = worker._Job("video", "[00:00] transcript", ["Type"], {}, {"video_id": "video"}, "summaries")
    structured_calls = []
    written = []

    def _structured(*args, **kwargs):
        structured_calls.append(1)
        raise SummarizationError("bad structured output")

    monkeypatch.setattr(worker, "summarize_transcript", _structured)
    monkeypatch.setattr(worker, "_write", lambda *args: written.append(args[-1]))

    _, outcome, _, _, _, error = worker._run_one(job)

    assert outcome == "failed"
    assert error == "bad structured output"
    assert len(structured_calls) == 1
    assert written[0]["status"] == "error"
    assert written[0]["usage"] == {"input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}


def test_failed_attempt_with_billed_usage_persists_it_on_the_artifact(monkeypatch):
    """Issue: a failed attempt (e.g. a safety refusal) can still have
    consumed real tokens - that usage must be persisted on the error
    artifact too, not just successful ones, so the admin cost/usage summary
    (app/insights_store.py's CostUsageSummary) accounts for the spend."""
    job = worker._Job("video", "[00:00] transcript", ["Type"], {}, {"video_id": "video"}, "summaries")
    written = []

    def _structured(*args, **kwargs):
        raise SummarizationError("refused", usage=Usage(input_tokens=100, output_tokens=10))

    monkeypatch.setattr(worker, "summarize_transcript", _structured)
    monkeypatch.setattr(worker, "_write", lambda *args: written.append(args[-1]))

    video_id, outcome, input_tokens, output_tokens, cost, error = worker._run_one(job)

    assert outcome == "failed"
    assert input_tokens == 100
    assert output_tokens == 10
    assert cost > 0
    assert written[0]["usage"] == {"input_tokens": 100, "output_tokens": 10, "estimated_cost_usd": cost}


def test_unanchored_point_is_persisted_without_timestamp(monkeypatch):
    job = worker._Job("video", "[00:00] transcript", ["Type"], {}, {"video_id": "video"}, "summaries")
    written = []
    output = ResolvedSummary(
        video_type="Type",
        summary="Summary",
        points=[
            ResolvedPoint("major", "Anchored", "Detail", 0),
            ResolvedPoint("minor", "Unanchored", "Detail", None),
        ],
    )
    monkeypatch.setattr(worker, "summarize_transcript", lambda *args, **kwargs: (output, Usage(input_tokens=3, output_tokens=2), False))
    monkeypatch.setattr(worker, "_write", lambda *args: written.append(args[-1]))

    _, outcome, *_ = worker._run_one(job)

    assert outcome == "structured"
    assert written[0]["points"] == [
        {"importance": "major", "main_point": "Anchored", "explanation": "Detail", "timestamp_seconds": 0, "timestamp": "00:00"},
        {"importance": "minor", "main_point": "Unanchored", "explanation": "Detail"},
    ]
