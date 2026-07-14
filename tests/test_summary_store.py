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


def test_needs_summarization_true_when_prior_status_was_error():
    existing = {"status": "error", "source_transcript_hash": "sha256:abc", "model": "claude-haiku-4-5", "prompt_version": "v1"}
    assert summary_store.needs_summarization(existing, "sha256:abc", "claude-haiku-4-5", "v1") is True


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


def test_summarize_eligible_summarizes_a_newly_eligible_video(monkeypatch):
    index = {
        "abc123XYZde": {
            "status": "ok",
            "filename": "A Title [abc123XYZde].md",
            "drive_file_id": "file-id-1",
            "title": "A Title",
            "url": "https://www.youtube.com/watch?v=abc123XYZde",
        }
    }
    written = _stub_drive(monkeypatch, index=index, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})

    output = ModelSummaryOutput(
        subject="Subject",
        summary="Summary.",
        points=[SummaryPoint(importance="major", main_point="Point", explanation="Because.", timestamp_seconds=0, timestamp="00:00")],
    )
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=10, output_tokens=5))
    )

    report = summary_store.summarize_eligible("folder-id")

    assert report.summarized == 1
    assert report.failed == 0
    assert report.eligible == 1
    written_artifact = written["abc123XYZde.json"]
    assert written_artifact["status"] == "ok"
    assert written_artifact["subject"] == "Subject"
    assert written_artifact["author"] == "A Channel"
    assert written_artifact["usage"]["input_tokens"] == 10


def test_summarize_eligible_skips_already_current_summary(monkeypatch):
    body = summary_store.strip_frontmatter(TRANSCRIPT_MARKDOWN)
    current_hash = summary_store.transcript_hash(body)
    index = {
        "abc123XYZde": {
            "status": "ok",
            "filename": "A Title [abc123XYZde].md",
            "drive_file_id": "file-id-1",
            "title": "A Title",
            "url": "https://www.youtube.com/watch?v=abc123XYZde",
        }
    }
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
        index=index,
        transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN},
        existing_summaries=existing_summaries,
    )
    calls = []
    monkeypatch.setattr(summary_store, "summarize_transcript", lambda *a, **k: calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert report.skipped_current == 1
    assert report.summarized == 0
    assert calls == []


def test_summarize_eligible_ignores_non_ok_index_entries(monkeypatch):
    index = {"abc123XYZde": {"status": "blocked", "filename": "x.md"}}
    _stub_drive(monkeypatch, index=index, transcripts={})
    calls = []
    monkeypatch.setattr(summary_store, "summarize_transcript", lambda *a, **k: calls.append(1))

    report = summary_store.summarize_eligible("folder-id")

    assert report.eligible == 0
    assert calls == []


def test_summarize_eligible_isolates_a_per_video_failure(monkeypatch):
    index = {
        "abc123XYZde": {
            "status": "ok",
            "filename": "A Title [abc123XYZde].md",
            "drive_file_id": "file-id-1",
            "title": "A Title",
            "url": "https://www.youtube.com/watch?v=abc123XYZde",
        }
    }
    written = _stub_drive(monkeypatch, index=index, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})

    def _raise(*a, **k):
        raise SummarizationError("boom")

    monkeypatch.setattr(summary_store, "summarize_transcript", _raise)

    report = summary_store.summarize_eligible("folder-id")

    assert report.failed == 1
    assert report.summarized == 0
    assert report.failures == [("abc123XYZde", "boom")]
    assert written["abc123XYZde.json"]["status"] == "error"


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


def test_summarize_eligible_calls_on_progress_after_each_video(monkeypatch):
    index = {
        "abc123XYZde": {
            "status": "ok",
            "filename": "A Title [abc123XYZde].md",
            "drive_file_id": "file-id-1",
            "title": "A Title",
            "url": "https://www.youtube.com/watch?v=abc123XYZde",
        }
    }
    _stub_drive(monkeypatch, index=index, transcripts={"A Title [abc123XYZde].md": TRANSCRIPT_MARKDOWN})
    output = ModelSummaryOutput(subject="S", summary="S.", points=[])
    monkeypatch.setattr(
        summary_store, "summarize_transcript", lambda body, model, max_output_tokens: (output, Usage(input_tokens=1, output_tokens=1))
    )

    calls = []
    summary_store.summarize_eligible("folder-id", on_progress=lambda: calls.append(1))

    assert calls == [1]
