from app import backlog_summarizer as worker
from app.summarize import ResolvedPoint, ResolvedSummary, SummarizationError, Usage


def test_only_current_success_suppresses_work():
    base = {"source_transcript_hash": "sha256:x", "model": worker.settings.summary_model,
            "prompt_version": worker.PROMPT_VERSION, "video_types_fingerprint": "types"}
    assert worker._is_current({**base, "status": "ok"}, "sha256:x", "types") is True
    assert worker._is_current({**base, "status": "error", "attempts": 99, "next_retry_at": "2099-01-01T00:00:00+00:00"}, "sha256:x", "types") is False


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
