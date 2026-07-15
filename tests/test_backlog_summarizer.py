from app import backlog_summarizer as worker
from app.summarize import SummarizationError, Usage


def test_only_current_success_suppresses_work():
    base = {"source_transcript_hash": "sha256:x", "model": worker.settings.summary_model,
            "prompt_version": worker.PROMPT_VERSION, "video_types_fingerprint": "types"}
    assert worker._is_current({**base, "status": "ok"}, "sha256:x", "types") is True
    assert worker._is_current({**base, "status": "error", "attempts": 99, "next_retry_at": "2099-01-01T00:00:00+00:00"}, "sha256:x", "types") is False


def test_three_structured_attempts_then_one_fallback(monkeypatch):
    job = worker._Job("video", "[00:00] transcript", ["Type"], {}, {"video_id": "video"}, "summaries")
    structured_calls = []
    fallback_calls = []
    written = []

    def _structured(*args, **kwargs):
        structured_calls.append(1)
        raise SummarizationError("bad structured output")

    def _fallback(*args, **kwargs):
        fallback_calls.append(1)
        return "A usable loose summary.", Usage(input_tokens=3, output_tokens=2)

    monkeypatch.setattr(worker, "summarize_transcript", _structured)
    monkeypatch.setattr(worker, "summarize_fallback", _fallback)
    monkeypatch.setattr(worker, "_write", lambda *args: written.append(args[-1]))

    _, outcome, _, _, _, error = worker._run_one(job)

    assert outcome == "fallback"
    assert error is None
    assert len(structured_calls) == 3
    assert len(fallback_calls) == 1
    assert written[0]["fallback_summary"] is True
    assert written[0]["structured_attempts"] == 3
