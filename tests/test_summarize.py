import anthropic
import httpx
import pytest

from app import summarize


def _fake_httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _fake_httpx_response() -> httpx.Response:
    return httpx.Response(429, request=_fake_httpx_request())

SAMPLE_MARKDOWN = """---
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


def test_strip_frontmatter_removes_only_the_leading_block():
    body = summarize.strip_frontmatter(SAMPLE_MARKDOWN)
    assert body == "[00:00] hello\n[00:05] world\n"


def test_strip_frontmatter_does_not_touch_a_bare_dashes_line_in_the_body():
    markdown = SAMPLE_MARKDOWN + "\n---\nnot frontmatter, just a line of dashes\n"
    body = summarize.strip_frontmatter(markdown)
    assert "not frontmatter, just a line of dashes" in body


def test_transcript_hash_is_deterministic_and_content_sensitive():
    a = summarize.transcript_hash("hello world")
    b = summarize.transcript_hash("hello world")
    c = summarize.transcript_hash("hello there")
    assert a == b
    assert a != c
    assert a.startswith("sha256:")


def test_estimate_cost_usd_known_model():
    cost = summarize.estimate_cost_usd("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(1.00 + 5.00)


def test_estimate_cost_usd_unknown_model_returns_none():
    assert summarize.estimate_cost_usd("some-future-model", 1000, 1000) is None


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeParsedMessage:
    def __init__(self, parsed_output, stop_reason="end_turn", usage=None):
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason
        self.usage = usage or _FakeUsage(100, 50)


def test_summarize_transcript_success(monkeypatch):
    expected_output = summarize.ModelSummaryOutput(
        subject="A subject",
        summary="A summary.",
        points=[
            summarize.SummaryPoint(
                importance="major", main_point="Point one", explanation="Because X.", timestamp_seconds=5, timestamp="00:05"
            )
        ],
    )

    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage(expected_output, usage=_FakeUsage(123, 45))

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _FakeClient)

    output, usage = summarize.summarize_transcript("transcript body", model="claude-haiku-4-5", max_output_tokens=1024)

    assert output == expected_output
    assert usage.input_tokens == 123
    assert usage.output_tokens == 45


def test_summarize_transcript_raises_on_refusal(monkeypatch):
    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage(None, stop_reason="refusal")

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _FakeClient)

    with pytest.raises(summarize.SummarizationError, match="refus"):
        summarize.summarize_transcript("transcript body", model="claude-haiku-4-5", max_output_tokens=1024)


def test_summarize_transcript_raises_when_output_did_not_parse(monkeypatch):
    class _FakeMessages:
        def parse(self, **kwargs):
            return _FakeParsedMessage(None, stop_reason="max_tokens")

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _FakeClient)

    with pytest.raises(summarize.SummarizationError, match="max_tokens"):
        summarize.summarize_transcript("transcript body", model="claude-haiku-4-5", max_output_tokens=1024)


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: anthropic.RateLimitError("boom", response=_fake_httpx_response(), body=None),
        lambda: anthropic.APIConnectionError(request=_fake_httpx_request()),
    ],
)
def test_summarize_transcript_wraps_sdk_exceptions(monkeypatch, exc_factory):
    class _FakeMessages:
        def parse(self, **kwargs):
            raise exc_factory()

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _FakeClient)

    with pytest.raises(summarize.SummarizationError):
        summarize.summarize_transcript("transcript body", model="claude-haiku-4-5", max_output_tokens=1024)


def test_summarize_transcript_wraps_client_construction_failure(monkeypatch):
    def _raise(*a, **k):
        raise anthropic.AnthropicError("no credentials found")

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _raise)

    with pytest.raises(summarize.SummarizationError, match="credentials"):
        summarize.summarize_transcript("transcript body", model="claude-haiku-4-5", max_output_tokens=1024)
