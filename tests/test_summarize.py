import anthropic
import httpx
import pytest

from app import summarize


def _fake_httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _fake_httpx_response(status_code: int = 429) -> httpx.Response:
    return httpx.Response(status_code, request=_fake_httpx_request())

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

SAMPLE_BODY = "[00:00] hello\n[00:05] world\n"


def test_strip_frontmatter_removes_only_the_leading_block():
    body = summarize.strip_frontmatter(SAMPLE_MARKDOWN)
    assert body == SAMPLE_BODY


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


def test_estimate_worst_case_cost_usd_uses_max_output_tokens_as_ceiling():
    # 4000 chars / 4 chars-per-token estimate = 1000 input tokens.
    cost = summarize.estimate_worst_case_cost_usd("claude-haiku-4-5", input_chars=4000, max_output_tokens=2000)
    expected = (1000 / 1_000_000) * 1.00 + (2000 / 1_000_000) * 5.00
    assert cost == pytest.approx(expected)


def test_estimate_worst_case_cost_usd_unknown_model_returns_none():
    assert summarize.estimate_worst_case_cost_usd("some-future-model", 4000, 2000) is None


def test_max_transcript_seconds_finds_the_last_timestamp():
    assert summarize._max_transcript_seconds(SAMPLE_BODY) == 5
    assert summarize._max_transcript_seconds("[01:02:03] hours case\n") == 3723
    assert summarize._max_transcript_seconds("no timestamps here") == 0


def test_validate_points_rejects_out_of_range_timestamp():
    points = [summarize.SummaryPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=999)]
    with pytest.raises(ValueError, match="outside the transcript's own range"):
        summarize._validate_points(points, SAMPLE_BODY)


def test_validate_points_rejects_negative_timestamp():
    points = [summarize.SummaryPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=-1)]
    with pytest.raises(ValueError, match="outside the transcript's own range"):
        summarize._validate_points(points, SAMPLE_BODY)


def test_validate_points_rejects_nonchronological_order():
    points = [
        summarize.SummaryPoint(importance="major", main_point="Second", explanation="E", timestamp_seconds=5),
        summarize.SummaryPoint(importance="minor", main_point="First", explanation="E", timestamp_seconds=0),
    ]
    with pytest.raises(ValueError, match="not in non-decreasing timestamp order"):
        summarize._validate_points(points, SAMPLE_BODY)


def test_validate_points_accepts_valid_in_range_chronological_points():
    points = [
        summarize.SummaryPoint(importance="major", main_point="First", explanation="E", timestamp_seconds=0),
        summarize.SummaryPoint(importance="minor", main_point="Second", explanation="E", timestamp_seconds=5),
    ]
    summarize._validate_points(points, SAMPLE_BODY)  # should not raise


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeParsedMessage:
    def __init__(self, parsed_output, stop_reason="end_turn", usage=None, status_code=None):
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason
        self.usage = usage or _FakeUsage(100, 50)
        self.status_code = status_code


def _fake_client(parse_result_or_raiser):
    class _FakeMessages:
        def parse(self, **kwargs):
            if isinstance(parse_result_or_raiser, Exception):
                raise parse_result_or_raiser
            return parse_result_or_raiser

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    return _FakeClient


def test_summarize_transcript_success(monkeypatch):
    expected_output = summarize.ModelSummaryOutput(
        subject="A subject",
        summary="A summary.",
        points=[summarize.SummaryPoint(importance="major", main_point="Point one", explanation="Because X.", timestamp_seconds=5)],
    )
    monkeypatch.setattr(
        summarize.anthropic, "Anthropic", _fake_client(_FakeParsedMessage(expected_output, usage=_FakeUsage(123, 45)))
    )

    output, usage = summarize.summarize_transcript(SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024)

    assert output == expected_output
    assert usage.input_tokens == 123
    assert usage.output_tokens == 45


def test_summarize_transcript_raises_on_invalid_points(monkeypatch):
    """The model can return well-typed but out-of-range/out-of-order points
    - Pydantic alone can't catch this, since it only validates int/str
    shape, not values against the actual transcript."""
    bad_output = summarize.ModelSummaryOutput(
        subject="S",
        summary="S.",
        points=[summarize.SummaryPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=9999)],
    )
    monkeypatch.setattr(summarize.anthropic, "Anthropic", _fake_client(_FakeParsedMessage(bad_output)))

    with pytest.raises(summarize.SummarizationError) as exc_info:
        summarize.summarize_transcript(SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024)
    assert exc_info.value.retryable is True
    assert exc_info.value.usage is not None


def test_summarize_transcript_raises_on_refusal(monkeypatch):
    monkeypatch.setattr(
        summarize.anthropic, "Anthropic", _fake_client(_FakeParsedMessage(None, stop_reason="refusal", usage=_FakeUsage(200, 10)))
    )

    with pytest.raises(summarize.SummarizationError, match="refus") as exc_info:
        summarize.summarize_transcript(SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024)

    # A refusal is deterministic for the same input (not worth retrying) but
    # still billed - both must be reflected on the exception so callers can
    # count real spend without retrying a guaranteed repeat failure.
    assert exc_info.value.retryable is False
    assert exc_info.value.usage.input_tokens == 200
    assert exc_info.value.usage.output_tokens == 10


def test_summarize_transcript_raises_when_output_did_not_parse(monkeypatch):
    monkeypatch.setattr(
        summarize.anthropic,
        "Anthropic",
        _fake_client(_FakeParsedMessage(None, stop_reason="max_tokens", usage=_FakeUsage(300, 20))),
    )

    with pytest.raises(summarize.SummarizationError, match="max_tokens") as exc_info:
        summarize.summarize_transcript(SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024)

    # Plausibly transient (a formatting hiccup) - worth retrying - but the
    # response was still billed, so usage must be counted.
    assert exc_info.value.retryable is True
    assert exc_info.value.usage.input_tokens == 300


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: anthropic.RateLimitError("boom", response=_fake_httpx_response(429), body=None),
        lambda: anthropic.APIConnectionError(request=_fake_httpx_request()),
    ],
)
def test_summarize_transcript_wraps_transient_sdk_exceptions_as_retryable(monkeypatch, exc_factory):
    monkeypatch.setattr(summarize.anthropic, "Anthropic", _fake_client(exc_factory()))

    with pytest.raises(summarize.SummarizationError) as exc_info:
        summarize.summarize_transcript(SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024)
    assert exc_info.value.retryable is True
    assert exc_info.value.usage is None


@pytest.mark.parametrize("status_code,expected_retryable", [(500, True), (429, True), (400, False), (401, False), (403, False)])
def test_summarize_transcript_api_status_error_retryability_depends_on_status(monkeypatch, status_code, expected_retryable):
    exc = anthropic.APIStatusError("boom", response=_fake_httpx_response(status_code), body=None)
    monkeypatch.setattr(summarize.anthropic, "Anthropic", _fake_client(exc))

    with pytest.raises(summarize.SummarizationError) as exc_info:
        summarize.summarize_transcript(SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024)
    assert exc_info.value.retryable is expected_retryable


def test_summarize_transcript_wraps_client_construction_failure(monkeypatch):
    def _raise(*a, **k):
        raise anthropic.AnthropicError("no credentials found")

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _raise)

    with pytest.raises(summarize.SummarizationError, match="credentials") as exc_info:
        summarize.summarize_transcript(SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024)
    # Needs a config/credential fix, not another attempt.
    assert exc_info.value.retryable is False
