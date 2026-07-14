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


class _FakeTokenCount:
    def __init__(self, input_tokens):
        self.input_tokens = input_tokens


def test_count_prompt_tokens_returns_the_real_input_token_count(monkeypatch):
    seen_kwargs = {}

    class _FakeMessages:
        def count_tokens(self, **kwargs):
            seen_kwargs.update(kwargs)
            return _FakeTokenCount(input_tokens=321)

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _FakeClient)

    count = summarize.count_prompt_tokens(SAMPLE_BODY, model="claude-haiku-4-5")

    assert count == 321
    # Must include the real system prompt and output schema, not just the
    # transcript - otherwise the count undercounts exactly like the old
    # chars-per-token heuristic did.
    assert seen_kwargs["system"] == summarize.SYSTEM_PROMPT
    assert seen_kwargs["output_format"] is summarize.ModelSummaryOutput


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: anthropic.AuthenticationError("invalid api key", response=_fake_httpx_response(401), body=None),
        lambda: anthropic.PermissionDeniedError("no access", response=_fake_httpx_response(403), body=None),
    ],
)
def test_count_prompt_tokens_propagates_credential_failures_unwrapped(monkeypatch, exc_factory):
    """A genuine, still-broken credential problem is deliberately not
    wrapped in SummarizationError - see the function's docstring: this
    runs before a video's attempt count is touched or anything is written
    for it, so this specific failure mode should abort the whole run, not
    poison one video."""
    exc = exc_factory()

    class _FakeMessages:
        def count_tokens(self, **kwargs):
            raise exc

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _FakeClient)

    with pytest.raises(type(exc)):
        summarize.count_prompt_tokens(SAMPLE_BODY, model="claude-haiku-4-5")


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: anthropic.RateLimitError("boom", response=_fake_httpx_response(429), body=None),
        lambda: anthropic.APIConnectionError(request=_fake_httpx_request()),
        lambda: anthropic.APIStatusError("boom", response=_fake_httpx_response(500), body=None),
    ],
)
def test_count_prompt_tokens_wraps_transient_failures_as_retryable(monkeypatch, exc_factory):
    """Regression test: a blip on the token-counting endpoint specifically
    (rate limit, connection error, 5xx) must not abort the whole run and
    skip every remaining video with no durable retry state - only a
    genuine credential problem should do that."""

    class _FakeMessages:
        def count_tokens(self, **kwargs):
            raise exc_factory()

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _FakeMessages()

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _FakeClient)

    with pytest.raises(summarize.SummarizationError) as exc_info:
        summarize.count_prompt_tokens(SAMPLE_BODY, model="claude-haiku-4-5")
    assert exc_info.value.retryable is True


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


def _fake_client(parse_result_or_raiser, captured_init_kwargs=None):
    class _FakeMessages:
        def parse(self, **kwargs):
            if isinstance(parse_result_or_raiser, Exception):
                raise parse_result_or_raiser
            return parse_result_or_raiser

    class _FakeClient:
        def __init__(self, *a, **k):
            if captured_init_kwargs is not None:
                captured_init_kwargs.update(k)
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


def test_summarize_transcript_disables_the_sdks_own_internal_retries(monkeypatch):
    """Regression test: the SDK's default max_retries=2 (each up to a
    10-minute timeout) can silently run a single call for up to ~30
    minutes - right up against DISCOVERY_LOCK_TTL_SECONDS's default with
    no chance to renew the lock in between. Our own outer per-video retry
    loop is the sole retry authority now."""
    captured = {}
    fake = _FakeParsedMessage(
        summarize.ModelSummaryOutput(
            subject="S", summary="S.", points=[summarize.SummaryPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=0)]
        )
    )
    monkeypatch.setattr(summarize.anthropic, "Anthropic", _fake_client(fake, captured_init_kwargs=captured))

    summarize.summarize_transcript(SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024)

    assert captured["max_retries"] == 0


def test_summarize_transcript_raises_on_real_pydantic_validation_error(monkeypatch):
    """Regression test: the pinned SDK's messages.parse() validates the
    model's JSON *inside* the same call via a post_parser hook, so a
    malformed/truncated response raises pydantic.ValidationError directly
    out of client.messages.parse() - not any anthropic.* exception type.
    Uncaught, this would escape summarize_transcript() and abort the whole
    run instead of being isolated to one video."""
    try:
        summarize.pydantic.TypeAdapter(summarize.ModelSummaryOutput).validate_json("not valid json at all")
    except summarize.pydantic.ValidationError as exc:
        real_validation_error = exc

    monkeypatch.setattr(summarize.anthropic, "Anthropic", _fake_client(real_validation_error))

    with pytest.raises(summarize.SummarizationError, match="schema validation") as exc_info:
        summarize.summarize_transcript(SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024)
    assert exc_info.value.retryable is True
    # Real usage is unavailable in this specific failure path, but a
    # response plausibly still happened and was billed - callers must
    # conservatively account for that rather than assuming zero cost.
    assert exc_info.value.usage is None
    assert exc_info.value.possibly_billed is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subject": "", "summary": "S.", "points": [{"importance": "major", "main_point": "P", "explanation": "E", "timestamp_seconds": 0}]},
        {"subject": "S", "summary": "", "points": [{"importance": "major", "main_point": "P", "explanation": "E", "timestamp_seconds": 0}]},
        {"subject": "S", "summary": "S.", "points": []},
    ],
)
def test_model_summary_output_rejects_empty_content(kwargs):
    """Regression test: an empty points list or blank subject/summary is
    schema-valid by Pydantic's default rules despite the output contract
    requiring actual timestamped insights."""
    with pytest.raises(summarize.pydantic.ValidationError):
        summarize.ModelSummaryOutput(**kwargs)


@pytest.mark.parametrize("field_name", ["main_point", "explanation"])
def test_summary_point_rejects_empty_strings(field_name):
    kwargs = {"importance": "major", "main_point": "P", "explanation": "E", "timestamp_seconds": 0}
    kwargs[field_name] = ""
    with pytest.raises(summarize.pydantic.ValidationError):
        summarize.SummaryPoint(**kwargs)
