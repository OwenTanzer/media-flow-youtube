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

# A richer fixture for _resolve_points()/anchor-window tests: a topic
# ("Palantir") mentioned twice, plus unrelated surrounding lines, so tests
# can exercise "anchor found on the exact cited line", "anchor found a
# couple lines away (within the window)", and "anchor nowhere nearby".
RICH_BODY = (
    "[00:00] Welcome back to the show everyone.\n"
    "[00:10] Palantir is testing resistance near 40 dollars today.\n"
    "[00:20] Traders are watching the 40 dollar level closely.\n"
    "[00:30] Meanwhile crude oil slipped below 70 dollars a barrel.\n"
    "[00:40] That's it for today, thanks for watching.\n"
)


def _point(source_timestamp="[00:05]", source_anchor="world", **overrides):
    kwargs = dict(
        importance="major",
        main_point="P",
        explanation="E",
        source_timestamp=source_timestamp,
        source_anchor=source_anchor,
    )
    kwargs.update(overrides)
    return summarize.SummaryPoint(**kwargs)


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
    # Must include the real system prompt (dynamically built for this
    # video's length) and output schema, not just the transcript -
    # otherwise the count undercounts exactly like the old chars-per-token
    # heuristic did.
    max_points = summarize._max_points_for_duration(summarize._max_transcript_seconds(SAMPLE_BODY))
    assert seen_kwargs["system"] == summarize._build_system_prompt(max_points)
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


def test_index_transcript_lines_returns_seconds_and_text_in_order():
    indexed = summarize._index_transcript_lines(RICH_BODY)
    assert [seconds for seconds, _ in indexed] == [0, 10, 20, 30, 40]
    assert indexed[1][1] == "Palantir is testing resistance near 40 dollars today."


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[14:32]", 872),
        ("14:32", 872),
        ("[1:02:15]", 3735),
        ("1:02:15", 3735),
        ("[00:05]", 5),
    ],
)
def test_parse_source_timestamp_parses_bracketed_and_bare_forms(raw, expected):
    assert summarize._parse_source_timestamp(raw) == expected


@pytest.mark.parametrize("raw", ["", "not a timestamp", "14:32:99:11", "[]", "yesterday"])
def test_parse_source_timestamp_returns_none_for_garbage(raw):
    assert summarize._parse_source_timestamp(raw) is None


def test_resolve_points_accepts_a_citation_on_the_exact_line():
    points = [_point(source_timestamp="[00:10]", source_anchor="Palantir is testing resistance")]
    resolved = summarize._resolve_points(points, RICH_BODY)
    assert resolved == [
        summarize.ResolvedPoint(importance="major", main_point="P", explanation="E", timestamp_seconds=10)
    ]


def test_resolve_points_accepts_anchor_within_the_window_but_not_on_the_exact_line():
    # Cited line is [00:10]; the anchor text actually lives on [00:30],
    # two lines later - within _ANCHOR_WINDOW_LINES (2).
    points = [_point(source_timestamp="[00:10]", source_anchor="crude oil slipped")]
    resolved = summarize._resolve_points(points, RICH_BODY)
    assert resolved[0].timestamp_seconds == 10


def test_resolve_points_rejects_anchor_outside_the_window():
    # [00:00] and [00:40] are 4 lines apart - outside the +-2 line window.
    points = [_point(source_timestamp="[00:00]", source_anchor="thanks for watching")]
    with pytest.raises(ValueError, match="was not found within"):
        summarize._resolve_points(points, RICH_BODY)


def test_resolve_points_anchor_match_is_case_and_whitespace_insensitive():
    points = [_point(source_timestamp="[00:10]", source_anchor="  PALANTIR is   testing RESISTANCE  ")]
    resolved = summarize._resolve_points(points, RICH_BODY)
    assert resolved[0].timestamp_seconds == 10


def test_resolve_points_rejects_unparseable_source_timestamp():
    points = [_point(source_timestamp="not a timestamp")]
    with pytest.raises(ValueError, match="not a valid"):
        summarize._resolve_points(points, SAMPLE_BODY)


def test_resolve_points_rejects_a_timestamp_that_is_not_a_real_transcript_line():
    # 999s isn't anywhere in SAMPLE_BODY (only 0s and 5s exist) - this is
    # the strict-equality replacement for the old "out of range" check,
    # and also rejects a plausible-looking but non-real in-range value.
    points = [_point(source_timestamp="[00:02]", source_anchor="hello")]
    with pytest.raises(ValueError, match="not one of the transcript's own line timestamps"):
        summarize._resolve_points(points, SAMPLE_BODY)


def test_resolve_points_rejects_anchor_text_that_does_not_appear_at_all():
    points = [_point(source_timestamp="[00:05]", source_anchor="something never said")]
    with pytest.raises(ValueError, match="was not found within"):
        summarize._resolve_points(points, SAMPLE_BODY)


def test_resolve_points_accepts_nonchronological_order():
    """Regression test: real videos (livestreams especially) revisit the
    same topic more than once, and a strict ordering requirement rejected
    genuinely well-formed output for that content - points only need to
    each independently resolve, not be strictly ordered."""
    points = [
        _point(source_timestamp="[00:05]", source_anchor="world"),
        _point(source_timestamp="[00:00]", source_anchor="hello"),
    ]
    resolved = summarize._resolve_points(points, SAMPLE_BODY)
    assert [p.timestamp_seconds for p in resolved] == [5, 0]


def test_max_points_for_duration_scales_with_length_and_has_a_ceiling():
    assert summarize._max_points_for_duration(0) == 1
    assert summarize._max_points_for_duration(179) == 1
    assert summarize._max_points_for_duration(180) == 1
    assert summarize._max_points_for_duration(360) == 2
    assert summarize._max_points_for_duration(999_999) == summarize._MAX_POINTS_CEILING


def _resolved(importance, timestamp_seconds, main_point="P"):
    return summarize.ResolvedPoint(importance=importance, main_point=main_point, explanation="E", timestamp_seconds=timestamp_seconds)


def test_select_significant_points_keeps_majors_over_minors_when_over_the_cap():
    major_a = _resolved("major", 0, "Major A")
    minor_a = _resolved("minor", 1, "Minor A")
    major_b = _resolved("major", 2, "Major B")
    minor_b = _resolved("minor", 3, "Minor B")

    selected, truncated = summarize._select_significant_points([major_a, minor_a, major_b, minor_b], max_points=2)

    assert truncated is True
    assert selected == [major_a, major_b]


def test_select_significant_points_preserves_original_order_among_kept_points():
    p1 = _resolved("major", 0, "P1")
    p2 = _resolved("minor", 1, "P2")
    p3 = _resolved("major", 2, "P3")

    selected, truncated = summarize._select_significant_points([p1, p2, p3], max_points=2)

    assert truncated is True
    # p1 and p3 are both major and kept; original relative order preserved.
    assert selected == [p1, p3]


def test_select_significant_points_does_not_truncate_when_under_the_cap():
    p1 = _resolved("major", 0, "P1")
    selected, truncated = summarize._select_significant_points([p1], max_points=5)
    assert truncated is False
    assert selected == [p1]


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
        video_type="Analytic Overview",
        summary="A summary.",
        points=[_point(source_timestamp="[00:05]", source_anchor="world", main_point="Point one", explanation="Because X.")],
    )
    monkeypatch.setattr(
        summarize.anthropic, "Anthropic", _fake_client(_FakeParsedMessage(expected_output, usage=_FakeUsage(123, 45)))
    )

    output, usage, points_truncated = summarize.summarize_transcript(
        SAMPLE_BODY, model="claude-haiku-4-5", max_output_tokens=1024
    )

    assert output.video_type == "Analytic Overview"
    assert output.summary == "A summary."
    assert output.points == [
        summarize.ResolvedPoint(importance="major", main_point="Point one", explanation="Because X.", timestamp_seconds=5)
    ]
    assert usage.input_tokens == 123
    assert usage.output_tokens == 45
    assert points_truncated is False


def test_summarize_transcript_raises_on_invalid_points(monkeypatch):
    """The model can return a well-typed but ungrounded citation - Pydantic
    alone can't catch this, since it only validates the string shape, not
    whether the cited line and excerpt are real."""
    bad_output = summarize.ModelSummaryOutput(
        video_type="Analytic Overview",
        summary="S.",
        points=[_point(source_timestamp="[00:02]", source_anchor="hello")],  # 2s isn't a real line in SAMPLE_BODY
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
            video_type="Analytic Overview", summary="S.", points=[_point(source_timestamp="[00:00]", source_anchor="hello")]
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
        {
            "video_type": "Not A Real Type",
            "summary": "S.",
            "points": [{"importance": "major", "main_point": "P", "explanation": "E", "source_timestamp": "[00:00]", "source_anchor": "hello"}],
        },
        {
            "video_type": "Analytic Overview",
            "summary": "",
            "points": [{"importance": "major", "main_point": "P", "explanation": "E", "source_timestamp": "[00:00]", "source_anchor": "hello"}],
        },
        {"video_type": "Analytic Overview", "summary": "S.", "points": []},
    ],
)
def test_model_summary_output_rejects_empty_or_invalid_content(kwargs):
    """Regression test: an empty points list, blank summary, or a
    video_type outside the four allowed categories must be rejected -
    an empty/mistyped response is otherwise schema-valid by Pydantic's
    default rules despite the output contract requiring actual content."""
    with pytest.raises(summarize.pydantic.ValidationError):
        summarize.ModelSummaryOutput(**kwargs)


def test_model_summary_output_accepts_each_valid_video_type():
    for video_type in summarize.VIDEO_TYPES:
        summarize.ModelSummaryOutput(
            video_type=video_type,
            summary="S.",
            points=[_point(source_timestamp="[00:00]", source_anchor="hello")],
        )  # should not raise


@pytest.mark.parametrize("field_name", ["main_point", "explanation", "source_timestamp", "source_anchor"])
def test_summary_point_rejects_empty_strings(field_name):
    kwargs = dict(importance="major", main_point="P", explanation="E", source_timestamp="[00:00]", source_anchor="hello")
    kwargs[field_name] = ""
    with pytest.raises(summarize.pydantic.ValidationError):
        summarize.SummaryPoint(**kwargs)
