"""Calls Claude to turn an archived transcript into a structured,
timestamped insight artifact. Only the model call and its output validation
live here - see app/summary_store.py for idempotency and persistence, and
discover_and_process.py for how this fits into the serialized job."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Literal

import anthropic
from pydantic import BaseModel

logger = logging.getLogger("media_flow.summarize")

# Bumped whenever SYSTEM_PROMPT (or the output schema) changes. Deliberately a
# code constant, not an env var - drifting it independently of the prompt
# text would corrupt the idempotency check in summary_store.needs_summarization().
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are extracting structured, timestamped insights from a YouTube video transcript.

Identify the transcript-supported major and minor points made in the video, in the order they first appear. The number of points should reflect the video's actual substantive content - a short, thin video may have only one or two points; a long, information-dense video may have many. Do not pad the list to hit a target count, and do not omit real content to keep it short.

For each point:
- "importance" is "major" for a point central to the video's purpose, "minor" for a supporting or secondary point.
- "main_point" is one sentence or phrase stating the point.
- "explanation" is one to three sentences of supporting detail, using only what the transcript actually supports.
- "timestamp_seconds" is when in the video this point is made or first substantiated, in whole seconds, taken from the transcript's own timestamps.

Points must be listed in non-decreasing order of timestamp_seconds, matching the order they first appear in the video.

Be concise and factual. State uncertainty explicitly (e.g. "the speaker suggests..." vs "the speaker states...") rather than presenting an inference as a stated fact. Do not include information not supported by the transcript text.

Also provide:
- "subject": one concise phrase naming what the video is about.
- "summary": one to three sentences summarizing the video as a whole.
"""

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n\n", re.DOTALL)

# Matches the "[HH:MM:SS] " / "[MM:SS] " prefix youtube.render_transcript_markdown()
# puts at the start of every transcript line.
_TIMESTAMP_LINE_RE = re.compile(r"^\[(?:(\d+):)?(\d{1,2}):(\d{2})\] ", re.MULTILINE)

# From the Claude API pricing table, USD per million tokens (input, output).
# Unrecognized models return None from estimate_cost_usd() rather than
# guessing - app/config.py requires SUMMARY_MODEL to have an entry here so
# spend tracking can't silently under-count.
PRICING_PER_MTOK_USD = {
    "claude-haiku-4-5": (1.00, 5.00),
}

# Rough chars-per-token heuristic (English text) used only to reserve budget
# for an upcoming call before we know its real usage - not used for anything
# billed or persisted.
_CHARS_PER_TOKEN_ESTIMATE = 4


class SummarizationError(RuntimeError):
    """Raised for any provider/schema failure summarizing a single video -
    callers isolate this per-video rather than letting it abort a run.

    retryable distinguishes failures likely to succeed on a later attempt
    (rate limits, connection errors, transient malformed output) from ones
    that won't (an auth/credential failure, or a safety refusal, which is
    deterministic for the same input) - callers use this to stop retrying
    the latter instead of burning budget on a guaranteed repeat failure.

    usage, when not None, means the API actually returned a response (the
    call was billed) even though it's being treated as a failure - e.g. a
    safety refusal or an unparseable structured output still consumes
    tokens. Callers should count this usage against the run's budget the
    same as a successful call's."""

    def __init__(self, message: str, *, retryable: bool = True, usage: "Usage | None" = None):
        super().__init__(message)
        self.retryable = retryable
        self.usage = usage


class SummaryPoint(BaseModel):
    importance: Literal["major", "minor"]
    main_point: str
    explanation: str
    timestamp_seconds: int


class ModelSummaryOutput(BaseModel):
    """Only the fields the model is trusted to produce. Everything else in
    the persisted artifact (title, author, url, source ids, hash, model,
    prompt_version, display timestamps, usage, status) is filled in or
    derived by application code from the known source artifact - see
    summary_store.py. In particular, the human-readable "timestamp" string
    is never taken from the model - it's derived deterministically from
    timestamp_seconds via youtube.format_timestamp(), so the two can't
    disagree with each other."""

    subject: str
    summary: str
    points: list[SummaryPoint]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


def strip_frontmatter(markdown: str) -> str:
    """Removes the leading YAML frontmatter block that
    youtube.render_transcript_markdown() always produces, leaving only the
    transcript body. Used both to build the model prompt and to compute the
    idempotency hash - deliberately excluding fetched_at (which changes on
    every re-fetch even when captions are byte-identical) and the other
    frontmatter fields from the hash, so an unchanged transcript doesn't
    trigger a wasteful re-summarization just because it was re-fetched."""

    return _FRONTMATTER_RE.sub("", markdown, count=1)


def transcript_hash(transcript_body: str) -> str:
    return "sha256:" + hashlib.sha256(transcript_body.encode("utf-8")).hexdigest()


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = PRICING_PER_MTOK_USD.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


def estimate_worst_case_cost_usd(model: str, input_chars: int, max_output_tokens: int) -> float | None:
    """Conservative pre-flight estimate used to reserve budget before making
    a call, rather than only checking already-accumulated totals from prior
    calls: treats max_output_tokens as fully consumed (the real worst case)
    and a rough chars-per-token heuristic for input. Returns None if the
    model has no known pricing (same as estimate_cost_usd)."""

    estimated_input_tokens = max(1, input_chars // _CHARS_PER_TOKEN_ESTIMATE)
    return estimate_cost_usd(model, estimated_input_tokens, max_output_tokens)


def _max_transcript_seconds(transcript_body: str) -> int:
    """Highest timestamp actually present in the transcript body shown to
    the model, used to validate the model didn't invent an out-of-range
    timestamp_seconds for some point."""

    max_seconds = 0
    for match in _TIMESTAMP_LINE_RE.finditer(transcript_body):
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        max_seconds = max(max_seconds, hours * 3600 + minutes * 60 + seconds)
    return max_seconds


def _validate_points(points: list[SummaryPoint], transcript_body: str) -> None:
    """Raises ValueError if the model's points aren't trustworthy: a
    negative or out-of-range timestamp_seconds, or points out of
    chronological order (the prompt asks for "order they first appear" -
    Pydantic only validates types, not this)."""

    max_seconds = _max_transcript_seconds(transcript_body)
    previous_seconds = -1
    for point in points:
        if point.timestamp_seconds < 0 or point.timestamp_seconds > max_seconds:
            raise ValueError(
                f"timestamp_seconds={point.timestamp_seconds} is outside the transcript's own "
                f"range [0, {max_seconds}] for point {point.main_point!r}."
            )
        if point.timestamp_seconds < previous_seconds:
            raise ValueError(
                f"Points are not in non-decreasing timestamp order: {point.main_point!r} "
                f"({point.timestamp_seconds}s) comes after a point at {previous_seconds}s."
            )
        previous_seconds = point.timestamp_seconds


def summarize_transcript(
    transcript_body: str, *, model: str, max_output_tokens: int
) -> tuple[ModelSummaryOutput, Usage]:
    """Calls Claude to produce a ModelSummaryOutput for one transcript.
    Raises SummarizationError on any provider failure or invalid/unparseable
    output - never raises a raw SDK exception, so callers don't need to know
    the SDK's exception hierarchy."""

    try:
        client = anthropic.Anthropic()
        response = client.messages.parse(
            model=model,
            max_tokens=max_output_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": transcript_body}],
            output_format=ModelSummaryOutput,
        )
    except anthropic.RateLimitError as exc:
        # Transient - the same request will very likely succeed shortly.
        raise SummarizationError(f"Rate limited: {exc}", retryable=True) from exc
    except anthropic.APIConnectionError as exc:
        # Transient network-level failure, unrelated to the request itself.
        raise SummarizationError(f"Connection error: {exc}", retryable=True) from exc
    except anthropic.APIStatusError as exc:
        # 5xx (and 429, though that's normally RateLimitError above) are
        # transient; 4xx other than that reflects a bad request/auth
        # problem that will recur identically on retry.
        retryable = exc.status_code >= 500 or exc.status_code == 429
        raise SummarizationError(f"API error ({exc.status_code}): {exc.message}", retryable=retryable) from exc
    except anthropic.AnthropicError as exc:
        # Catch-all for anything not already covered above (e.g. a
        # credential-resolution failure at client construction, which is
        # never an HTTP-level error and so isn't an APIStatusError). Not
        # retryable - this needs a config/credential fix, not another
        # attempt.
        raise SummarizationError(f"Anthropic SDK error: {exc}", retryable=False) from exc

    response_usage = Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)

    if response.stop_reason == "refusal":
        # A safety refusal still consumes tokens (it's a completed
        # response), and is deterministic for the same input - not worth
        # retrying.
        raise SummarizationError(
            "Model declined to summarize this transcript (safety refusal).", retryable=False, usage=response_usage
        )
    if response.parsed_output is None:
        # A response was returned (and billed) but didn't parse against the
        # schema - plausibly a transient formatting hiccup, worth retrying.
        raise SummarizationError(
            f"Model response did not contain valid structured output (stop_reason={response.stop_reason!r}).",
            retryable=True,
            usage=response_usage,
        )

    try:
        _validate_points(response.parsed_output.points, transcript_body)
    except ValueError as exc:
        raise SummarizationError(f"Model output failed validation: {exc}", retryable=True, usage=response_usage) from exc

    return response.parsed_output, response_usage
