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
- "timestamp_seconds" and "timestamp" locate where in the video this point is made or first substantiated, taken from the transcript's own timestamps.

Be concise and factual. State uncertainty explicitly (e.g. "the speaker suggests..." vs "the speaker states...") rather than presenting an inference as a stated fact. Do not include information not supported by the transcript text.

Also provide:
- "subject": one concise phrase naming what the video is about.
- "summary": one to three sentences summarizing the video as a whole.
"""

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n\n", re.DOTALL)

# From the Claude API pricing table, USD per million tokens (input, output).
# Unrecognized models return None from estimate_cost_usd() rather than guessing.
_PRICING_PER_MTOK_USD = {
    "claude-haiku-4-5": (1.00, 5.00),
}


class SummarizationError(RuntimeError):
    """Raised for any provider/schema failure summarizing a single video -
    callers isolate this per-video rather than letting it abort a run."""


class SummaryPoint(BaseModel):
    importance: Literal["major", "minor"]
    main_point: str
    explanation: str
    timestamp_seconds: int
    timestamp: str


class ModelSummaryOutput(BaseModel):
    """Only the fields the model is trusted to produce. Everything else in
    the persisted artifact (title, author, url, source ids, hash, model,
    prompt_version, timestamps, usage, status) is filled in by application
    code from the known source artifact - see summary_store.py."""

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
    pricing = _PRICING_PER_MTOK_USD.get(model)
    if pricing is None:
        return None
    input_price, output_price = pricing
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


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
        raise SummarizationError(f"Rate limited: {exc}") from exc
    except anthropic.APIConnectionError as exc:
        raise SummarizationError(f"Connection error: {exc}") from exc
    except anthropic.APIStatusError as exc:
        raise SummarizationError(f"API error ({exc.status_code}): {exc.message}") from exc
    except anthropic.AnthropicError as exc:
        # Catch-all for anything not already covered above (e.g. a
        # credential-resolution failure at client construction, which is
        # never an HTTP-level error and so isn't an APIStatusError). Keeps
        # every SDK failure mode isolated to this one video rather than
        # letting an unanticipated exception type escape as a raw crash.
        raise SummarizationError(f"Anthropic SDK error: {exc}") from exc

    if response.stop_reason == "refusal":
        raise SummarizationError("Model declined to summarize this transcript (safety refusal).")
    if response.parsed_output is None:
        raise SummarizationError(
            f"Model response did not contain valid structured output (stop_reason={response.stop_reason!r})."
        )

    usage = Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
    return response.parsed_output, usage
