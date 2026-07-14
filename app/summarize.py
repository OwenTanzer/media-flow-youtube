"""Calls Claude to turn an archived transcript into a structured,
timestamped insight artifact. Only the model call and its output validation
live here - see app/summary_store.py for idempotency and persistence, and
discover_and_process.py for how this fits into the serialized job."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Literal

import anthropic
import pydantic
from pydantic import BaseModel, Field

logger = logging.getLogger("media_flow.summarize")

# Bumped whenever the system prompt template (or the output schema)
# changes. Deliberately a code constant, not an env var - drifting it
# independently of the prompt text would corrupt the idempotency check in
# summary_store.needs_summarization().
PROMPT_VERSION = "v6"

# Convenience tuple for callers/tests - must be kept in sync with
# ModelSummaryOutput.video_type's Literal values below.
VIDEO_TYPES = ("Post-Market Update", "Thesis Piece", "Analytic Overview", "Pre-Market Brief")

# Upper bound on how many points the model may return, scaling with video
# length but never exceeding _MAX_POINTS_CEILING - without this, a very
# long video (a multi-hour livestream, say) has no natural ceiling on how
# many points a model might try to produce. This is an upper bound only:
# short videos are already naturally kept to a handful of points by the
# prompt's own guidance below, so this rarely binds for them.
_POINT_INTERVAL_SECONDS = 180  # roughly one point per 3 minutes of video
_MAX_POINTS_CEILING = 20


def _max_points_for_duration(duration_seconds: int) -> int:
    return max(1, min(_MAX_POINTS_CEILING, duration_seconds // _POINT_INTERVAL_SECONDS))


def _build_system_prompt(max_points: int) -> str:
    return f"""You are extracting structured, timestamped insights from a YouTube video transcript.

Identify the transcript-supported major and minor points made in the video. Include at most {max_points} points - however many major/minor points are actually present in the video's substantive content, up to that limit. A short, thin video may have only one or two points; a long, information-dense video may have many, up to {max_points}. Do not pad the list to hit a target count. If the video's substantive content exceeds {max_points} distinguishable points, select only the {max_points} most significant ones (favoring "major" points over "minor" ones) rather than trying to cram in everything - do not omit real content to keep the list short otherwise.

Do NOT create points for routine housekeeping/administrative content - schedule or streaming-cadence announcements, membership/Patreon/sponsor plugs, "welcome back"/sign-off preambles, asks to like/subscribe, or similar channel-admin remarks - even if the transcript spends real time on them. Exclude this by default so the (especially limited, on longer videos) point budget goes to actual substantive content instead. The only exception: if a piece of "housekeeping" is itself substantively important to understanding the video (e.g. a schedule change that materially affects when to expect the next analysis), it may still get a point.

Every transcript line is prefixed with its own timestamp in brackets, e.g. "[14:32] ..." or "[1:02:15] ...". For each point:
- "importance" is "major" for a point central to the video's purpose, "minor" for a supporting or secondary point.
- "main_point" is one sentence or phrase stating the point.
- "explanation" is 2 to 4 sentences of supporting detail, using only what the transcript actually supports - favor giving real substance and specifics (numbers, reasoning, context) over being terse.
- "source_timestamp" is the exact bracketed timestamp of ONE transcript line that supports this point - copy it verbatim, brackets included (e.g. "[14:32]" or "[1:02:15]"). Do not compute, estimate, convert, or invent anything here - just copy the bracket text of a real transcript line.
- "source_anchor" is a short excerpt (a few words to one sentence) copied verbatim from that same transcript line, so your citation can be checked against the transcript. Do not paraphrase or summarize it - copy the actual words.

Many videos (livestreams especially) revisit the same topic more than once - e.g. the same asset, subject, or claim comes up early, then again later in more depth. When that happens, "explanation" may still draw on the fullest picture of what was said about it across all those mentions - but "source_timestamp"/"source_anchor" must always point at ONE single real line you are citing, never a value that combines, averages, or estimates across multiple mentions. Pick whichever one mention you're citing evidence from. Points do not need to be in strict chronological order if a topic is revisited.

Be concise and factual. State uncertainty explicitly (e.g. "the speaker suggests..." vs "the speaker states...") rather than presenting an inference as a stated fact. Do not include information not supported by the transcript text.

Also provide:
- "video_type": classify the video as exactly one of "Post-Market Update", "Pre-Market Brief", "Thesis Piece", or "Analytic Overview".
  - "Post-Market Update": a recap/review of a trading session or period that has already happened.
  - "Pre-Market Brief": forward-looking commentary or a plan for a session/period that hasn't happened yet.
  - "Thesis Piece": an in-depth case for or against a single specific idea, asset, or claim.
  - "Analytic Overview": a broader technical/analytical walkthrough across multiple assets or topics, not tied to a single specific thesis or session (e.g. a live multi-asset chart-reading session).
  If more than one could apply, pick whichever describes the video's primary purpose.
- "summary": one to three sentences summarizing the video as a whole.
"""

def _build_fallback_system_prompt() -> str:
    return """You are writing a summary of a YouTube video transcript for a viewer who wants the gist without watching it.

Write 2 to 3 paragraphs covering the main ideas, claims, and conclusions the speaker makes, in your own words. Cover the video as a whole - do not focus narrowly on just its opening or closing minutes. Do not include timestamps, line citations, or references to "the transcript" - write as if describing the video's content directly.

Be concise and factual. State uncertainty explicitly (e.g. "the speaker suggests..." vs "the speaker states...") rather than presenting an inference as a stated fact. Do not include information not supported by the transcript text. Skip routine housekeeping/administrative content - schedule announcements, membership/sponsor plugs, sign-off preambles, like-and-subscribe asks - unless it's itself substantively important to understanding the video."""


_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n\n", re.DOTALL)

# Matches the "[HH:MM:SS] " / "[MM:SS] " prefix youtube.render_transcript_markdown()
# puts at the start of every transcript line.
_TIMESTAMP_LINE_RE = re.compile(r"^\[(?:(\d+):)?(\d{1,2}):(\d{2})\] ", re.MULTILINE)

# Matches a bare "source_timestamp" value the model is asked to copy
# verbatim from a transcript line's own bracket - e.g. "[14:32]" or
# "[1:02:15]" - optionally without the brackets, in case the model drops
# them despite the prompt's instruction to keep them.
_SOURCE_TIMESTAMP_RE = re.compile(r"^\[?(?:(\d+):)?(\d{1,2}):(\d{2})\]?$")

# How many transcript lines on either side of the cited line source_anchor
# is allowed to be found in - a little slack for an excerpt that straddles
# a line break, without being loose enough to accept a citation for
# unrelated content elsewhere in the transcript.
_ANCHOR_WINDOW_LINES = 2

# Fraction of source_anchor's significant words that must appear in the
# cited window for the anchor to count as grounded. Not a stricter exact-
# substring match: live testing showed the model reliably finds the right
# real transcript line (source_timestamp was correct 16/16 times) but
# routinely cleans up raw, informal ASR captions into a grammatical
# paraphrase when asked to "copy verbatim" - so a byte-exact substring
# requirement rejected genuinely well-grounded citations, not just
# hallucinated ones. Word overlap tolerates that paraphrasing while still
# catching an anchor that describes unrelated content.
#
# 0.4 was the initial guess; a live A/B re-test against it showed several
# genuinely well-grounded paraphrases clustered at 29-36% overlap - just
# under the cutoff - while citations describing clearly unrelated content
# scored far lower (0-12%). 0.25 was chosen empirically from that gap: it
# recovers the near-miss cluster without accepting the low-overlap ones,
# which stay rejected (and simply get retried on a later run) rather than
# being waved through on a guess.
_ANCHOR_WORD_OVERLAP_THRESHOLD = 0.25

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """
    a an the and or but is are was were be been being to of in on for with
    that this these those it its as at by from up down out over under
    again then once here there when where why how all any both each few
    more most other some such no nor not only own same so than too very
    can will just should now we you he she they i them their what which
    who whom
    """.split()
)


def _significant_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}

# From the Claude API pricing table, USD per million tokens (input, output).
# Unrecognized models return None from estimate_cost_usd() rather than
# guessing - app/config.py requires SUMMARY_MODEL to have an entry here so
# spend tracking can't silently under-count.
PRICING_PER_MTOK_USD = {
    "claude-haiku-4-5": (1.00, 5.00),
}

# The SDK's own automatic retry/backoff (default: 2 retries, each up to a
# 10-minute timeout - up to ~30 minutes for one call) stacks badly with our
# own outer per-video retry loop (SUMMARY_MAX_ATTEMPTS_PER_VIDEO, across
# scheduled runs) and can silently run right up against
# DISCOVERY_LOCK_TTL_SECONDS's default 30-minute window with no chance to
# renew the lock in between. Disabling it bounds a single call to roughly
# one request's timeout and makes our own outer retry (which the lock IS
# renewed around) the sole retry authority - the same fix already applied
# to the Webshare proxy's internal retries in app/youtube.py.
_CLIENT_MAX_RETRIES = 0


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
    same as a successful call's.

    possibly_billed is for the rarer case where usage is unknown but a
    response plausibly still happened (the SDK's own schema validation
    raising pydantic.ValidationError, where the exception propagates
    before real usage is accessible) - as opposed to usage=None meaning
    definitely not billed (e.g. a connection error before any response).
    Callers should conservatively charge an estimated cost against the
    run's budget in this case rather than contributing zero, since
    repeated failures like this could otherwise let real spend exceed the
    configured cap unnoticed.

    fallback_eligible marks a failure as a content/structured-output
    problem with *this specific attempt's response* - unparseable/invalid
    structured output, or a citation that failed _resolve_points()'s
    grounding check - as opposed to a provider-level problem (rate limit,
    connection error, 5xx) that's retryable in the sense that a *later*
    attempt may succeed, but where an immediate second call right now
    can't fix anything and, for a rate limit specifically, actively makes
    it worse. Only fallback_eligible failures are candidates for
    summary_store.summarize_eligible()'s last-attempt fallback path
    (summarize_fallback()); every other retryable failure just waits for
    next_retry_at like normal, even on the final attempt."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        usage: "Usage | None" = None,
        possibly_billed: bool = False,
        fallback_eligible: bool = False,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.usage = usage
        self.possibly_billed = possibly_billed
        self.fallback_eligible = fallback_eligible


class SummaryPoint(BaseModel):
    """The schema the model must produce for one point. Deliberately asks
    for literal transcript evidence only, never a computed value: the
    model copies a real line's bracketed timestamp and a verbatim excerpt
    from it, and application code (_resolve_points()) is solely
    responsible for verifying that evidence against the transcript and
    converting it to timestamp_seconds - the earlier design let the model
    both "remember" a line and compute its own timestamp_seconds in one
    step, and validated only that the result was plausible (in range),
    which let a wrong-but-plausible number through. Requiring literal
    copy-paste evidence, checked against the transcript, is a strictly
    stronger guarantee than range-checking a self-reported number."""

    importance: Literal["major", "minor"]
    main_point: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    source_timestamp: str = Field(min_length=1)
    source_anchor: str = Field(min_length=1)


class ModelSummaryOutput(BaseModel):
    """Only the fields the model is trusted to produce. Everything else in
    the persisted artifact (title, author, url, source ids, hash, model,
    prompt_version, display timestamps, usage, status) is filled in or
    derived by application code from the known source artifact - see
    summary_store.py. In particular, points' timestamp_seconds and the
    human-readable "timestamp" string are never taken from the model - see
    ResolvedPoint below.

    Minimum lengths on every field (including requiring at least one point)
    are deliberate: an empty/blank response is schema-valid by Pydantic's
    default rules despite the output contract requiring actual timestamped
    insights, so without these an empty summary would be accepted as "ok"."""

    video_type: Literal["Post-Market Update", "Thesis Piece", "Analytic Overview", "Pre-Market Brief"]
    summary: str = Field(min_length=1)
    points: list[SummaryPoint] = Field(min_length=1)


class FallbackSummaryOutput(BaseModel):
    """The much simpler schema used when a video has exhausted its normal
    per-point citation attempts (see summarize_fallback() and
    summary_store.summarize_eligible()'s last-attempt handling). Some
    speakers - meandering, conversational, non-linear delivery - make it
    hard for the model to pin one point to one specific transcript line
    even though it clearly understood the content; asking for a plain
    prose summary instead sidesteps that entirely, since there's no
    per-line citation to get wrong. No timestamps, no points - just a
    freeform summary of the video as a whole."""

    summary: str = Field(min_length=1)


@dataclass
class ResolvedPoint:
    """A SummaryPoint after its source_timestamp/source_anchor have been
    verified against the transcript and converted to a real
    timestamp_seconds - see _resolve_points(). This, not SummaryPoint, is
    what summarize_transcript() actually returns to callers (see
    ResolvedSummary): nothing downstream ever sees a model-reported
    timestamp, computed or otherwise."""

    importance: Literal["major", "minor"]
    main_point: str
    explanation: str
    timestamp_seconds: int


@dataclass
class ResolvedSummary:
    """summarize_transcript()'s actual return shape - same fields as
    ModelSummaryOutput, but with points resolved (see ResolvedPoint)."""

    video_type: str
    summary: str
    points: list[ResolvedPoint]


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


def count_prompt_tokens(transcript_body: str, *, model: str) -> int:
    """Real input token count for this exact prompt (system prompt + output
    schema overhead + the transcript itself) via the Anthropic token-
    counting endpoint - used to reserve accurate per-run budget before a
    call, replacing an earlier chars-per-token heuristic that both
    excluded the system prompt/schema and wasn't a true upper bound either
    way.

    A genuine, still-broken credential problem (AuthenticationError /
    PermissionDeniedError - the key is present but invalid/expired/lacks
    access) propagates unwrapped: this call happens before
    summarize_eligible() commits to processing a given video (before its
    attempt count is touched or anything is written for it), so this
    specific failure mode should abort the whole run rather than being
    recorded as a permanent per-video failure - a credential problem isn't
    a property of any one video's content, and poisoning every video it
    touches wouldn't get fixed by fixing the credential, since nothing
    about the video's own hash/model/prompt_version changes.

    Anything else recognized as transient (rate limits, connection errors,
    5xx) is wrapped in a retryable SummarizationError instead, same as
    summarize_transcript()'s own classification - a blip on this endpoint
    specifically must not abort the whole run and skip every remaining
    video with no durable retry state, the way an unconditional re-raise
    would."""

    max_points = _max_points_for_duration(_max_transcript_seconds(transcript_body))
    try:
        client = anthropic.Anthropic(max_retries=_CLIENT_MAX_RETRIES)
        result = client.messages.count_tokens(
            model=model,
            system=_build_system_prompt(max_points),
            messages=[{"role": "user", "content": transcript_body}],
            output_format=ModelSummaryOutput,
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        raise
    except anthropic.RateLimitError as exc:
        raise SummarizationError(f"Token count rate limited: {exc}", retryable=True) from exc
    except anthropic.APIConnectionError as exc:
        raise SummarizationError(f"Token count connection error: {exc}", retryable=True) from exc
    except anthropic.APIStatusError as exc:
        retryable = exc.status_code >= 500 or exc.status_code == 429
        raise SummarizationError(f"Token count API error ({exc.status_code}): {exc.message}", retryable=retryable) from exc
    except anthropic.AnthropicError:
        # Anything else unanticipated (e.g. a credential-resolution
        # failure at client construction) is treated conservatively as a
        # config/environment problem, same as AuthenticationError above,
        # rather than guessed to be transient.
        raise
    return result.input_tokens


def _index_transcript_lines(transcript_body: str) -> list[tuple[int, str]]:
    """Returns (timestamp_seconds, line_text) for every timestamped line in
    the transcript, in original order - the shared source of truth for the
    max-timestamp check, the "is this a real line" check, and source_anchor
    window matching below."""

    lines = []
    for match in _TIMESTAMP_LINE_RE.finditer(transcript_body):
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        line_end = transcript_body.find("\n", match.end())
        if line_end == -1:
            line_end = len(transcript_body)
        lines.append((hours * 3600 + minutes * 60 + seconds, transcript_body[match.end() : line_end]))
    return lines


def _max_transcript_seconds(transcript_body: str) -> int:
    """Highest timestamp actually present in the transcript body shown to
    the model - used to size the point budget (_max_points_for_duration())."""

    return max((seconds for seconds, _ in _index_transcript_lines(transcript_body)), default=0)


def _parse_source_timestamp(raw: str) -> int | None:
    """Parses a model-supplied source_timestamp value (expected verbatim
    from a transcript line's own bracket, e.g. "[14:32]") into seconds.
    Returns None if it doesn't match the expected [H:MM:SS]/[MM:SS] shape
    at all - a distinct failure from "parses fine but isn't a real
    transcript line", which _resolve_points() checks separately."""

    match = _SOURCE_TIMESTAMP_RE.match(raw.strip())
    if not match:
        return None
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _resolve_points(points: list[SummaryPoint], transcript_body: str) -> list[ResolvedPoint]:
    """Verifies each point's cited evidence against the transcript and
    computes its real timestamp_seconds - the model never computes or
    reports this number itself; it only cites a literal transcript line
    (source_timestamp) and a short excerpt from it (source_anchor). Raises
    ValueError - a content/grounding failure Pydantic's schema check alone
    can't catch, same as the old range check - if:

      - source_timestamp doesn't parse as a "[H:MM:SS]"/"[MM:SS]" value;
      - that exact value isn't one of the transcript's own real line
        timestamps (strict equality, not "in range" - the model must cite
        a real line, not merely land on a plausible-sounding number); or
      - fewer than _ANCHOR_WORD_OVERLAP_THRESHOLD of source_anchor's
        significant (non-stopword) words appear within _ANCHOR_WINDOW_LINES
        lines of that timestamp - a fuzzy check, not exact-substring
        containment, since the model reliably finds the right real line but
        doesn't reliably reproduce raw ASR caption text byte-for-byte even
        when told to; see _ANCHOR_WORD_OVERLAP_THRESHOLD's own comment.

    Points are deliberately *not* required to be in chronological order:
    real videos (livestreams especially) revisit the same topic more than
    once, and a strict ordering requirement rejected genuinely well-formed
    output for that content - see _select_significant_points() for the
    actual bound that matters for long videos (a cap on point count, not
    their order)."""

    indexed = _index_transcript_lines(transcript_body)
    # Transcript timestamps are truncated to whole seconds
    # (youtube.format_timestamp()), so more than one caption line can
    # legitimately share the same rendered [MM:SS] - map each second to
    # *every* matching line index, not just the first, or a valid anchor
    # on a later same-second cue would be rejected just because an
    # earlier cue at that same second happened to come first.
    line_indices_by_second: dict[int, list[int]] = {}
    for i, (seconds, _text) in enumerate(indexed):
        line_indices_by_second.setdefault(seconds, []).append(i)

    resolved: list[ResolvedPoint] = []
    for point in points:
        seconds = _parse_source_timestamp(point.source_timestamp)
        if seconds is None:
            raise ValueError(
                f"source_timestamp {point.source_timestamp!r} is not a valid [H:MM:SS]/[MM:SS] "
                f"value for point {point.main_point!r}."
            )
        line_indices = line_indices_by_second.get(seconds)
        if line_indices is None:
            raise ValueError(
                f"source_timestamp {point.source_timestamp!r} ({seconds}s) is not one of the "
                f"transcript's own line timestamps for point {point.main_point!r}."
            )

        anchor_words = _significant_words(point.source_anchor)
        best_overlap = 0.0
        for line_index in line_indices:
            window_start = max(0, line_index - _ANCHOR_WINDOW_LINES)
            window_end = min(len(indexed), line_index + _ANCHOR_WINDOW_LINES + 1)
            window_text = " ".join(text for _, text in indexed[window_start:window_end])
            window_words = _significant_words(window_text)
            overlap = len(anchor_words & window_words) / len(anchor_words) if anchor_words else 0.0
            best_overlap = max(best_overlap, overlap)
        if best_overlap < _ANCHOR_WORD_OVERLAP_THRESHOLD:
            raise ValueError(
                f"source_anchor {point.source_anchor!r} has only {best_overlap:.0%} word overlap with "
                f"content within {_ANCHOR_WINDOW_LINES} line(s) of {point.source_timestamp!r} "
                f"(need >= {_ANCHOR_WORD_OVERLAP_THRESHOLD:.0%}) for point {point.main_point!r}."
            )

        resolved.append(
            ResolvedPoint(
                importance=point.importance,
                main_point=point.main_point,
                explanation=point.explanation,
                timestamp_seconds=seconds,
            )
        )
    return resolved


def _select_significant_points(points: list[ResolvedPoint], max_points: int) -> tuple[list[ResolvedPoint], bool]:
    """Enforces _max_points_for_duration()'s cap as a hard backstop,
    independent of whether the model already respected it in the prompt:
    if the model still returns more than max_points, keep only the most
    significant ones - "major" points before "minor" ones - rather than
    failing/retrying, since this is a "which points to keep" selection,
    not a correctness problem with any individual point. Returns
    (selected_points, was_truncated); selected_points preserves the
    model's original relative order among whichever points are kept."""

    if len(points) <= max_points:
        return points, False
    majors = [p for p in points if p.importance == "major"]
    minors = [p for p in points if p.importance == "minor"]
    kept_ids = {id(p) for p in (majors + minors)[:max_points]}
    return [p for p in points if id(p) in kept_ids], True


def summarize_transcript(
    transcript_body: str, *, model: str, max_output_tokens: int
) -> tuple[ResolvedSummary, Usage, bool]:
    """Calls Claude to produce a ResolvedSummary for one transcript (the
    model itself produces a ModelSummaryOutput; _resolve_points() verifies
    and converts it - see that function's docstring). Raises
    SummarizationError on any provider failure or invalid/unparseable
    output - never raises a raw SDK exception, so callers don't need to know
    the SDK's exception hierarchy. Returns (output, usage, points_truncated) -
    points_truncated is True if _select_significant_points() had to drop
    points to enforce the length-based cap."""

    max_points = _max_points_for_duration(_max_transcript_seconds(transcript_body))

    try:
        client = anthropic.Anthropic(max_retries=_CLIENT_MAX_RETRIES)
        response = client.messages.parse(
            model=model,
            max_tokens=max_output_tokens,
            system=_build_system_prompt(max_points),
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
    except pydantic.ValidationError as exc:
        # The pinned SDK's messages.parse() validates the model's JSON
        # against our schema *inside* the same call (via a post_parser
        # hook that runs before parse() returns), not afterward - so a
        # malformed/truncated response raises pydantic.ValidationError
        # directly out of client.messages.parse() itself, not any
        # anthropic.* exception type. Uncaught, this would escape
        # summarize_transcript() entirely and abort the whole run instead
        # of being isolated to this one video. Usage can't be recovered
        # here (the exception propagates before we get access to the raw
        # response object) - a known limitation, not a missed case.
        raise SummarizationError(
            f"Model response failed schema validation: {exc}",
            retryable=True,
            possibly_billed=True,
            fallback_eligible=True,
        ) from exc

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
        # Also fallback_eligible: the fallback schema is far simpler (one
        # free-text field, no per-point structure), so it needs much less
        # output budget and is meaningfully more likely to actually parse.
        raise SummarizationError(
            f"Model response did not contain valid structured output (stop_reason={response.stop_reason!r}).",
            retryable=True,
            usage=response_usage,
            fallback_eligible=True,
        )

    try:
        resolved_points = _resolve_points(response.parsed_output.points, transcript_body)
    except ValueError as exc:
        # The core citation/grounding failure this whole mechanism exists
        # for - fallback_eligible=True since a plain-prose fallback has no
        # per-line citation to fail this same way.
        raise SummarizationError(
            f"Model output failed validation: {exc}", retryable=True, usage=response_usage, fallback_eligible=True
        ) from exc

    selected_points, points_truncated = _select_significant_points(resolved_points, max_points)
    output = ResolvedSummary(
        video_type=response.parsed_output.video_type,
        summary=response.parsed_output.summary,
        points=selected_points,
    )

    return output, response_usage, points_truncated


def summarize_fallback(transcript_body: str, *, model: str, max_output_tokens: int) -> tuple[str, Usage]:
    """Last-resort path used only after a video has exhausted its normal
    per-point citation attempts (see summary_store.summarize_eligible()'s
    last-attempt handling) - produces a plain 2-3 paragraph summary
    instead, with no per-line citations to get wrong. Some speakers
    (meandering, conversational, non-linear delivery) make source_timestamp/
    source_anchor grounding hard even when the model clearly understood
    the content; this sidesteps that failure mode entirely rather than
    trying to fix it with more retries of the same approach.

    Same error-handling shape as summarize_transcript() (see there for the
    rationale behind each branch) - deliberately not shared code, since
    this call has no points to resolve and a much simpler failure surface."""

    try:
        client = anthropic.Anthropic(max_retries=_CLIENT_MAX_RETRIES)
        response = client.messages.parse(
            model=model,
            max_tokens=max_output_tokens,
            system=_build_fallback_system_prompt(),
            messages=[{"role": "user", "content": transcript_body}],
            output_format=FallbackSummaryOutput,
        )
    except anthropic.RateLimitError as exc:
        raise SummarizationError(f"Rate limited: {exc}", retryable=True) from exc
    except anthropic.APIConnectionError as exc:
        raise SummarizationError(f"Connection error: {exc}", retryable=True) from exc
    except anthropic.APIStatusError as exc:
        retryable = exc.status_code >= 500 or exc.status_code == 429
        raise SummarizationError(f"API error ({exc.status_code}): {exc.message}", retryable=retryable) from exc
    except anthropic.AnthropicError as exc:
        raise SummarizationError(f"Anthropic SDK error: {exc}", retryable=False) from exc
    except pydantic.ValidationError as exc:
        raise SummarizationError(
            f"Fallback model response failed schema validation: {exc}", retryable=True, possibly_billed=True
        ) from exc

    response_usage = Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)

    if response.stop_reason == "refusal":
        raise SummarizationError(
            "Model declined to produce a fallback summary (safety refusal).", retryable=False, usage=response_usage
        )
    if response.parsed_output is None:
        raise SummarizationError(
            f"Fallback model response did not contain valid structured output "
            f"(stop_reason={response.stop_reason!r}).",
            retryable=True,
            usage=response_usage,
        )

    return response.parsed_output.summary, response_usage
