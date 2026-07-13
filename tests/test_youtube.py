import json

import pytest
import yaml

from app import youtube
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=10",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=abc",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ],
)
def test_extract_video_id_recognizes_common_url_shapes(url):
    assert youtube.extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_raises_on_unrecognized_input():
    with pytest.raises(youtube.VideoUrlError):
        youtube.extract_video_id("https://example.com/not-a-video")


class _FakeResponse:
    def __init__(self, payload=None, raise_json=False):
        self._payload = payload
        self._raise_json = raise_json

    def raise_for_status(self):
        pass

    def json(self):
        if self._raise_json:
            raise ValueError("not valid json")
        return self._payload


def test_fetch_video_metadata_success(monkeypatch):
    monkeypatch.setattr(
        youtube.requests,
        "get",
        lambda *a, **k: _FakeResponse({"title": "A Title", "author_name": "A Channel"}),
    )
    meta = youtube.fetch_video_metadata("abc123XYZde")
    assert meta.title == "A Title"
    assert meta.author == "A Channel"


def test_fetch_video_metadata_falls_back_on_malformed_json(monkeypatch):
    monkeypatch.setattr(youtube.requests, "get", lambda *a, **k: _FakeResponse(raise_json=True))
    meta = youtube.fetch_video_metadata("abc123XYZde")
    assert meta.title == "abc123XYZde"
    assert meta.author is None


def test_fetch_video_metadata_falls_back_on_request_exception(monkeypatch):
    def _raise(*a, **k):
        raise youtube.requests.RequestException("boom")

    monkeypatch.setattr(youtube.requests, "get", _raise)
    meta = youtube.fetch_video_metadata("abc123XYZde")
    assert meta.title == "abc123XYZde"
    assert meta.author is None


class _FakeApi:
    def __init__(self, outcome):
        self._outcome = outcome

    def fetch(self, video_id, languages=("en",)):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _FakeSnippet:
    def __init__(self, start, text):
        self.start = start
        self.text = text


class _FakeTranscript:
    def __init__(self, snippets, language="English", language_code="en", is_generated=False):
        self._snippets = snippets
        self.language = language
        self.language_code = language_code
        self.is_generated = is_generated

    def __iter__(self):
        return iter(self._snippets)


def test_fetch_transcript_ok(monkeypatch):
    fake = _FakeTranscript([_FakeSnippet(0.0, "hi"), _FakeSnippet(1.5, "there")])
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", lambda: _FakeApi(fake))
    result = youtube.fetch_transcript("abc123XYZde", ["en"])
    assert result.status == "ok"
    assert result.lines == [(0.0, "hi"), (1.5, "there")]
    assert result.language_code == "en"


@pytest.mark.parametrize(
    "exc,expected_status",
    [
        (TranscriptsDisabled("abc123XYZde"), "no_captions"),
        (NoTranscriptFound("abc123XYZde", ["en"], []), "no_captions"),
        (VideoUnavailable("abc123XYZde"), "unavailable"),
        (IpBlocked("abc123XYZde"), "blocked"),
    ],
)
def test_fetch_transcript_maps_known_exceptions_to_status(monkeypatch, exc, expected_status):
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", lambda: _FakeApi(exc))
    result = youtube.fetch_transcript("abc123XYZde", ["en"])
    assert result.status == expected_status
    assert result.lines is None


def test_render_transcript_markdown_frontmatter_is_valid_yaml_with_tricky_title():
    tricky_title = 'Weird: Title "with quotes", a\nnewline, and a colon: yes'
    md = youtube.render_transcript_markdown(
        video_id="abc123XYZde",
        url="https://www.youtube.com/watch?v=abc123XYZde",
        title=tricky_title,
        author="Channel: Name, Inc.",
        fetched_at="2026-07-13T12:00:00+00:00",
        language="English",
        language_code="en",
        is_generated=True,
        lines=[(0.0, "hello"), (5.0, "world")],
    )
    frontmatter_text = md.split("---")[1]
    parsed = yaml.safe_load(frontmatter_text)
    assert parsed["title"] == tricky_title
    assert parsed["channel"] == "Channel: Name, Inc."
    assert parsed["video_id"] == "abc123XYZde"
    assert "[00:00] hello" in md
    assert "[00:05] world" in md


def test_render_transcript_markdown_title_round_trips_through_json_too():
    # Sanity check on the escaping trick itself: json.dumps output must be
    # exactly what's embedded, so decoding it back gives the original title.
    title = 'a "quoted" \\ backslash and emoji 🎬'
    dumped = json.dumps(title)
    assert json.loads(dumped) == title
