import json
from datetime import datetime, timezone

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


def test_fetch_video_metadata_falls_back_on_request_exception(monkeypatch, caplog):
    def _raise(*a, **k):
        raise youtube.requests.RequestException("boom")

    monkeypatch.setattr(youtube.requests, "get", _raise)
    with caplog.at_level("WARNING", logger="media_flow.youtube"):
        meta = youtube.fetch_video_metadata("abc123XYZde")
    assert meta.title == "abc123XYZde"
    assert meta.author is None
    assert "abc123XYZde" in caplog.text


def test_fetch_video_metadata_retries_once_before_succeeding(monkeypatch):
    calls = []

    def _get(*a, **k):
        calls.append(None)
        if len(calls) == 1:
            raise youtube.requests.RequestException("transient proxy failure")
        return _FakeResponse({"title": "A Title", "author_name": "A Channel"})

    monkeypatch.setattr(youtube.requests, "get", _get)
    meta = youtube.fetch_video_metadata("abc123XYZde")
    assert meta.title == "A Title"
    assert len(calls) == 2


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
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", lambda **kwargs: _FakeApi(fake))
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
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", lambda **kwargs: _FakeApi(exc))
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


def _render_kwargs(**overrides):
    kwargs = dict(
        video_id="abc123XYZde",
        url="https://www.youtube.com/watch?v=abc123XYZde",
        title="A Title",
        author="A Channel",
        fetched_at="2026-07-13T12:00:00+00:00",
        language="English",
        language_code="en",
        is_generated=True,
        lines=[(0.0, "hello")],
    )
    kwargs.update(overrides)
    return kwargs


def test_render_transcript_markdown_includes_published_at_when_given():
    md = youtube.render_transcript_markdown(**_render_kwargs(published_at="2026-07-01T00:00:00+00:00"))
    assert "published_at: 2026-07-01T00:00:00+00:00" in md
    frontmatter_text = md.split("---")[1]
    parsed = yaml.safe_load(frontmatter_text)
    # YAML auto-parses an unquoted ISO8601-looking scalar into a real
    # datetime (same as fetched_at already does) - still valid frontmatter,
    # just not a string once loaded.
    assert parsed["published_at"] == datetime(2026, 7, 1, tzinfo=timezone.utc)


def test_render_transcript_markdown_omits_published_at_when_not_given():
    """Manually-queued/direct-URL videos have no known publish date (only
    discover_and_process.py's RSS feed reader sees one) - the key should be
    entirely absent rather than a fabricated null."""
    md = youtube.render_transcript_markdown(**_render_kwargs())
    frontmatter_text = md.split("---")[1]
    parsed = yaml.safe_load(frontmatter_text)
    assert "published_at" not in parsed


def _clear_proxy_settings(monkeypatch):
    for attr, value in (
        ("youtube_proxy_type", None),
        ("webshare_proxy_username", None),
        ("webshare_proxy_password", None),
        ("webshare_proxy_locations", []),
        ("youtube_proxy_http_url", None),
        ("youtube_proxy_https_url", None),
    ):
        monkeypatch.setattr(youtube.settings, attr, value)


def test_build_proxy_config_defaults_to_none(monkeypatch):
    _clear_proxy_settings(monkeypatch)
    assert youtube.build_proxy_config() is None


def test_build_proxy_config_none_is_explicit_no_op(monkeypatch):
    _clear_proxy_settings(monkeypatch)
    monkeypatch.setattr(youtube.settings, "youtube_proxy_type", "none")
    assert youtube.build_proxy_config() is None


def test_build_proxy_config_webshare(monkeypatch):
    _clear_proxy_settings(monkeypatch)
    monkeypatch.setattr(youtube.settings, "youtube_proxy_type", "webshare")
    monkeypatch.setattr(youtube.settings, "webshare_proxy_username", "wsuser")
    monkeypatch.setattr(youtube.settings, "webshare_proxy_password", "wspass")

    config = youtube.build_proxy_config()

    assert isinstance(config, youtube.WebshareProxyConfig)
    assert config.proxy_username == "wsuser"
    assert config.proxy_password == "wspass"
    # fetch_transcript() is the single retry authority now (a fresh client
    # instance per attempt) - the library's own internal same-session
    # retries-when-blocked must be disabled, or TRANSCRIPT_FETCH_MAX_ATTEMPTS
    # attempts could each fan out into up to 10 more requests internally.
    assert config.retries_when_blocked == 0


def test_build_proxy_config_webshare_missing_credentials_raises(monkeypatch):
    _clear_proxy_settings(monkeypatch)
    monkeypatch.setattr(youtube.settings, "youtube_proxy_type", "webshare")

    with pytest.raises(youtube.ConfigError, match="WEBSHARE_PROXY_USERNAME"):
        youtube.build_proxy_config()


def test_build_proxy_config_generic(monkeypatch):
    _clear_proxy_settings(monkeypatch)
    monkeypatch.setattr(youtube.settings, "youtube_proxy_type", "generic")
    monkeypatch.setattr(youtube.settings, "youtube_proxy_http_url", "http://user:pass@proxy.example:8080")

    config = youtube.build_proxy_config()

    assert isinstance(config, youtube.GenericProxyConfig)
    assert config.to_requests_dict() == {
        "http": "http://user:pass@proxy.example:8080",
        "https": "http://user:pass@proxy.example:8080",
    }


def test_build_proxy_config_generic_missing_urls_raises(monkeypatch):
    _clear_proxy_settings(monkeypatch)
    monkeypatch.setattr(youtube.settings, "youtube_proxy_type", "generic")

    with pytest.raises(youtube.ConfigError, match="YOUTUBE_PROXY_HTTP_URL"):
        youtube.build_proxy_config()


def test_build_proxy_config_unknown_type_raises(monkeypatch):
    _clear_proxy_settings(monkeypatch)
    monkeypatch.setattr(youtube.settings, "youtube_proxy_type", "socks5-magic")

    with pytest.raises(youtube.ConfigError, match="socks5-magic"):
        youtube.build_proxy_config()


def test_fetch_transcript_retries_transient_failure_before_succeeding(monkeypatch):
    fake = _FakeTranscript([_FakeSnippet(0.0, "hi")])
    outcomes = [youtube.requests.RequestException("blocked draw"), fake]
    calls = []

    def fake_api(**kwargs):
        calls.append(None)
        return _FakeApi(outcomes[len(calls) - 1])

    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)
    result = youtube.fetch_transcript("abc123XYZde", ["en"])

    assert result.status == "ok"
    assert len(calls) == 2


def test_fetch_transcript_gives_up_after_max_attempts_still_blocked(monkeypatch):
    """An exhausted RequestBlocked/IpBlocked must stay classified as
    "blocked", not folded into a generic "error" - accurate status is what
    lets us tell whether the new retry strategy is actually fixing YouTube
    429s versus just exchanging them for proxy instability."""
    monkeypatch.setattr(youtube.settings, "transcript_fetch_max_attempts", 3)
    calls = []

    def fake_api(**kwargs):
        calls.append(None)
        return _FakeApi(IpBlocked("abc123XYZde"))

    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)
    result = youtube.fetch_transcript("abc123XYZde", ["en"])

    assert result.status == "blocked"
    assert len(calls) == 3
    assert "3 attempts" in result.message


def test_fetch_transcript_gives_up_after_max_attempts_still_erroring(monkeypatch):
    """A transport-level failure (DNS/TLS/dropped connection/proxy error,
    not an explicit YouTube block) must be classified as "error", distinct
    from "blocked", even after exhausting all retry attempts."""
    monkeypatch.setattr(youtube.settings, "transcript_fetch_max_attempts", 3)
    calls = []

    def fake_api(**kwargs):
        calls.append(None)
        return _FakeApi(youtube.requests.exceptions.ConnectionError("connection reset"))

    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)
    result = youtube.fetch_transcript("abc123XYZde", ["en"])

    assert result.status == "error"
    assert len(calls) == 3
    assert "3 attempts" in result.message


def test_fetch_transcript_each_retry_is_a_fresh_client_instance(monkeypatch):
    """Regression test for the actual fix: a blocked draw must be retried
    with a brand-new YouTubeTranscriptApi instance (a fresh connection, and
    thus another independent draw from the rotating proxy pool) rather than
    reusing the same client/session, which is what left the library's own
    internal same-session retries unable to reliably rotate off a blocked IP."""
    fake = _FakeTranscript([_FakeSnippet(0.0, "hi")])
    outcomes = [youtube.requests.RequestException("blocked draw"), fake]
    instances = []

    def fake_api(**kwargs):
        api = _FakeApi(outcomes[len(instances)])
        instances.append(api)
        return api

    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)
    youtube.fetch_transcript("abc123XYZde", ["en"])

    assert len(instances) == 2
    assert instances[0] is not instances[1]


def test_fetch_transcript_passes_proxy_config_through(monkeypatch):
    _clear_proxy_settings(monkeypatch)
    monkeypatch.setattr(youtube.settings, "youtube_proxy_type", "generic")
    monkeypatch.setattr(youtube.settings, "youtube_proxy_http_url", "http://proxy.example:8080")

    fake = _FakeTranscript([_FakeSnippet(0.0, "hi")])
    seen_kwargs = {}

    def fake_api(**kwargs):
        seen_kwargs.update(kwargs)
        return _FakeApi(fake)

    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", fake_api)
    youtube.fetch_transcript("abc123XYZde", ["en"])

    assert isinstance(seen_kwargs["proxy_config"], youtube.GenericProxyConfig)
