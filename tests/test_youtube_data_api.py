import pytest
import requests

from app import youtube_data_api
from app.discovered_video import DiscoveredVideo


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_resolve_uploads_playlist_id_success(monkeypatch):
    payload = {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "PL_uploads"}}}]}
    monkeypatch.setattr(youtube_data_api.requests, "get", lambda *a, **k: _FakeResponse(payload))

    assert youtube_data_api.resolve_uploads_playlist_id("UC_a", "key") == "PL_uploads"


def test_resolve_uploads_playlist_id_raises_on_empty_items(monkeypatch):
    monkeypatch.setattr(youtube_data_api.requests, "get", lambda *a, **k: _FakeResponse({"items": []}))

    with pytest.raises(youtube_data_api.YouTubeDataApiError):
        youtube_data_api.resolve_uploads_playlist_id("UC_missing", "key")


def test_resolve_uploads_playlist_id_raises_on_malformed_item(monkeypatch):
    monkeypatch.setattr(
        youtube_data_api.requests, "get", lambda *a, **k: _FakeResponse({"items": [{"contentDetails": {}}]})
    )

    with pytest.raises(youtube_data_api.YouTubeDataApiError):
        youtube_data_api.resolve_uploads_playlist_id("UC_a", "key")


def test_resolve_uploads_playlist_id_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(youtube_data_api.requests, "get", lambda *a, **k: _FakeResponse({}, status_code=403))

    with pytest.raises(youtube_data_api.YouTubeDataApiError):
        youtube_data_api.resolve_uploads_playlist_id("UC_a", "bad-key")


def test_resolve_uploads_playlist_id_raises_on_request_exception(monkeypatch):
    def _raise(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(youtube_data_api.requests, "get", _raise)

    with pytest.raises(youtube_data_api.YouTubeDataApiError):
        youtube_data_api.resolve_uploads_playlist_id("UC_a", "key")


def _playlist_item(video_id: str, published: str | None = None) -> dict:
    content_details = {"videoId": video_id}
    if published:
        content_details["videoPublishedAt"] = published
    return {"contentDetails": content_details, "snippet": {}}


def test_fetch_uploads_playlist_videos_single_page(monkeypatch):
    payload = {"items": [_playlist_item("videoAAAAAAA", "2026-07-01T00:00:00Z"), _playlist_item("videoBBBBBBB")]}
    monkeypatch.setattr(youtube_data_api.requests, "get", lambda *a, **k: _FakeResponse(payload))

    videos = youtube_data_api.fetch_uploads_playlist_videos("PL_x", "key", known_ids=set(), channel_id="UC_a")

    assert videos == [
        DiscoveredVideo("videoAAAAAAA", "UC_a", "2026-07-01T00:00:00Z"),
        DiscoveredVideo("videoBBBBBBB", "UC_a", None),
    ]


def test_fetch_uploads_playlist_videos_stops_at_first_known_id(monkeypatch):
    payload = {"items": [_playlist_item("newvideo11"), _playlist_item("knownvideo1"), _playlist_item("older1111")]}
    monkeypatch.setattr(youtube_data_api.requests, "get", lambda *a, **k: _FakeResponse(payload))

    videos = youtube_data_api.fetch_uploads_playlist_videos(
        "PL_x", "key", known_ids={"knownvideo1"}, channel_id="UC_a"
    )

    assert [v.video_id for v in videos] == ["newvideo11"]


def test_fetch_uploads_playlist_videos_paginates_when_no_known_id_seen(monkeypatch):
    pages = [
        {"items": [_playlist_item("videoPage1A")], "nextPageToken": "page2"},
        {"items": [_playlist_item("videoPage2A")]},
    ]
    calls = []

    def _get(url, params, timeout):
        calls.append(params.get("pageToken"))
        return _FakeResponse(pages[len(calls) - 1])

    monkeypatch.setattr(youtube_data_api.requests, "get", lambda *a, **k: _get(*a, **k))

    videos = youtube_data_api.fetch_uploads_playlist_videos("PL_x", "key", known_ids=set(), channel_id="UC_a")

    assert [v.video_id for v in videos] == ["videoPage1A", "videoPage2A"]
    assert calls == [None, "page2"]


def test_fetch_uploads_playlist_videos_empty_playlist(monkeypatch):
    monkeypatch.setattr(youtube_data_api.requests, "get", lambda *a, **k: _FakeResponse({"items": []}))

    videos = youtube_data_api.fetch_uploads_playlist_videos("PL_x", "key", known_ids=set(), channel_id="UC_a")

    assert videos == []


def test_fetch_uploads_playlist_videos_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(youtube_data_api.requests, "get", lambda *a, **k: _FakeResponse({}, status_code=500))

    with pytest.raises(youtube_data_api.YouTubeDataApiError):
        youtube_data_api.fetch_uploads_playlist_videos("PL_x", "key", known_ids=set(), channel_id="UC_a")
