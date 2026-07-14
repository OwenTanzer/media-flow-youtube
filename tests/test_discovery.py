import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pytest
import requests

from app import discovery
from app.channel_store import Channel

FIXED_NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns="http://www.w3.org/2005/Atom">
  <id>yt:channel:UC_sample</id>
  <entry>
    <id>yt:video:videoAAAAAAA</id>
    <yt:videoId>videoAAAAAAA</yt:videoId>
    <yt:channelId>UC_sample</yt:channelId>
    <title>First video</title>
    <published>2026-07-01T12:00:00+00:00</published>
  </entry>
  <entry>
    <id>yt:video:videoBBBBBBB</id>
    <yt:videoId>videoBBBBBBB</yt:videoId>
    <yt:channelId>UC_sample</yt:channelId>
    <title>Second video</title>
    <published>2026-07-02T12:00:00+00:00</published>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


def test_fetch_channel_feed_parses_entries(monkeypatch):
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: _FakeResponse(SAMPLE_FEED.encode("utf-8")))

    videos = discovery.fetch_channel_feed("UC_sample")

    assert [v.video_id for v in videos] == ["videoAAAAAAA", "videoBBBBBBB"]
    assert videos[0].channel_id == "UC_sample"
    assert videos[0].published == "2026-07-01T12:00:00+00:00"


def test_fetch_channel_feed_raises_on_request_failure(monkeypatch):
    def _raise(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(discovery.requests, "get", _raise)

    with pytest.raises(requests.RequestException):
        discovery.fetch_channel_feed("UC_sample")


def test_fetch_channel_feed_raises_on_malformed_xml(monkeypatch):
    monkeypatch.setattr(discovery.requests, "get", lambda *a, **k: _FakeResponse(b"not xml"))

    with pytest.raises(ET.ParseError):
        discovery.fetch_channel_feed("UC_sample")


def _stub_stores(monkeypatch, *, channels, index=None, queue=None):
    monkeypatch.setattr(discovery.channel_store, "read_channels", lambda folder_id: channels)
    monkeypatch.setattr(discovery.drive, "read_index", lambda folder_id: index or {})
    monkeypatch.setattr(discovery.queue_store, "read_queue", lambda folder_id: queue or [])
    monkeypatch.setattr(discovery, "_utcnow", lambda: FIXED_NOW)
    written = {}
    monkeypatch.setattr(
        discovery.queue_store, "write_queue", lambda folder_id, entries: written.setdefault("entries", entries)
    )
    return written


def test_discover_and_enqueue_queues_newly_discovered_video(monkeypatch):
    channel = Channel("UC_a", "Channel A", enabled=True)
    written = _stub_stores(monkeypatch, channels=[channel])
    monkeypatch.setattr(
        discovery,
        "fetch_channel_feed",
        lambda channel_id: [discovery.DiscoveredVideo("newvideo11", "UC_a", "2026-07-01T00:00:00+00:00")],
    )

    report = discovery.discover_and_enqueue("folder-id")

    assert written["entries"] == [
        {
            "url": "https://www.youtube.com/watch?v=newvideo11",
            "first_seen_at": FIXED_NOW.isoformat(),
            "channel_id": "UC_a",
            "published_at": "2026-07-01T00:00:00+00:00",
        }
    ]
    assert report.newly_queued == 1
    assert report.discovered_total == 1
    assert report.duplicates_skipped == 0
    assert report.feed_failures == []


def test_discover_and_enqueue_omits_published_at_when_feed_has_none(monkeypatch):
    """A video whose feed entry has no <published> element (DiscoveredVideo.published
    is None) shouldn't get a fabricated published_at key."""
    channel = Channel("UC_a", "Channel A", enabled=True)
    written = _stub_stores(monkeypatch, channels=[channel])
    monkeypatch.setattr(
        discovery,
        "fetch_channel_feed",
        lambda channel_id: [discovery.DiscoveredVideo("newvideo11", "UC_a", None)],
    )

    discovery.discover_and_enqueue("folder-id")

    assert "published_at" not in written["entries"][0]
    assert written["entries"][0]["channel_id"] == "UC_a"


def test_discover_and_enqueue_skips_already_indexed_video(monkeypatch):
    channel = Channel("UC_a", "Channel A", enabled=True)
    written = _stub_stores(
        monkeypatch,
        channels=[channel],
        index={"knownvideo1": {"status": "ok"}},
    )
    monkeypatch.setattr(
        discovery,
        "fetch_channel_feed",
        lambda channel_id: [discovery.DiscoveredVideo("knownvideo1", "UC_a", None)],
    )

    report = discovery.discover_and_enqueue("folder-id")

    assert "entries" not in written  # write_queue never called, nothing new to add
    assert report.newly_queued == 0
    assert report.duplicates_skipped == 1


def test_discover_and_enqueue_isolates_one_channels_feed_failure(monkeypatch):
    good = Channel("UC_good", "Good Channel", enabled=True)
    bad = Channel("UC_bad", "Bad Channel", enabled=True)
    written = _stub_stores(monkeypatch, channels=[bad, good])

    def _fetch(channel_id):
        if channel_id == "UC_bad":
            raise requests.RequestException("feed unavailable")
        return [discovery.DiscoveredVideo("goodvideo1", "UC_good", None)]

    monkeypatch.setattr(discovery, "fetch_channel_feed", _fetch)

    report = discovery.discover_and_enqueue("folder-id")

    assert written["entries"] == [
        {
            "url": "https://www.youtube.com/watch?v=goodvideo1",
            "first_seen_at": FIXED_NOW.isoformat(),
            "channel_id": "UC_good",
        }
    ]
    assert report.newly_queued == 1
    assert len(report.feed_failures) == 1
    assert report.feed_failures[0][0] == "UC_bad"


def test_discover_and_enqueue_applies_channel_language_override(monkeypatch):
    channel = Channel("UC_a", "Channel A", enabled=True, languages=["es", "pt"])
    written = _stub_stores(monkeypatch, channels=[channel])
    monkeypatch.setattr(
        discovery, "fetch_channel_feed", lambda channel_id: [discovery.DiscoveredVideo("newvideo22", "UC_a", None)]
    )

    discovery.discover_and_enqueue("folder-id")

    assert written["entries"] == [
        {
            "url": "https://www.youtube.com/watch?v=newvideo22",
            "first_seen_at": FIXED_NOW.isoformat(),
            "channel_id": "UC_a",
            "languages": ["es", "pt"],
        }
    ]


def test_discover_and_enqueue_skips_disabled_channels(monkeypatch):
    disabled = Channel("UC_off", "Disabled", enabled=False)
    written = _stub_stores(monkeypatch, channels=[disabled])
    called = []
    monkeypatch.setattr(discovery, "fetch_channel_feed", lambda channel_id: called.append(channel_id))

    report = discovery.discover_and_enqueue("folder-id")

    assert not called
    assert report.channels_configured == 1
    assert report.channels_enabled == 0
    assert "entries" not in written


def test_discover_and_enqueue_deduplicates_against_existing_queue(monkeypatch):
    channel = Channel("UC_a", "Channel A", enabled=True)
    # The video is already sitting in queue.json as a plain string.
    written = _stub_stores(
        monkeypatch, channels=[channel], queue=["https://www.youtube.com/watch?v=queuedvidA1"]
    )
    monkeypatch.setattr(
        discovery,
        "fetch_channel_feed",
        lambda channel_id: [discovery.DiscoveredVideo("queuedvidA1", "UC_a", None)],
    )

    report = discovery.discover_and_enqueue("folder-id")

    assert "entries" not in written
    assert report.duplicates_skipped == 1


def test_find_unbackfilled_channels_returns_channels_with_no_known_videos(monkeypatch):
    never_discovered = Channel("UC_new", "Brand New Channel", enabled=True)
    already_discovered = Channel("UC_old", "Established Channel", enabled=True)
    _stub_stores(
        monkeypatch,
        channels=[never_discovered, already_discovered],
        index={"vid1": {"status": "ok", "channel_id": "UC_old"}},
    )

    assert discovery.find_unbackfilled_channels("folder-id") == [never_discovered]


def test_find_unbackfilled_channels_checks_queue_as_well_as_index(monkeypatch):
    """A channel discovered so recently its video hasn't been processed
    into _index.json yet (still sitting in queue.json) must not be
    treated as unbackfilled - that queue entry proves discovery already
    saw it once."""
    channel = Channel("UC_a", "Channel A", enabled=True)
    _stub_stores(
        monkeypatch,
        channels=[channel],
        queue=[{"url": "https://www.youtube.com/watch?v=queuedvidA1", "channel_id": "UC_a"}],
    )

    assert discovery.find_unbackfilled_channels("folder-id") == []


def test_find_unbackfilled_channels_excludes_disabled_channels(monkeypatch):
    disabled = Channel("UC_off", "Disabled", enabled=False)
    _stub_stores(monkeypatch, channels=[disabled])

    assert discovery.find_unbackfilled_channels("folder-id") == []


def test_find_unbackfilled_channels_returns_empty_when_all_channels_seen(monkeypatch):
    channel = Channel("UC_a", "Channel A", enabled=True)
    _stub_stores(monkeypatch, channels=[channel], index={"vid1": {"status": "ok", "channel_id": "UC_a"}})

    assert discovery.find_unbackfilled_channels("folder-id") == []


def test_backfill_new_channels_only_fetches_unbackfilled_channels(monkeypatch):
    never_discovered = Channel("UC_new", "Brand New Channel", enabled=True)
    already_discovered = Channel("UC_old", "Established Channel", enabled=True)
    written = _stub_stores(
        monkeypatch,
        channels=[never_discovered, already_discovered],
        index={"vid1": {"status": "ok", "channel_id": "UC_old"}},
    )
    fetched = []

    def _fetch(channel_id):
        fetched.append(channel_id)
        return [discovery.DiscoveredVideo("newvideo11", "UC_new", "2026-07-01T00:00:00+00:00")]

    monkeypatch.setattr(discovery, "fetch_channel_feed", _fetch)

    report = discovery.backfill_new_channels("folder-id")

    assert fetched == ["UC_new"]
    assert report.channels_configured == 1
    assert report.newly_queued == 1
    assert written["entries"][0]["channel_id"] == "UC_new"


def test_backfill_new_channels_is_a_noop_when_nothing_needs_it(monkeypatch):
    channel = Channel("UC_a", "Channel A", enabled=True)
    written = _stub_stores(monkeypatch, channels=[channel], index={"vid1": {"status": "ok", "channel_id": "UC_a"}})
    called = []
    monkeypatch.setattr(discovery, "fetch_channel_feed", lambda channel_id: called.append(channel_id))

    report = discovery.backfill_new_channels("folder-id")

    assert not called
    assert report.channels_configured == 0
    assert report.newly_queued == 0
    assert "entries" not in written


def test_backfill_new_channels_isolates_one_channels_feed_failure(monkeypatch):
    ok_channel = Channel("UC_ok", "Fine Channel", enabled=True)
    broken_channel = Channel("UC_broken", "Broken Channel", enabled=True)
    written = _stub_stores(monkeypatch, channels=[ok_channel, broken_channel])

    def _fetch(channel_id):
        if channel_id == "UC_broken":
            raise requests.RequestException("network down")
        return [discovery.DiscoveredVideo("newvideo11", "UC_ok", None)]

    monkeypatch.setattr(discovery, "fetch_channel_feed", _fetch)

    report = discovery.backfill_new_channels("folder-id")

    assert report.newly_queued == 1
    assert report.feed_failures == [("UC_broken", "network down")]
    assert written["entries"][0]["channel_id"] == "UC_ok"
