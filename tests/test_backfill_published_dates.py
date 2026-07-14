import backfill_published_dates as backfill
from app.channel_store import Channel
from app.discovery import DiscoveredVideo


def test_main_returns_1_when_drive_folder_id_missing(monkeypatch):
    monkeypatch.setattr(backfill.settings, "drive_folder_id", None)
    assert backfill.main() == 1


def test_build_published_at_map_combines_multiple_channels(monkeypatch):
    channel_a = Channel("UC_a", "Channel A")
    channel_b = Channel("UC_b", "Channel B")
    monkeypatch.setattr(backfill.channel_store, "read_channels", lambda folder_id: [channel_a, channel_b])

    def _fake_feed(channel_id):
        if channel_id == "UC_a":
            return [DiscoveredVideo("vid1", "UC_a", "2026-07-01T00:00:00+00:00")]
        return [DiscoveredVideo("vid2", "UC_b", "2026-07-02T00:00:00+00:00")]

    monkeypatch.setattr(backfill.discovery, "fetch_channel_feed", _fake_feed)

    result = backfill.build_published_at_map("folder-id")

    assert result == {"vid1": "2026-07-01T00:00:00+00:00", "vid2": "2026-07-02T00:00:00+00:00"}


def test_build_published_at_map_skips_a_failed_channel_feed(monkeypatch):
    channel_a = Channel("UC_a", "Channel A")
    channel_b = Channel("UC_b", "Channel B")
    monkeypatch.setattr(backfill.channel_store, "read_channels", lambda folder_id: [channel_a, channel_b])

    def _fake_feed(channel_id):
        if channel_id == "UC_a":
            raise RuntimeError("feed down")
        return [DiscoveredVideo("vid2", "UC_b", "2026-07-02T00:00:00+00:00")]

    monkeypatch.setattr(backfill.discovery, "fetch_channel_feed", _fake_feed)

    result = backfill.build_published_at_map("folder-id")

    assert result == {"vid2": "2026-07-02T00:00:00+00:00"}


def test_build_published_at_map_ignores_videos_with_no_published_date(monkeypatch):
    channel_a = Channel("UC_a", "Channel A")
    monkeypatch.setattr(backfill.channel_store, "read_channels", lambda folder_id: [channel_a])
    monkeypatch.setattr(
        backfill.discovery, "fetch_channel_feed", lambda channel_id: [DiscoveredVideo("vid1", "UC_a", None)]
    )

    assert backfill.build_published_at_map("folder-id") == {}


SAMPLE_MARKDOWN = """---
video_id: vid1
title: "A Title"
url: https://www.youtube.com/watch?v=vid1
channel: "A Channel"
fetched_at: 2026-07-13T10:00:00+00:00
language: "English (en)"
auto_generated: false
---

[00:00] hello
"""


def _stub_drive(monkeypatch, *, index, feed_map, transcripts=None, existing_summaries=None):
    monkeypatch.setattr(backfill, "build_published_at_map", lambda folder_id: feed_map)
    monkeypatch.setattr(backfill.drive, "read_index", lambda folder_id: index)

    index_writes = {}
    monkeypatch.setattr(
        backfill.drive, "update_index_entry", lambda folder_id, video_id, entry: index_writes.setdefault(video_id, entry)
    )

    transcript_writes = {}
    transcripts = transcripts or {}

    def _download_text(folder_id, filename):
        return transcripts.get(filename)

    def _upload_text_file(folder_id, filename, content, **kwargs):
        transcript_writes[filename] = content

    monkeypatch.setattr(backfill.drive, "download_text", _download_text)
    monkeypatch.setattr(backfill.drive, "upload_text_file", _upload_text_file)

    existing_summaries = dict(existing_summaries or {})
    monkeypatch.setattr(backfill.summary_store, "read_summary", lambda folder_id, video_id: existing_summaries.get(video_id))
    summary_writes = {}
    monkeypatch.setattr(
        backfill.summary_store, "write_summary", lambda folder_id, video_id, artifact: summary_writes.setdefault(video_id, artifact)
    )

    return index_writes, transcript_writes, summary_writes


def test_main_backfills_a_video_found_in_the_feed(monkeypatch):
    index = {"vid1": {"status": "ok", "filename": "A Title [vid1].md", "video_id": "vid1"}}
    index_writes, transcript_writes, _ = _stub_drive(
        monkeypatch,
        index=index,
        feed_map={"vid1": "2026-07-01T00:00:00+00:00"},
        transcripts={"A Title [vid1].md": SAMPLE_MARKDOWN},
    )

    exit_code = backfill.main()

    assert exit_code == 0
    assert index_writes["vid1"]["published_at"] == "2026-07-01T00:00:00+00:00"
    assert "published_at: 2026-07-01T00:00:00+00:00" in transcript_writes["A Title [vid1].md"]


def test_main_leaves_a_video_untouched_when_not_found_in_any_feed(monkeypatch):
    index = {"vid1": {"status": "ok", "filename": "A Title [vid1].md", "video_id": "vid1"}}
    index_writes, transcript_writes, _ = _stub_drive(monkeypatch, index=index, feed_map={})

    backfill.main()

    assert index_writes == {}
    assert transcript_writes == {}


def test_main_does_not_overwrite_an_existing_published_at(monkeypatch):
    index = {
        "vid1": {
            "status": "ok",
            "filename": "A Title [vid1].md",
            "video_id": "vid1",
            "published_at": "2026-06-01T00:00:00+00:00",
        }
    }
    index_writes, _, _ = _stub_drive(monkeypatch, index=index, feed_map={"vid1": "2026-07-01T00:00:00+00:00"})

    backfill.main()

    assert index_writes == {}


def test_main_ignores_non_ok_index_entries(monkeypatch):
    index = {"vid1": {"status": "blocked", "filename": "x.md", "video_id": "vid1"}}
    index_writes, _, _ = _stub_drive(monkeypatch, index=index, feed_map={"vid1": "2026-07-01T00:00:00+00:00"})

    backfill.main()

    assert index_writes == {}


def test_main_updates_an_existing_summary_artifact_too(monkeypatch):
    index = {"vid1": {"status": "ok", "filename": "A Title [vid1].md", "video_id": "vid1"}}
    existing_summaries = {"vid1": {"status": "ok", "video_published_at": None}}
    _, _, summary_writes = _stub_drive(
        monkeypatch,
        index=index,
        feed_map={"vid1": "2026-07-01T00:00:00+00:00"},
        transcripts={"A Title [vid1].md": SAMPLE_MARKDOWN},
        existing_summaries=existing_summaries,
    )

    backfill.main()

    assert summary_writes["vid1"]["video_published_at"] == "2026-07-01T00:00:00+00:00"


def test_main_does_not_touch_a_summary_artifact_that_already_has_a_published_at(monkeypatch):
    index = {"vid1": {"status": "ok", "filename": "A Title [vid1].md", "video_id": "vid1"}}
    existing_summaries = {"vid1": {"status": "ok", "video_published_at": "2026-06-01T00:00:00+00:00"}}
    _, _, summary_writes = _stub_drive(
        monkeypatch,
        index=index,
        feed_map={"vid1": "2026-07-01T00:00:00+00:00"},
        transcripts={"A Title [vid1].md": SAMPLE_MARKDOWN},
        existing_summaries=existing_summaries,
    )

    backfill.main()

    assert summary_writes == {}
