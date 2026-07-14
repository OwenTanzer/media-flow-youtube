import backfill_channel_ids as backfill
from app.channel_store import Channel
from app.discovery import DiscoveredVideo


def test_main_returns_1_when_drive_folder_id_missing(monkeypatch):
    monkeypatch.setattr(backfill.settings, "drive_folder_id", None)
    assert backfill.main() == 1


def test_build_channel_id_map_combines_multiple_channels(monkeypatch):
    channel_a = Channel("UC_a", "Channel A")
    channel_b = Channel("UC_b", "Channel B")
    monkeypatch.setattr(backfill.channel_store, "read_channels", lambda folder_id: [channel_a, channel_b])

    def _fake_feed(channel_id):
        if channel_id == "UC_a":
            return [DiscoveredVideo("vid1", "UC_a", "2026-07-01T00:00:00+00:00")]
        return [DiscoveredVideo("vid2", "UC_b", "2026-07-02T00:00:00+00:00")]

    monkeypatch.setattr(backfill.discovery, "fetch_channel_feed", _fake_feed)

    result = backfill.build_channel_id_map("folder-id")

    assert result == {"vid1": "UC_a", "vid2": "UC_b"}


def test_build_channel_id_map_skips_a_failed_channel_feed(monkeypatch):
    channel_a = Channel("UC_a", "Channel A")
    channel_b = Channel("UC_b", "Channel B")
    monkeypatch.setattr(backfill.channel_store, "read_channels", lambda folder_id: [channel_a, channel_b])

    def _fake_feed(channel_id):
        if channel_id == "UC_a":
            raise RuntimeError("feed down")
        return [DiscoveredVideo("vid2", "UC_b", "2026-07-02T00:00:00+00:00")]

    monkeypatch.setattr(backfill.discovery, "fetch_channel_feed", _fake_feed)

    result = backfill.build_channel_id_map("folder-id")

    assert result == {"vid2": "UC_b"}


def test_build_channel_id_map_includes_videos_with_no_published_date(monkeypatch):
    """Unlike published_at, channel_id doesn't depend on the feed entry
    itself carrying a <published> element - every video in a channel's
    feed is known to belong to that channel regardless."""
    channel_a = Channel("UC_a", "Channel A")
    monkeypatch.setattr(backfill.channel_store, "read_channels", lambda folder_id: [channel_a])
    monkeypatch.setattr(
        backfill.discovery, "fetch_channel_feed", lambda channel_id: [DiscoveredVideo("vid1", "UC_a", None)]
    )

    assert backfill.build_channel_id_map("folder-id") == {"vid1": "UC_a"}


def _stub_drive(monkeypatch, *, index, feed_map, existing_summaries=None):
    monkeypatch.setattr(backfill, "build_channel_id_map", lambda folder_id: feed_map)
    monkeypatch.setattr(backfill.drive, "read_index", lambda folder_id: index)

    # write_index() is now called at most once, with the whole (in-place
    # mutated) index - not once per updated video_id like the old
    # update_index_entry() calls this replaced.
    index_writes = []
    monkeypatch.setattr(backfill.drive, "write_index", lambda folder_id, written: index_writes.append(written))

    existing_summaries = dict(existing_summaries or {})
    monkeypatch.setattr(backfill.summary_store, "read_summary", lambda folder_id, video_id: existing_summaries.get(video_id))
    summary_writes = {}
    monkeypatch.setattr(
        backfill.summary_store, "write_summary", lambda folder_id, video_id, artifact: summary_writes.setdefault(video_id, artifact)
    )

    return index_writes, summary_writes


def test_main_backfills_a_video_found_in_the_feed(monkeypatch):
    index = {"vid1": {"status": "ok", "filename": "A Title [vid1].md", "video_id": "vid1"}}
    index_writes, _ = _stub_drive(monkeypatch, index=index, feed_map={"vid1": "UC_a"})

    exit_code = backfill.main()

    assert exit_code == 0
    assert index_writes[0]["vid1"]["channel_id"] == "UC_a"


def test_main_leaves_a_video_untouched_when_not_found_in_any_feed(monkeypatch):
    index = {"vid1": {"status": "ok", "filename": "A Title [vid1].md", "video_id": "vid1"}}
    index_writes, _ = _stub_drive(monkeypatch, index=index, feed_map={})

    backfill.main()

    assert index_writes == []


def test_main_does_not_overwrite_an_existing_channel_id(monkeypatch):
    index = {
        "vid1": {
            "status": "ok",
            "filename": "A Title [vid1].md",
            "video_id": "vid1",
            "channel_id": "UC_existing",
        }
    }
    index_writes, _ = _stub_drive(monkeypatch, index=index, feed_map={"vid1": "UC_a"})

    backfill.main()

    assert index_writes == []


def test_main_ignores_non_ok_index_entries(monkeypatch):
    index = {"vid1": {"status": "blocked", "filename": "x.md", "video_id": "vid1"}}
    index_writes, _ = _stub_drive(monkeypatch, index=index, feed_map={"vid1": "UC_a"})

    backfill.main()

    assert index_writes == []


def test_main_updates_an_existing_summary_artifact_too(monkeypatch):
    index = {"vid1": {"status": "ok", "filename": "A Title [vid1].md", "video_id": "vid1"}}
    existing_summaries = {"vid1": {"status": "ok", "channel_id": None}}
    _, summary_writes = _stub_drive(
        monkeypatch, index=index, feed_map={"vid1": "UC_a"}, existing_summaries=existing_summaries
    )

    backfill.main()

    assert summary_writes["vid1"]["channel_id"] == "UC_a"


def test_main_does_not_touch_a_summary_artifact_that_already_has_a_channel_id(monkeypatch):
    index = {"vid1": {"status": "ok", "filename": "A Title [vid1].md", "video_id": "vid1"}}
    existing_summaries = {"vid1": {"status": "ok", "channel_id": "UC_existing"}}
    _, summary_writes = _stub_drive(
        monkeypatch, index=index, feed_map={"vid1": "UC_a"}, existing_summaries=existing_summaries
    )

    backfill.main()

    assert summary_writes == {}
