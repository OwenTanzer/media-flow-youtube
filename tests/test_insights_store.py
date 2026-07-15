from datetime import datetime, timezone

import pytest

from app import insights_store
from app.channel_store import Channel

FINANCE_CHANNEL = Channel("UC_finance", "Finance Channel", group="Finance")
GOOGLE_CHANNEL = Channel("UC_google", "Google for Developers", group="Google")
UNGROUPED_CHANNEL = Channel("UC_ungrouped", "Ungrouped Channel")  # group=None -> Finance fallback


def _artifact(**overrides) -> dict:
    base = {
        "video_id": "vid1",
        "title": "A Title",
        "author": "A Channel",
        "url": "https://www.youtube.com/watch?v=vid1",
        "channel_id": "UC_finance",
        "video_type": "Analytic Overview",
        "video_published_at": "2026-07-10T14:00:00+00:00",
        "generated_at": "2026-07-11T00:00:00+00:00",
        "summary": "A short summary.",
        "points": [
            {
                "importance": "major",
                "main_point": "The main point.",
                "explanation": "Because reasons.",
                "timestamp_seconds": 12,
                "timestamp": "00:12",
            }
        ],
        "status": "ok",
        "source_drive_file_id": "drive-file-1",
    }
    base.update(overrides)
    return base


def _stub(monkeypatch, *, channels, index, summaries=None, ledger=None):
    monkeypatch.setattr(insights_store.channel_store, "read_channels", lambda folder_id: channels)
    monkeypatch.setattr(insights_store.drive, "read_index", lambda folder_id: index)
    summaries = summaries or {}
    monkeypatch.setattr(
        insights_store.summary_store,
        "read_summaries_bulk",
        lambda folder_id, video_ids: {vid: summaries[vid] for vid in video_ids if vid in summaries},
    )
    monkeypatch.setattr(insights_store.usage_ledger, "read_ledger", lambda folder_id: ledger or [])


def test_load_snapshot_happy_path_resolves_group_and_channel_name(monkeypatch):
    _stub(
        monkeypatch,
        channels=[FINANCE_CHANNEL, GOOGLE_CHANNEL],
        index={"vid1": {"status": "ok"}},
        summaries={"vid1": _artifact()},
    )

    snapshot = insights_store.load_snapshot("folder-id")

    assert len(snapshot.videos) == 1
    video = snapshot.videos[0]
    assert video.video_id == "vid1"
    assert video.channel_id == "UC_finance"
    assert video.channel_name == "Finance Channel"
    assert video.group == "Finance"
    assert video.video_published_at == datetime(2026, 7, 10, 14, 0, 0, tzinfo=timezone.utc)
    assert video.points[0].main_point == "The main point."
    assert snapshot.channels == [FINANCE_CHANNEL, GOOGLE_CHANNEL]
    assert snapshot.pending_count == 0
    assert snapshot.load_errors == []


def test_load_snapshot_resolves_google_group(monkeypatch):
    _stub(
        monkeypatch,
        channels=[GOOGLE_CHANNEL],
        index={"vid1": {"status": "ok"}},
        summaries={"vid1": _artifact(channel_id="UC_google")},
    )

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.videos[0].group == "Google"


def test_load_snapshot_channel_with_no_explicit_group_falls_back_to_finance(monkeypatch):
    _stub(
        monkeypatch,
        channels=[UNGROUPED_CHANNEL],
        index={"vid1": {"status": "ok"}},
        summaries={"vid1": _artifact(channel_id="UC_ungrouped")},
    )

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.videos[0].group == insights_store.DEFAULT_GROUP


def test_load_snapshot_unknown_channel_id_falls_back_to_finance_and_no_channel_name(monkeypatch):
    _stub(
        monkeypatch,
        channels=[GOOGLE_CHANNEL],
        index={"vid1": {"status": "ok"}},
        summaries={"vid1": _artifact(channel_id="UC_no_longer_configured")},
    )

    snapshot = insights_store.load_snapshot("folder-id")

    video = snapshot.videos[0]
    assert video.group == insights_store.DEFAULT_GROUP
    assert video.channel_name is None
    assert video.channel_id == "UC_no_longer_configured"


def test_load_snapshot_null_channel_id_falls_back_to_finance(monkeypatch):
    _stub(
        monkeypatch,
        channels=[GOOGLE_CHANNEL],
        index={"vid1": {"status": "ok"}},
        summaries={"vid1": _artifact(channel_id=None)},
    )

    snapshot = insights_store.load_snapshot("folder-id")

    video = snapshot.videos[0]
    assert video.group == insights_store.DEFAULT_GROUP
    assert video.channel_id is None


def test_load_snapshot_missing_channels_json_yields_empty_list_not_a_crash(monkeypatch):
    monkeypatch.setattr(
        insights_store.channel_store,
        "read_channels",
        lambda folder_id: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(insights_store.drive, "read_index", lambda folder_id: {})
    monkeypatch.setattr(insights_store.usage_ledger, "read_ledger", lambda folder_id: [])

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.channels == []
    assert snapshot.load_errors == ["Channel registry (channels.json) could not be read."]


def test_load_snapshot_propagates_a_genuine_summary_read_failure(monkeypatch):
    """Regression test: a broken-Drive-access failure (folder resolution/
    listing, not a single video's download - see read_summaries_bulk())
    must propagate to the caller (the Streamlit app's public failure
    boundary), not be silently swallowed into a misleading all-pending
    snapshot."""
    monkeypatch.setattr(insights_store.channel_store, "read_channels", lambda folder_id: [])
    monkeypatch.setattr(insights_store.drive, "read_index", lambda folder_id: {"vid1": {"status": "ok"}})

    def _raise(folder_id, video_ids):
        raise ConnectionError("boom")

    monkeypatch.setattr(insights_store.summary_store, "read_summaries_bulk", _raise)

    with pytest.raises(ConnectionError):
        insights_store.load_snapshot("folder-id")


def test_load_snapshot_excludes_never_summarized_video_and_counts_it_pending(monkeypatch):
    _stub(monkeypatch, channels=[], index={"vid1": {"status": "ok"}}, summaries={})

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.videos == []
    assert snapshot.pending_count == 1
    assert snapshot.load_errors == []


def test_load_snapshot_excludes_error_status_summary_and_counts_it_pending(monkeypatch):
    _stub(
        monkeypatch,
        channels=[],
        index={"vid1": {"status": "ok"}},
        summaries={"vid1": {"status": "error", "message": "boom"}},
    )

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.videos == []
    assert snapshot.pending_count == 1
    assert snapshot.load_errors == []


def test_load_snapshot_excludes_malformed_ok_artifact_and_records_load_error(monkeypatch):
    """Unlike a recorded status: 'error', a status: 'ok' artifact missing
    required fields is unexpected - a real data problem worth surfacing,
    not a routine pending state."""
    _stub(
        monkeypatch,
        channels=[],
        index={"vid1": {"status": "ok"}},
        summaries={"vid1": {"status": "ok", "title": "Missing everything else"}},
    )

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.videos == []
    assert snapshot.pending_count == 0
    assert snapshot.load_errors == ["Summary artifact for vid1 is malformed."]


def test_load_snapshot_ignores_non_ok_index_entries(monkeypatch):
    _stub(monkeypatch, channels=[], index={"vid1": {"status": "blocked"}}, summaries={})

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.videos == []
    assert snapshot.pending_count == 0


def test_load_snapshot_skips_malformed_points_but_keeps_valid_ones(monkeypatch):
    artifact = _artifact(
        points=[
            {"importance": "major", "main_point": "Good point", "explanation": "Yes", "timestamp_seconds": 1},
            {"importance": "not-a-real-importance", "main_point": "Bad", "explanation": "No"},
            "not-a-dict",
        ]
    )
    _stub(monkeypatch, channels=[], index={"vid1": {"status": "ok"}}, summaries={"vid1": artifact})

    snapshot = insights_store.load_snapshot("folder-id")

    assert len(snapshot.videos[0].points) == 1
    assert snapshot.videos[0].points[0].main_point == "Good point"


def test_load_snapshot_missing_video_published_at_is_none(monkeypatch):
    artifact = _artifact(video_published_at=None)
    _stub(monkeypatch, channels=[], index={"vid1": {"status": "ok"}}, summaries={"vid1": artifact})

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.videos[0].video_published_at is None
    # generated_at is still parsed - it's the sort fallback for undated videos.
    assert snapshot.videos[0].generated_at == datetime(2026, 7, 11, 0, 0, 0, tzinfo=timezone.utc)


def test_load_snapshot_cost_usage_aggregates_ledger_entries(monkeypatch):
    ledger = [
        {"video_id": "vid1", "outcome": "ok", "input_tokens": 1000, "output_tokens": 200, "estimated_cost_usd": 0.01},
        {"video_id": "vid2", "outcome": "error", "input_tokens": 500, "output_tokens": 0, "estimated_cost_usd": 0.002},
    ]
    _stub(monkeypatch, channels=[], index={}, ledger=ledger)

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.cost_usage.total_attempts == 2
    assert snapshot.cost_usage.successful_attempts == 1
    assert snapshot.cost_usage.failed_attempts == 1
    assert snapshot.cost_usage.total_input_tokens == 1500
    assert snapshot.cost_usage.total_output_tokens == 200
    assert snapshot.cost_usage.total_estimated_cost_usd == pytest.approx(0.012)


def test_load_snapshot_cost_usage_counts_every_retry_attempt_separately(monkeypatch):
    """Issue: a summary artifact is overwritten in place on retry, so its
    current state alone can't distinguish "one attempt" from "several" -
    the ledger records each attempt as its own entry, so a video retried
    after an earlier failure must have both attempts' usage counted, not
    just whatever the artifact currently shows."""
    ledger = [
        {"video_id": "vid1", "outcome": "error", "input_tokens": 400, "output_tokens": 0, "estimated_cost_usd": 0.004},
        {"video_id": "vid1", "outcome": "ok", "input_tokens": 400, "output_tokens": 150, "estimated_cost_usd": 0.006},
    ]
    _stub(monkeypatch, channels=[], index={}, ledger=ledger)

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.cost_usage.total_attempts == 2
    assert snapshot.cost_usage.successful_attempts == 1
    assert snapshot.cost_usage.failed_attempts == 1
    assert snapshot.cost_usage.total_input_tokens == 800
    assert snapshot.cost_usage.total_estimated_cost_usd == pytest.approx(0.01)


def test_load_snapshot_cost_usage_ignores_malformed_ledger_entries(monkeypatch):
    ledger = [{"video_id": "vid1", "outcome": "ok"}, "not a dict", {"video_id": "vid2", "outcome": "ok", "input_tokens": 10, "output_tokens": 5, "estimated_cost_usd": 0.001}]
    _stub(monkeypatch, channels=[], index={}, ledger=ledger)

    snapshot = insights_store.load_snapshot("folder-id")

    assert snapshot.cost_usage.total_attempts == 1
    assert snapshot.cost_usage.total_input_tokens == 10


def test_compute_cost_usage_with_empty_ledger_returns_zeroed_summary():
    summary = insights_store._compute_cost_usage([])
    assert summary == insights_store.CostUsageSummary()
