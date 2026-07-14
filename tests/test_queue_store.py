from datetime import datetime, timezone

from app import queue_store


def test_entry_first_seen_at_parses_valid_dict_entry():
    entry = {"url": "https://youtu.be/x", "first_seen_at": "2026-07-01T12:00:00+00:00"}
    assert queue_store.entry_first_seen_at(entry) == datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_entry_first_seen_at_returns_none_for_plain_string_entry():
    assert queue_store.entry_first_seen_at("https://youtu.be/x") is None


def test_entry_first_seen_at_returns_none_when_field_missing():
    assert queue_store.entry_first_seen_at({"url": "https://youtu.be/x"}) is None


def test_entry_first_seen_at_returns_none_for_unparseable_value():
    assert queue_store.entry_first_seen_at({"url": "https://youtu.be/x", "first_seen_at": "not a date"}) is None


def test_read_queue_preserves_first_seen_at(monkeypatch):
    monkeypatch.setattr(queue_store.settings, "dry_run", False)
    monkeypatch.setattr(
        queue_store.drive,
        "download_text",
        lambda folder_id, filename: '[{"url": "https://youtu.be/x", "first_seen_at": "2026-07-01T12:00:00+00:00"}]',
    )

    entries = queue_store.read_queue("folder-id")

    assert entries == [{"url": "https://youtu.be/x", "first_seen_at": "2026-07-01T12:00:00+00:00"}]


def test_read_queue_ignores_non_string_first_seen_at(monkeypatch):
    monkeypatch.setattr(queue_store.settings, "dry_run", False)
    monkeypatch.setattr(
        queue_store.drive,
        "download_text",
        lambda folder_id, filename: '[{"url": "https://youtu.be/x", "first_seen_at": 12345}]',
    )

    entries = queue_store.read_queue("folder-id")

    assert entries == [{"url": "https://youtu.be/x"}]
