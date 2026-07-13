import json

from app import channel_store


def _with_real_drive(monkeypatch, text):
    monkeypatch.setattr(channel_store.settings, "dry_run", False)
    monkeypatch.setattr(channel_store.drive, "download_text", lambda folder_id, filename: text)


def test_read_channels_parses_valid_registry(monkeypatch):
    _with_real_drive(
        monkeypatch,
        json.dumps(
            {
                "version": 1,
                "channels": [
                    {"channel_id": "UC_enabled", "name": "Enabled Channel", "enabled": True, "languages": ["en", "es"]},
                    {"channel_id": "UC_disabled", "name": "Disabled Channel", "enabled": False},
                    {"channel_id": "UC_default_enabled"},
                ],
            }
        ),
    )

    channels = channel_store.read_channels("folder-id")

    assert len(channels) == 3
    assert channels[0] == channel_store.Channel("UC_enabled", "Enabled Channel", True, ["en", "es"])
    assert channels[1] == channel_store.Channel("UC_disabled", "Disabled Channel", False, None)
    # Missing "name"/"enabled" default to the channel_id and True respectively.
    assert channels[2] == channel_store.Channel("UC_default_enabled", "UC_default_enabled", True, None)


def test_read_channels_missing_file_returns_empty(monkeypatch):
    _with_real_drive(monkeypatch, None)
    assert channel_store.read_channels("folder-id") == []


def test_read_channels_malformed_json_returns_empty(monkeypatch):
    _with_real_drive(monkeypatch, "not json")
    assert channel_store.read_channels("folder-id") == []


def test_read_channels_missing_channels_key_returns_empty(monkeypatch):
    _with_real_drive(monkeypatch, json.dumps({"version": 1}))
    assert channel_store.read_channels("folder-id") == []


def test_read_channels_skips_malformed_entries(monkeypatch):
    _with_real_drive(
        monkeypatch,
        json.dumps({"channels": [{"no_channel_id": "oops"}, {"channel_id": "UC_ok"}, "not-a-dict"]}),
    )

    channels = channel_store.read_channels("folder-id")

    assert len(channels) == 1
    assert channels[0].channel_id == "UC_ok"


def test_read_channels_dry_run_returns_empty_without_touching_drive(monkeypatch):
    monkeypatch.setattr(channel_store.settings, "dry_run", True)
    called = []
    monkeypatch.setattr(channel_store.drive, "download_text", lambda *a, **k: called.append(1))

    assert channel_store.read_channels("folder-id") == []
    assert not called
