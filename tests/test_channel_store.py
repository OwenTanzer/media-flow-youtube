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
                    {
                        "channel_id": "UC_enabled",
                        "name": "Enabled Channel",
                        "enabled": True,
                        "languages": ["en", "es"],
                        "group": "Google",
                    },
                    {"channel_id": "UC_disabled", "name": "Disabled Channel", "enabled": False},
                    {"channel_id": "UC_default_enabled"},
                    {"channel_id": "UC_blank_group", "group": "   "},
                    {"channel_id": "UC_numeric_group", "group": 5},
                ],
            }
        ),
    )

    channels = channel_store.read_channels("folder-id")

    assert len(channels) == 5
    assert channels[0] == channel_store.Channel("UC_enabled", "Enabled Channel", True, ["en", "es"], "Google")
    assert channels[1] == channel_store.Channel("UC_disabled", "Disabled Channel", False, None, None)
    # Missing "name"/"enabled" default to the channel_id and True respectively.
    assert channels[2] == channel_store.Channel("UC_default_enabled", "UC_default_enabled", True, None, None)
    # A blank/whitespace-only or non-string "group" is treated as absent, not taken literally.
    assert channels[3].group is None
    assert channels[4].group is None


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


def _capture_upload(monkeypatch):
    written = {}

    def _upload(folder_id, filename, content, **kwargs):
        written["folder_id"] = folder_id
        written["filename"] = filename
        written["content"] = json.loads(content)

    monkeypatch.setattr(channel_store.drive, "upload_text_file", _upload)
    return written


def test_write_channels_round_trips_through_read_channels(monkeypatch):
    written = _capture_upload(monkeypatch)
    channels = [
        channel_store.Channel("UC_a", "Channel A", True, ["en", "es"], "Google"),
        channel_store.Channel("UC_b", "Channel B", False, None, None),
    ]

    channel_store.write_channels("folder-id", channels)

    assert written["filename"] == channel_store.CHANNELS_FILENAME
    assert written["folder_id"] == "folder-id"

    monkeypatch.setattr(channel_store.settings, "dry_run", False)
    monkeypatch.setattr(channel_store.drive, "download_text", lambda folder_id, filename: json.dumps(written["content"]))
    assert channel_store.read_channels("folder-id") == channels


def test_write_channels_omits_absent_languages_and_group(monkeypatch):
    written = _capture_upload(monkeypatch)

    channel_store.write_channels("folder-id", [channel_store.Channel("UC_a", "Channel A")])

    entry = written["content"]["channels"][0]
    assert "languages" not in entry
    assert "group" not in entry
