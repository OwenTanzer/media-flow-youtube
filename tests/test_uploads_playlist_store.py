import json

from app import uploads_playlist_store


def test_read_cache_returns_empty_dict_when_file_missing(monkeypatch):
    monkeypatch.setattr(uploads_playlist_store.drive, "download_text", lambda folder_id, filename: None)
    assert uploads_playlist_store.read_cache("folder-id") == {}


def test_read_cache_parses_existing_file(monkeypatch):
    monkeypatch.setattr(
        uploads_playlist_store.drive,
        "download_text",
        lambda folder_id, filename: json.dumps({"UC_a": "PL_a", "UC_b": "PL_b"}),
    )
    assert uploads_playlist_store.read_cache("folder-id") == {"UC_a": "PL_a", "UC_b": "PL_b"}


def test_read_cache_treats_malformed_json_as_empty(monkeypatch):
    monkeypatch.setattr(uploads_playlist_store.drive, "download_text", lambda folder_id, filename: "not json")
    assert uploads_playlist_store.read_cache("folder-id") == {}


def test_read_cache_treats_non_dict_json_as_empty(monkeypatch):
    monkeypatch.setattr(uploads_playlist_store.drive, "download_text", lambda folder_id, filename: "[1, 2, 3]")
    assert uploads_playlist_store.read_cache("folder-id") == {}


def test_read_cache_skips_non_string_values(monkeypatch):
    monkeypatch.setattr(
        uploads_playlist_store.drive,
        "download_text",
        lambda folder_id, filename: json.dumps({"UC_a": "PL_a", "UC_b": 123}),
    )
    assert uploads_playlist_store.read_cache("folder-id") == {"UC_a": "PL_a"}


def test_write_cache_round_trips_through_read_cache(monkeypatch):
    written = {}
    monkeypatch.setattr(
        uploads_playlist_store.drive,
        "upload_text_file",
        lambda folder_id, filename, content, mime_type=None: written.setdefault("content", content),
    )
    uploads_playlist_store.write_cache("folder-id", {"UC_a": "PL_a"})

    monkeypatch.setattr(uploads_playlist_store.drive, "download_text", lambda folder_id, filename: written["content"])
    assert uploads_playlist_store.read_cache("folder-id") == {"UC_a": "PL_a"}
