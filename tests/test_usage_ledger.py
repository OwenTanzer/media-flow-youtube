import json

from app import usage_ledger


def test_read_ledger_returns_empty_list_when_file_missing(monkeypatch):
    monkeypatch.setattr(usage_ledger.drive, "download_text", lambda folder_id, filename: None)
    assert usage_ledger.read_ledger("folder-id") == []


def test_read_ledger_parses_existing_file(monkeypatch):
    entries = [{"video_id": "vid1", "outcome": "ok", "input_tokens": 10, "output_tokens": 5, "estimated_cost_usd": 0.001}]
    monkeypatch.setattr(usage_ledger.drive, "download_text", lambda folder_id, filename: json.dumps(entries))
    assert usage_ledger.read_ledger("folder-id") == entries


def test_read_ledger_treats_malformed_json_as_empty(monkeypatch):
    monkeypatch.setattr(usage_ledger.drive, "download_text", lambda folder_id, filename: "not json")
    assert usage_ledger.read_ledger("folder-id") == []


def test_read_ledger_treats_non_list_json_as_empty(monkeypatch):
    monkeypatch.setattr(usage_ledger.drive, "download_text", lambda folder_id, filename: json.dumps({"not": "a list"}))
    assert usage_ledger.read_ledger("folder-id") == []


def test_append_entries_is_a_noop_with_no_entries(monkeypatch):
    called = []
    monkeypatch.setattr(usage_ledger, "read_ledger", lambda folder_id: called.append("read") or [])
    monkeypatch.setattr(usage_ledger.drive, "upload_text_file", lambda *a, **k: called.append("write"))

    usage_ledger.append_entries("folder-id", [])

    assert called == []


def test_append_entries_appends_to_existing_ledger_in_one_write(monkeypatch):
    existing = [{"video_id": "old", "outcome": "ok", "input_tokens": 1, "output_tokens": 1, "estimated_cost_usd": 0.0001}]
    monkeypatch.setattr(usage_ledger, "read_ledger", lambda folder_id: existing)
    written = {}
    monkeypatch.setattr(
        usage_ledger.drive,
        "upload_text_file",
        lambda folder_id, filename, content, mime_type=None: written.setdefault("content", content),
    )
    new_entry = {"video_id": "new", "outcome": "error", "input_tokens": 2, "output_tokens": 0, "estimated_cost_usd": 0.0002}

    usage_ledger.append_entries("folder-id", [new_entry])

    assert json.loads(written["content"]) == [*existing, new_entry]
