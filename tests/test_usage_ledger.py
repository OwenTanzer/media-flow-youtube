import json

import pytest

from app import usage_ledger


def test_read_ledger_returns_empty_list_when_file_missing(monkeypatch):
    monkeypatch.setattr(usage_ledger.drive, "download_text", lambda folder_id, filename: None)
    assert usage_ledger.read_ledger("folder-id") == []


def test_read_ledger_parses_existing_file(monkeypatch):
    entries = [{"video_id": "vid1", "outcome": "ok", "input_tokens": 10, "output_tokens": 5, "estimated_cost_usd": 0.001}]
    monkeypatch.setattr(usage_ledger.drive, "download_text", lambda folder_id, filename: json.dumps(entries))
    assert usage_ledger.read_ledger("folder-id") == entries


@pytest.mark.parametrize("bad_text", ["not json", json.dumps({"not": "a list"})])
def test_read_ledger_raises_usage_ledger_corrupt_error(monkeypatch, bad_text):
    """Unlike a missing file (empty list, a totally normal state), an
    existing-but-unparseable file must not be silently swallowed - see
    UsageLedgerCorruptError's docstring on why append_entries() needs to
    be able to tell the two apart."""
    monkeypatch.setattr(usage_ledger.drive, "download_text", lambda folder_id, filename: bad_text)
    with pytest.raises(usage_ledger.UsageLedgerCorruptError):
        usage_ledger.read_ledger("folder-id")


def _stub_lock(monkeypatch, *, acquired=True):
    monkeypatch.setattr(
        usage_ledger.job_lock, "acquire_lock", lambda folder_id, ttl, lock_filename: "token" if acquired else None
    )
    released = []
    monkeypatch.setattr(
        usage_ledger.job_lock, "release_lock", lambda folder_id, token, lock_filename: released.append(token)
    )
    return released


def test_append_entries_is_a_noop_with_no_entries(monkeypatch):
    called = []
    monkeypatch.setattr(usage_ledger.job_lock, "acquire_lock", lambda *a, **k: called.append("lock") or "token")
    monkeypatch.setattr(usage_ledger, "read_ledger", lambda folder_id: called.append("read") or [])
    monkeypatch.setattr(usage_ledger.drive, "upload_text_file", lambda *a, **k: called.append("write"))

    usage_ledger.append_entries("folder-id", [])

    assert called == []


def test_append_entries_appends_to_existing_ledger_under_lock(monkeypatch):
    existing = [{"video_id": "old", "attempt_id": "a1", "outcome": "ok", "input_tokens": 1, "output_tokens": 1, "estimated_cost_usd": 0.0001}]
    released = _stub_lock(monkeypatch)
    monkeypatch.setattr(usage_ledger, "read_ledger", lambda folder_id: existing)
    written = {}
    monkeypatch.setattr(
        usage_ledger.drive,
        "upload_text_file",
        lambda folder_id, filename, content, mime_type=None: written.setdefault("content", content),
    )
    new_entry = {"video_id": "new", "attempt_id": "a2", "outcome": "error", "input_tokens": 2, "output_tokens": 0, "estimated_cost_usd": 0.0002}

    usage_ledger.append_entries("folder-id", [new_entry])

    assert json.loads(written["content"]) == [*existing, new_entry]
    assert released == ["token"]  # lock released even on the success path


def test_append_entries_skips_when_lock_not_acquired(monkeypatch):
    _stub_lock(monkeypatch, acquired=False)
    called = []
    monkeypatch.setattr(usage_ledger, "read_ledger", lambda folder_id: called.append("read") or [])
    monkeypatch.setattr(usage_ledger.drive, "upload_text_file", lambda *a, **k: called.append("write"))

    usage_ledger.append_entries("folder-id", [{"attempt_id": "a1"}])

    assert called == []


def test_append_entries_deduplicates_by_attempt_id(monkeypatch):
    """Idempotency: submitting an entry whose attempt_id is already present
    must not double-count it."""
    existing = [{"video_id": "vid1", "attempt_id": "dup", "outcome": "ok", "input_tokens": 5, "output_tokens": 5, "estimated_cost_usd": 0.001}]
    _stub_lock(monkeypatch)
    monkeypatch.setattr(usage_ledger, "read_ledger", lambda folder_id: existing)
    written = {}
    monkeypatch.setattr(
        usage_ledger.drive,
        "upload_text_file",
        lambda folder_id, filename, content, mime_type=None: written.setdefault("content", content),
    )
    duplicate_entry = {"video_id": "vid1", "attempt_id": "dup", "outcome": "ok", "input_tokens": 5, "output_tokens": 5, "estimated_cost_usd": 0.001}

    usage_ledger.append_entries("folder-id", [duplicate_entry])

    assert "content" not in written  # nothing new to write - no upload call at all


def test_append_entries_fails_closed_on_corrupt_ledger(monkeypatch):
    """Must not overwrite a corrupt ledger with just the new entries -
    that would silently discard whatever the corrupt file still held."""
    _stub_lock(monkeypatch)

    def _raise(folder_id):
        raise usage_ledger.UsageLedgerCorruptError("corrupt")

    monkeypatch.setattr(usage_ledger, "read_ledger", _raise)
    called = []
    monkeypatch.setattr(usage_ledger.drive, "upload_text_file", lambda *a, **k: called.append("write"))

    usage_ledger.append_entries("folder-id", [{"attempt_id": "a1"}])

    assert called == []
