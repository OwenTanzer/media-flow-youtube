import json
from datetime import datetime, timedelta, timezone

from app import job_lock


def _set_real_drive(monkeypatch):
    monkeypatch.setattr(job_lock.settings, "dry_run", False)


def test_acquire_lock_succeeds_when_no_lock_exists(monkeypatch):
    _set_real_drive(monkeypatch)
    monkeypatch.setattr(job_lock.drive, "download_text", lambda folder_id, filename: None)
    written = {}
    monkeypatch.setattr(
        job_lock.drive,
        "upload_text_file",
        lambda folder_id, filename, content, **k: written.setdefault("content", content),
    )

    assert job_lock.acquire_lock("folder-id", ttl_seconds=1800) is True
    assert "acquired_at" in written["content"]


def test_acquire_lock_fails_when_fresh_lock_held(monkeypatch):
    _set_real_drive(monkeypatch)
    fresh = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(job_lock.drive, "download_text", lambda folder_id, filename: json.dumps({"acquired_at": fresh}))
    write_calls = []
    monkeypatch.setattr(job_lock.drive, "upload_text_file", lambda *a, **k: write_calls.append(1))

    assert job_lock.acquire_lock("folder-id", ttl_seconds=1800) is False
    assert not write_calls


def test_acquire_lock_succeeds_and_overwrites_stale_lock(monkeypatch):
    _set_real_drive(monkeypatch)
    stale = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat()
    monkeypatch.setattr(job_lock.drive, "download_text", lambda folder_id, filename: json.dumps({"acquired_at": stale}))
    written = {}
    monkeypatch.setattr(
        job_lock.drive,
        "upload_text_file",
        lambda folder_id, filename, content, **k: written.setdefault("content", content),
    )

    assert job_lock.acquire_lock("folder-id", ttl_seconds=1800) is True
    assert "acquired_at" in written["content"]


def test_acquire_lock_treats_unreadable_lock_file_as_stale(monkeypatch):
    _set_real_drive(monkeypatch)
    monkeypatch.setattr(job_lock.drive, "download_text", lambda folder_id, filename: "not json")
    written = {}
    monkeypatch.setattr(
        job_lock.drive,
        "upload_text_file",
        lambda folder_id, filename, content, **k: written.setdefault("content", content),
    )

    assert job_lock.acquire_lock("folder-id", ttl_seconds=1800) is True
    assert written


def test_release_lock_deletes_the_file(monkeypatch):
    _set_real_drive(monkeypatch)
    deleted = []
    monkeypatch.setattr(job_lock.drive, "delete_file", lambda folder_id, filename: deleted.append(filename))

    job_lock.release_lock("folder-id")

    assert deleted == [job_lock.LOCK_FILENAME]


def test_acquire_and_release_are_no_ops_in_dry_run(monkeypatch):
    monkeypatch.setattr(job_lock.settings, "dry_run", True)
    called = []
    monkeypatch.setattr(job_lock.drive, "download_text", lambda *a, **k: called.append(1))
    monkeypatch.setattr(job_lock.drive, "upload_text_file", lambda *a, **k: called.append(1))
    monkeypatch.setattr(job_lock.drive, "delete_file", lambda *a, **k: called.append(1))

    assert job_lock.acquire_lock("folder-id", ttl_seconds=1800) is True
    job_lock.release_lock("folder-id")
    assert not called
