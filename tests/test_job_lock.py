import json
from datetime import datetime, timedelta, timezone

from app import job_lock


def _set_real_drive(monkeypatch):
    monkeypatch.setattr(job_lock.settings, "dry_run", False)


def _no_other_lock_files(monkeypatch):
    """Simulates the common (non-racing) case: after we write the lock,
    we're the only file with that name."""
    monkeypatch.setattr(job_lock.drive, "list_file_ids", lambda folder_id, filename: ["only-file-id"])


def test_acquire_lock_succeeds_when_no_lock_exists(monkeypatch):
    _set_real_drive(monkeypatch)
    _no_other_lock_files(monkeypatch)
    written = {}
    monkeypatch.setattr(
        job_lock.drive,
        "upload_text_file",
        lambda folder_id, filename, content, **k: written.setdefault("content", content),
    )
    # download_text is consulted twice: once before writing (no lock yet),
    # once after writing to confirm our own token stuck.
    calls = {"n": 0}

    def _download(folder_id, filename):
        calls["n"] += 1
        return None if calls["n"] == 1 else written["content"]

    monkeypatch.setattr(job_lock.drive, "download_text", _download)

    token = job_lock.acquire_lock("folder-id", ttl_seconds=1800)

    assert token is not None
    assert json.loads(written["content"])["token"] == token


def test_acquire_lock_fails_when_fresh_lock_held(monkeypatch):
    _set_real_drive(monkeypatch)
    fresh = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(
        job_lock.drive,
        "download_text",
        lambda folder_id, filename: json.dumps({"acquired_at": fresh, "token": "someone-elses-token"}),
    )
    write_calls = []
    monkeypatch.setattr(job_lock.drive, "upload_text_file", lambda *a, **k: write_calls.append(1))

    assert job_lock.acquire_lock("folder-id", ttl_seconds=1800) is None
    assert not write_calls


def test_acquire_lock_succeeds_and_overwrites_stale_lock(monkeypatch):
    _set_real_drive(monkeypatch)
    _no_other_lock_files(monkeypatch)
    stale = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat()
    written = {}
    calls = {"n": 0}

    def _download(folder_id, filename):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"acquired_at": stale, "token": "old-token"})
        return written["content"]

    monkeypatch.setattr(job_lock.drive, "download_text", _download)
    monkeypatch.setattr(
        job_lock.drive,
        "upload_text_file",
        lambda folder_id, filename, content, **k: written.setdefault("content", content),
    )

    token = job_lock.acquire_lock("folder-id", ttl_seconds=1800)

    assert token is not None
    assert token != "old-token"
    assert json.loads(written["content"])["token"] == token


def test_acquire_lock_treats_unreadable_lock_file_as_stale(monkeypatch):
    _set_real_drive(monkeypatch)
    _no_other_lock_files(monkeypatch)
    written = {}
    calls = {"n": 0}

    def _download(folder_id, filename):
        calls["n"] += 1
        return "not json" if calls["n"] == 1 else written["content"]

    monkeypatch.setattr(job_lock.drive, "download_text", _download)
    monkeypatch.setattr(
        job_lock.drive,
        "upload_text_file",
        lambda folder_id, filename, content, **k: written.setdefault("content", content),
    )

    assert job_lock.acquire_lock("folder-id", ttl_seconds=1800) is not None
    assert written


def test_acquire_lock_backs_off_when_a_duplicate_lock_file_appears(monkeypatch):
    """Drive permits duplicate filenames, so a near-simultaneous writer
    could create a second _discovery_lock.json instead of racing to
    update this one. Detecting more than one file must back off rather
    than let both callers believe they hold the lock."""
    _set_real_drive(monkeypatch)
    monkeypatch.setattr(job_lock.drive, "download_text", lambda folder_id, filename: None)
    monkeypatch.setattr(job_lock.drive, "upload_text_file", lambda *a, **k: None)
    monkeypatch.setattr(job_lock.drive, "list_file_ids", lambda folder_id, filename: ["file-a", "file-b"])

    assert job_lock.acquire_lock("folder-id", ttl_seconds=1800) is None


def test_acquire_lock_backs_off_when_a_concurrent_writer_overwrote_the_token(monkeypatch):
    """Even with exactly one file present, if its token doesn't match what
    we just wrote, someone else's write landed on top of ours."""
    _set_real_drive(monkeypatch)
    _no_other_lock_files(monkeypatch)
    monkeypatch.setattr(job_lock.drive, "upload_text_file", lambda *a, **k: None)
    calls = {"n": 0}

    def _download(folder_id, filename):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return json.dumps({"acquired_at": datetime.now(timezone.utc).isoformat(), "token": "someone-elses-token"})

    monkeypatch.setattr(job_lock.drive, "download_text", _download)

    assert job_lock.acquire_lock("folder-id", ttl_seconds=1800) is None


def test_release_lock_deletes_the_file_when_token_matches(monkeypatch):
    _set_real_drive(monkeypatch)
    monkeypatch.setattr(
        job_lock.drive,
        "download_text",
        lambda folder_id, filename: json.dumps({"acquired_at": "2026-01-01T00:00:00+00:00", "token": "my-token"}),
    )
    deleted = []
    monkeypatch.setattr(job_lock.drive, "delete_file", lambda folder_id, filename: deleted.append(filename))

    job_lock.release_lock("folder-id", "my-token")

    assert deleted == [job_lock.LOCK_FILENAME]


def test_release_lock_does_not_delete_a_different_runs_lease(monkeypatch):
    """Regression test for the review finding: a run that outlives its own
    TTL must not blindly delete whatever lock currently exists - a third
    run may have already taken over the lease."""
    _set_real_drive(monkeypatch)
    monkeypatch.setattr(
        job_lock.drive,
        "download_text",
        lambda folder_id, filename: json.dumps({"acquired_at": "2026-01-01T00:00:00+00:00", "token": "someone-elses-token"}),
    )
    deleted = []
    monkeypatch.setattr(job_lock.drive, "delete_file", lambda folder_id, filename: deleted.append(filename))

    job_lock.release_lock("folder-id", "my-token")

    assert deleted == []


def test_release_lock_is_a_noop_when_no_lock_file_exists(monkeypatch):
    _set_real_drive(monkeypatch)
    monkeypatch.setattr(job_lock.drive, "download_text", lambda folder_id, filename: None)
    deleted = []
    monkeypatch.setattr(job_lock.drive, "delete_file", lambda folder_id, filename: deleted.append(filename))

    job_lock.release_lock("folder-id", "my-token")

    assert deleted == []


def test_acquire_and_release_are_no_ops_in_dry_run(monkeypatch):
    monkeypatch.setattr(job_lock.settings, "dry_run", True)
    called = []
    monkeypatch.setattr(job_lock.drive, "download_text", lambda *a, **k: called.append(1))
    monkeypatch.setattr(job_lock.drive, "upload_text_file", lambda *a, **k: called.append(1))
    monkeypatch.setattr(job_lock.drive, "delete_file", lambda *a, **k: called.append(1))
    monkeypatch.setattr(job_lock.drive, "list_file_ids", lambda *a, **k: called.append(1))

    token = job_lock.acquire_lock("folder-id", ttl_seconds=1800)
    assert token is not None
    job_lock.release_lock("folder-id", token)
    assert not called
