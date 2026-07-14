from datetime import datetime, timedelta, timezone

from app import batch
from app.models import VideoResult


def _result(video_id, status, url=None):
    return VideoResult(video_id=video_id, url=url or f"https://www.youtube.com/watch?v={video_id}", status=status)


def test_run_batch_with_explicit_urls_never_touches_the_queue(monkeypatch):
    queue_reads = []
    queue_writes = []
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: queue_reads.append(folder_id))
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: queue_writes.append(urls))
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result("vid", "ok", url))

    results = batch.run_batch(urls=["https://www.youtube.com/watch?v=vid"])

    assert len(results) == 1
    assert results[0].status == "ok"
    assert not queue_reads
    assert not queue_writes


def test_run_batch_from_queue_keeps_only_transient_failures(monkeypatch):
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: ["a", "b", "c", "d"])
    written = {}
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: written.setdefault("urls", urls))

    outcomes = {
        "a": _result("a", "ok"),
        "b": _result("b", "no_captions"),
        "c": _result("c", "blocked"),  # transient - should stay queued
        "d": _result("d", "error"),  # transient - should stay queued
    }
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: outcomes[url])

    results = batch.run_batch()

    assert len(results) == 4
    assert written["urls"] == ["c", "d"]


def test_run_batch_passes_dict_entry_languages_to_safe_process_video(monkeypatch):
    monkeypatch.setattr(
        batch.queue_store, "read_queue", lambda folder_id: [{"url": "https://youtu.be/x", "languages": ["es"]}]
    )
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: None)

    seen = {}

    def _fake(url, languages=None):
        seen["url"] = url
        seen["languages"] = languages
        return _result("x", "ok", url)

    monkeypatch.setattr(batch, "safe_process_video", _fake)

    batch.run_batch()

    assert seen["url"] == "https://youtu.be/x"
    assert seen["languages"] == ["es"]


def test_run_batch_requeues_transient_failure_as_the_same_dict_entry(monkeypatch):
    entry = {"url": "https://youtu.be/x", "languages": ["es"]}
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: [entry])
    written = {}
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: written.setdefault("urls", urls))
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result("x", "blocked", url))

    batch.run_batch()

    assert written["urls"] == [entry]


def test_run_batch_retries_no_captions_within_grace_period(monkeypatch):
    """Regression test for livestream handling: a video discovered recently
    (e.g. a livestream still in progress) shouldn't have "no_captions"
    treated as final - it should stay queued for the next run."""
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    entry = {"url": "https://youtu.be/x", "first_seen_at": recent}
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: [entry])
    written = {}
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: written.setdefault("urls", urls))
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result("x", "no_captions", url))

    batch.run_batch()

    assert written["urls"] == [entry]


def test_run_batch_drops_no_captions_past_grace_period(monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    entry = {"url": "https://youtu.be/x", "first_seen_at": old}
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: [entry])
    written = {}
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: written.setdefault("urls", urls))
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result("x", "no_captions", url))

    batch.run_batch()

    assert written["urls"] == []


def test_run_batch_drops_no_captions_immediately_when_entry_has_no_first_seen_at(monkeypatch):
    """Manually-added queue.json entries (plain strings, or dicts without
    first_seen_at) get no grace period - same behavior as before this
    existed."""
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: ["https://youtu.be/x"])
    written = {}
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: written.setdefault("urls", urls))
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result("x", "no_captions", url))

    batch.run_batch()

    assert written["urls"] == []


def test_run_batch_empty_queue_is_a_noop(monkeypatch):
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: [])
    write_calls = []
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: write_calls.append(urls))

    results = batch.run_batch()

    assert results == []
    assert not write_calls
