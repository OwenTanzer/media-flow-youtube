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


def test_run_batch_empty_queue_is_a_noop(monkeypatch):
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: [])
    write_calls = []
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: write_calls.append(urls))

    results = batch.run_batch()

    assert results == []
    assert not write_calls
