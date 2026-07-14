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


def test_run_batch_does_not_crash_on_a_timezone_naive_first_seen_at(monkeypatch):
    """Regression test for the review finding: queue.json is operator-editable,
    so a timezone-less first_seen_at (valid per datetime.fromisoformat) must
    not crash the whole batch run via an aware-vs-naive subtraction."""
    entry = {"url": "https://youtu.be/x", "first_seen_at": "2026-07-14T12:00:00"}
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: [entry])
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: None)
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result("x", "no_captions", url))

    results = batch.run_batch()  # must not raise

    assert len(results) == 1


def test_run_batch_does_not_pace_when_at_or_below_threshold(monkeypatch):
    monkeypatch.setattr(batch.settings, "batch_size_threshold", 5)
    monkeypatch.setattr(batch.settings, "batch_cooldown_seconds", 300)
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: ["a", "b", "c", "d", "e"])
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: None)
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result(url, "ok", url))
    sleep_calls = []
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    results = batch.run_batch()

    assert len(results) == 5
    assert sleep_calls == []


def test_run_batch_paces_in_cooled_down_chunks_when_above_threshold(monkeypatch):
    """Regression test: a continuous run of many requests measurably
    degrades the rotating proxy pool's success rate (see README). Above
    the threshold, process in chunks with a real cooldown between them."""
    monkeypatch.setattr(batch.settings, "batch_size_threshold", 2)
    monkeypatch.setattr(batch.settings, "batch_cooldown_seconds", 300)
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: ["a", "b", "c", "d", "e"])
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: None)
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result(url, "ok", url))
    sleep_calls = []
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    results = batch.run_batch()

    # 5 entries chunked into [a,b] [c,d] [e] - a cooldown before chunk 2 and chunk 3, none before chunk 1.
    assert len(results) == 5
    assert sleep_calls == [300, 300]


def test_run_batch_does_not_pace_explicit_url_lists(monkeypatch):
    """Regression test for the review finding: pacing an explicit URL list
    (e.g. from a live POST /batch/run request) would hold the HTTP
    connection open for the full cooldown duration - confined to the
    queue path, which is only ever driven by standalone scripts."""
    monkeypatch.setattr(batch.settings, "batch_size_threshold", 1)
    monkeypatch.setattr(batch.settings, "batch_cooldown_seconds", 300)
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result(url, "ok", url))
    sleep_calls = []
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    results = batch.run_batch(urls=["a", "b", "c"])

    assert len(results) == 3
    assert sleep_calls == []


def test_run_batch_checkpoints_queue_after_every_chunk(monkeypatch):
    """Regression test for the review finding: writing queue.json only once
    at the very end means a crash partway through a long, multi-chunk run
    loses all progress - the next run would reprocess everything from
    scratch, including videos that already succeeded, recreating the exact
    proxy pressure batching exists to relieve."""
    monkeypatch.setattr(batch.settings, "batch_size_threshold", 2)
    monkeypatch.setattr(batch.settings, "batch_cooldown_seconds", 300)
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: ["a", "b", "c", "d", "e"])
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)

    outcomes = {
        "a": _result("a", "ok"),
        "b": _result("b", "blocked"),  # transient - stays queued
        "c": _result("c", "ok"),
        "d": _result("d", "blocked"),  # transient - stays queued
        "e": _result("e", "ok"),
    }
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: outcomes[url])

    checkpoints = []
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: checkpoints.append(list(urls)))

    batch.run_batch()

    # Chunk 1 [a,b]: "b" transient, "c,d,e" untouched -> checkpoint = [b, c, d, e]
    # Chunk 2 [c,d]: "d" transient, "e" untouched -> checkpoint = [b, d, e]
    # Chunk 3 [e]: no untouched tail left -> checkpoint = [b, d]
    assert checkpoints == [["b", "c", "d", "e"], ["b", "d", "e"], ["b", "d"]]


def test_run_batch_renews_after_every_entry_and_before_each_checkpoint(monkeypatch):
    """Regression test for the review finding: renewing only once per
    chunk (rather than once per entry) leaves a window where a slow chunk
    goes stale and gets taken over before the run notices, since a single
    entry's own internal retries can themselves run for minutes. Renew
    after every entry, and once more immediately before each checkpoint
    write (the point of the dangerous write)."""
    monkeypatch.setattr(batch.settings, "batch_size_threshold", 2)
    monkeypatch.setattr(batch.settings, "batch_cooldown_seconds", 300)
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: ["a", "b", "c", "d", "e"])
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: None)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result(url, "ok", url))

    calls = []
    batch.run_batch(on_progress=lambda: calls.append(1))

    # 3 chunks of [a,b] [c,d] [e]: one call per entry (5) + one more right
    # before each of the 3 checkpoint writes = 8.
    assert len(calls) == 8


def test_run_batch_calls_on_progress_in_the_unpaced_path_too(monkeypatch):
    """The unpaced path (queue at or below the threshold) still calls
    on_progress after the entry and before the final write - a single slow
    entry can still eat meaningfully into the lock's TTL even without
    chunking, so discover_and_process.py should get a chance to renew."""
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: ["a"])
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: None)
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result(url, "ok", url))

    calls = []
    batch.run_batch(on_progress=lambda: calls.append(1))

    assert len(calls) == 2  # once after the entry, once before the final write


def test_run_batch_does_not_call_on_progress_for_explicit_url_lists(monkeypatch):
    monkeypatch.setattr(batch, "safe_process_video", lambda url, languages=None: _result(url, "ok", url))

    calls = []
    batch.run_batch(urls=["a", "b"], on_progress=lambda: calls.append(1))

    assert calls == []


def test_run_batch_empty_queue_is_a_noop(monkeypatch):
    monkeypatch.setattr(batch.queue_store, "read_queue", lambda folder_id: [])
    write_calls = []
    monkeypatch.setattr(batch.queue_store, "write_queue", lambda folder_id, urls: write_calls.append(urls))

    results = batch.run_batch()

    assert results == []
    assert not write_calls
