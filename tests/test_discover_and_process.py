import pytest

import discover_and_process
from app.discovery import DiscoveryReport
from app.summary_store import SummaryReport

_EMPTY_SUMMARY_REPORT = SummaryReport(
    eligible=0,
    skipped_current=0,
    summarized=0,
    failed=0,
    retried=0,
    total_input_tokens=0,
    total_output_tokens=0,
    total_estimated_cost_usd=0.0,
)


def test_main_exits_without_acting_when_lock_is_held(monkeypatch):
    monkeypatch.setattr(discover_and_process.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(discover_and_process.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: None)

    called = []
    monkeypatch.setattr(discover_and_process, "discover_and_enqueue", lambda folder_id: called.append("discover"))
    monkeypatch.setattr(discover_and_process, "run_batch", lambda: called.append("run_batch"))
    release_calls = []
    monkeypatch.setattr(discover_and_process.job_lock, "release_lock", lambda folder_id, token: release_calls.append(1))

    exit_code = discover_and_process.main()

    assert exit_code == 1
    assert called == []
    # The lock was never acquired, so this run must not release someone else's lock.
    assert release_calls == []


def test_main_runs_discovery_then_batch_and_releases_lock_when_acquired(monkeypatch):
    monkeypatch.setattr(discover_and_process.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(discover_and_process.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: "the-token")

    order = []
    monkeypatch.setattr(
        discover_and_process,
        "discover_and_enqueue",
        lambda folder_id: order.append("discover") or DiscoveryReport(0, 0, 0, 0, 0, []),
    )
    monkeypatch.setattr(discover_and_process, "run_batch", lambda **kwargs: order.append("run_batch") or [])
    monkeypatch.setattr(
        discover_and_process,
        "summarize_eligible",
        lambda folder_id, **kwargs: order.append("summarize") or _EMPTY_SUMMARY_REPORT,
    )
    release_tokens = []
    monkeypatch.setattr(
        discover_and_process.job_lock,
        "release_lock",
        lambda folder_id, token: (release_tokens.append(token), order.append("release")),
    )

    exit_code = discover_and_process.main()

    assert exit_code == 0
    assert order == ["discover", "run_batch", "summarize", "release"]
    assert release_tokens == ["the-token"]


def test_main_passes_a_lock_renewing_callback_to_run_batch(monkeypatch):
    """Regression test for the review finding: a large queue's cooldowns
    alone can exceed DISCOVERY_LOCK_TTL_SECONDS. main() must give
    run_batch() a way to keep the lease fresh during a long run."""
    monkeypatch.setattr(discover_and_process.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(discover_and_process.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: "the-token")
    monkeypatch.setattr(
        discover_and_process, "discover_and_enqueue", lambda folder_id: DiscoveryReport(0, 0, 0, 0, 0, [])
    )
    monkeypatch.setattr(discover_and_process.job_lock, "release_lock", lambda folder_id, token: None)
    monkeypatch.setattr(discover_and_process, "summarize_eligible", lambda folder_id, **kwargs: _EMPTY_SUMMARY_REPORT)

    renew_calls = []
    monkeypatch.setattr(
        discover_and_process.job_lock,
        "renew_lock",
        lambda folder_id, token: renew_calls.append((folder_id, token)) or True,
    )

    captured = {}

    def _fake_run_batch(**kwargs):
        captured["on_progress"] = kwargs["on_progress"]
        captured["on_progress"]()  # simulate batch.py invoking it after a chunk
        return []

    monkeypatch.setattr(discover_and_process, "run_batch", _fake_run_batch)

    discover_and_process.main()

    assert renew_calls == [("folder-id", "the-token")]


def test_main_aborts_if_lock_renewal_fails_mid_run(monkeypatch):
    monkeypatch.setattr(discover_and_process.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(discover_and_process.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: "the-token")
    monkeypatch.setattr(
        discover_and_process, "discover_and_enqueue", lambda folder_id: DiscoveryReport(0, 0, 0, 0, 0, [])
    )
    monkeypatch.setattr(discover_and_process.job_lock, "renew_lock", lambda folder_id, token: False)

    release_calls = []
    monkeypatch.setattr(
        discover_and_process.job_lock, "release_lock", lambda folder_id, token: release_calls.append(token)
    )

    def _fake_run_batch(**kwargs):
        kwargs["on_progress"]()  # renewal fails - should raise and stop the run
        return []

    monkeypatch.setattr(discover_and_process, "run_batch", _fake_run_batch)

    with pytest.raises(RuntimeError, match="Lost the discovery lock"):
        discover_and_process.main()

    # The lock must still be released (best-effort) even though the run aborted.
    assert release_calls == ["the-token"]


def test_main_passes_a_lock_renewing_callback_to_summarize_eligible_too(monkeypatch):
    """Issue #7 requires the lock to be renewed during model calls as well
    as transcript processing - summarize_eligible() gets the same callback
    run_batch() does."""
    monkeypatch.setattr(discover_and_process.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(discover_and_process.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: "the-token")
    monkeypatch.setattr(
        discover_and_process, "discover_and_enqueue", lambda folder_id: DiscoveryReport(0, 0, 0, 0, 0, [])
    )
    monkeypatch.setattr(discover_and_process, "run_batch", lambda **kwargs: [])
    monkeypatch.setattr(discover_and_process.job_lock, "release_lock", lambda folder_id, token: None)

    renew_calls = []
    monkeypatch.setattr(
        discover_and_process.job_lock,
        "renew_lock",
        lambda folder_id, token: renew_calls.append((folder_id, token)) or True,
    )

    def _fake_summarize_eligible(folder_id, **kwargs):
        kwargs["on_progress"]()  # simulate summary_store.py invoking it after a video
        return _EMPTY_SUMMARY_REPORT

    monkeypatch.setattr(discover_and_process, "summarize_eligible", _fake_summarize_eligible)

    discover_and_process.main()

    assert renew_calls == [("folder-id", "the-token")]


def test_main_releases_lock_even_if_summarize_eligible_raises(monkeypatch):
    monkeypatch.setattr(discover_and_process.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(discover_and_process.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: "the-token")
    monkeypatch.setattr(
        discover_and_process, "discover_and_enqueue", lambda folder_id: DiscoveryReport(0, 0, 0, 0, 0, [])
    )
    monkeypatch.setattr(discover_and_process, "run_batch", lambda **kwargs: [])

    def _raise(folder_id, **kwargs):
        raise RuntimeError("unexpected summarization stage failure")

    monkeypatch.setattr(discover_and_process, "summarize_eligible", _raise)

    release_calls = []
    monkeypatch.setattr(
        discover_and_process.job_lock, "release_lock", lambda folder_id, token: release_calls.append(token)
    )

    with pytest.raises(RuntimeError, match="unexpected summarization"):
        discover_and_process.main()

    assert release_calls == ["the-token"]
