import discover_and_process
from app.discovery import DiscoveryReport


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
    monkeypatch.setattr(discover_and_process, "run_batch", lambda: order.append("run_batch") or [])
    release_tokens = []
    monkeypatch.setattr(
        discover_and_process.job_lock,
        "release_lock",
        lambda folder_id, token: (release_tokens.append(token), order.append("release")),
    )

    exit_code = discover_and_process.main()

    assert exit_code == 0
    assert order == ["discover", "run_batch", "release"]
    assert release_tokens == ["the-token"]
