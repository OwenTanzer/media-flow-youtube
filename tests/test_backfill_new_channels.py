import backfill_new_channels
from app.discovery import DiscoveryReport


def test_main_returns_1_when_drive_folder_id_missing(monkeypatch):
    monkeypatch.setattr(backfill_new_channels.settings, "drive_folder_id", None)
    assert backfill_new_channels.main() == 1


def test_main_exits_without_acting_when_lock_is_held(monkeypatch):
    monkeypatch.setattr(backfill_new_channels.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(backfill_new_channels.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: None)

    called = []
    monkeypatch.setattr(backfill_new_channels, "backfill_new_channels", lambda folder_id: called.append(1))
    release_calls = []
    monkeypatch.setattr(
        backfill_new_channels.job_lock, "release_lock", lambda folder_id, token: release_calls.append(1)
    )

    exit_code = backfill_new_channels.main()

    assert exit_code == 1
    assert called == []
    # The lock was never acquired, so this run must not release someone else's lock,
    # and it must not have blocked/waited for it either.
    assert release_calls == []


def test_main_shares_discover_and_processs_lock_and_ttl(monkeypatch):
    """Regression test for the review finding: a distinct lock only
    serializes this script against itself, not against
    discover_and_process.py's concurrent read-modify-write of the same
    queue.json. Must use the same LOCK_FILENAME (job_lock's default) and
    the same DISCOVERY_LOCK_TTL_SECONDS setting discover_and_process.py
    uses, not a lock/TTL of its own."""
    monkeypatch.setattr(backfill_new_channels.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(backfill_new_channels.settings, "discovery_lock_ttl_seconds", 1800)
    acquire_calls = []
    monkeypatch.setattr(
        backfill_new_channels.job_lock,
        "acquire_lock",
        lambda folder_id, ttl_seconds: acquire_calls.append((folder_id, ttl_seconds)) or "the-token",
    )
    monkeypatch.setattr(
        backfill_new_channels, "backfill_new_channels", lambda folder_id: DiscoveryReport(0, 0, 0, 0, 0, [])
    )
    monkeypatch.setattr(backfill_new_channels.job_lock, "release_lock", lambda folder_id, token: None)

    backfill_new_channels.main()

    assert acquire_calls == [("folder-id", 1800)]


def test_main_runs_backfill_and_releases_lock_when_acquired(monkeypatch):
    monkeypatch.setattr(backfill_new_channels.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(backfill_new_channels.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: "the-token")

    order = []
    monkeypatch.setattr(
        backfill_new_channels,
        "backfill_new_channels",
        lambda folder_id: order.append("backfill") or DiscoveryReport(1, 1, 2, 0, 0, []),
    )
    release_tokens = []
    monkeypatch.setattr(
        backfill_new_channels.job_lock,
        "release_lock",
        lambda folder_id, token: (release_tokens.append(token), order.append("release")),
    )

    exit_code = backfill_new_channels.main()

    assert exit_code == 0
    assert order == ["backfill", "release"]
    assert release_tokens == ["the-token"]


def test_main_releases_lock_even_if_backfill_raises(monkeypatch):
    monkeypatch.setattr(backfill_new_channels.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(backfill_new_channels.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: "the-token")

    def _raise(folder_id):
        raise RuntimeError("unexpected backfill failure")

    monkeypatch.setattr(backfill_new_channels, "backfill_new_channels", _raise)

    release_calls = []
    monkeypatch.setattr(
        backfill_new_channels.job_lock, "release_lock", lambda folder_id, token: release_calls.append(token)
    )

    try:
        backfill_new_channels.main()
        raised = False
    except RuntimeError:
        raised = True

    assert raised
    assert release_calls == ["the-token"]


def test_main_is_a_noop_when_no_channels_need_backfilling(monkeypatch):
    monkeypatch.setattr(backfill_new_channels.settings, "drive_folder_id", "folder-id")
    monkeypatch.setattr(backfill_new_channels.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: "the-token")
    monkeypatch.setattr(
        backfill_new_channels, "backfill_new_channels", lambda folder_id: DiscoveryReport(0, 0, 0, 0, 0, [])
    )
    monkeypatch.setattr(backfill_new_channels.job_lock, "release_lock", lambda folder_id, token: None)

    assert backfill_new_channels.main() == 0
