import pytest
import requests

from app.channel_store import Channel
from app.discovery import DiscoveryReport
from app.group_store import Group
from vidproc import admin

VALID_CHANNEL_ID = "UC1234567890123456789012"  # UC + 22 chars


def _stub_groups(monkeypatch, *, existing=()):
    monkeypatch.setattr(admin.group_store, "read_groups", lambda folder_id: list(existing))
    written = {}
    monkeypatch.setattr(
        admin.group_store, "write_groups", lambda folder_id, groups: written.setdefault("groups", groups)
    )
    return written


def test_check_admin_token_accepts_matching_token():
    assert admin.check_admin_token("the-secret", "the-secret") is True


def test_check_admin_token_rejects_wrong_token():
    assert admin.check_admin_token("wrong", "the-secret") is False


def test_check_admin_token_rejects_when_nothing_configured():
    """The panel must never unlock just because someone typed an empty
    string against an empty/unset configured token."""
    assert admin.check_admin_token("", None) is False
    assert admin.check_admin_token("", "") is False


def test_resolve_group_selection_returns_an_existing_group_verbatim(monkeypatch):
    written = _stub_groups(monkeypatch)
    assert admin.resolve_group_selection("folder-id", "Google", "", []) == "Google"
    assert "groups" not in written  # no Drive write for an existing group


def test_resolve_group_selection_creates_a_new_group_when_explicitly_chosen(monkeypatch):
    written = _stub_groups(monkeypatch)
    result = admin.resolve_group_selection("folder-id", admin.NEW_GROUP_OPTION, "Crypto", ["Market Update"])
    assert result == "Crypto"
    assert written["groups"] == [Group(name="Crypto", video_types=["Market Update"])]


def test_create_group_strips_name_and_video_types(monkeypatch):
    written = _stub_groups(monkeypatch)
    group = admin.create_group("folder-id", "  Crypto  ", ["  Market Update  ", "  "])
    assert group == Group(name="Crypto", video_types=["Market Update"])
    assert written["groups"] == [group]


def test_create_group_rejects_a_blank_name(monkeypatch):
    _stub_groups(monkeypatch)
    with pytest.raises(ValueError, match="Group name is required"):
        admin.create_group("folder-id", "   ", ["Tutorial"])


def test_create_group_rejects_no_non_blank_video_types(monkeypatch):
    """Regression test for the whole point of this function: a new group
    with no video types isn't a valid choice - it must not silently fall
    back to the finance-flavored default categories."""
    _stub_groups(monkeypatch)
    with pytest.raises(ValueError, match="At least one video type is required"):
        admin.create_group("folder-id", "Crypto", ["  ", ""])


def test_create_group_rejects_an_already_registered_name(monkeypatch):
    _stub_groups(monkeypatch, existing=[Group(name="Google", video_types=["Tutorial"])])
    with pytest.raises(ValueError, match="'Google' is already registered"):
        admin.create_group("folder-id", "Google", ["Something"])


def test_create_group_preserves_existing_groups(monkeypatch):
    existing_group = Group(name="Google", video_types=["Tutorial"])
    written = _stub_groups(monkeypatch, existing=[existing_group])
    admin.create_group("folder-id", "Crypto", ["Market Update"])
    assert written["groups"] == [existing_group, Group(name="Crypto", video_types=["Market Update"])]


def _stub(monkeypatch, *, existing, lock_token="the-token", backfill_report=None, feed_ok=True):
    monkeypatch.setattr(admin.channel_store, "read_channels", lambda folder_id: existing)
    written = {}
    monkeypatch.setattr(
        admin.channel_store, "write_channels", lambda folder_id, channels: written.setdefault("channels", channels)
    )
    if feed_ok:
        monkeypatch.setattr(admin.discovery, "fetch_channel_feed", lambda channel_id: [])
    else:
        def _raise(channel_id):
            raise requests.RequestException("feed unavailable")

        monkeypatch.setattr(admin.discovery, "fetch_channel_feed", _raise)
    monkeypatch.setattr(admin.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: lock_token)
    release_calls = []
    monkeypatch.setattr(admin.job_lock, "release_lock", lambda folder_id, token: release_calls.append(token))
    monkeypatch.setattr(
        admin.discovery,
        "backfill_new_channels",
        lambda folder_id: backfill_report or DiscoveryReport(1, 1, 3, 2, 0, []),
    )
    return written, release_calls


def test_add_channel_and_backfill_writes_the_new_channel_and_runs_backfill(monkeypatch):
    written, release_calls = _stub(monkeypatch, existing=[Channel("UC_old234567890123456789", "Old Channel")])

    result = admin.add_channel_and_backfill(
        "folder-id", channel_id=VALID_CHANNEL_ID, name="New Channel", enabled=True, group="Google", languages=["en", "es"]
    )

    assert written["channels"] == [
        Channel("UC_old234567890123456789", "Old Channel"),
        Channel(VALID_CHANNEL_ID, "New Channel", True, ["en", "es"], "Google"),
    ]
    assert result.backfill_report.newly_queued == 2
    assert result.backfill_deferred is False
    assert result.backfill_error is None
    assert release_calls == ["the-token"]


def test_add_channel_and_backfill_strips_whitespace(monkeypatch):
    written, _ = _stub(monkeypatch, existing=[])

    admin.add_channel_and_backfill(
        "folder-id", channel_id=f"  {VALID_CHANNEL_ID}  ", name="  New Channel  ", group="  ", languages=["  en  ", "  "]
    )

    added = written["channels"][0]
    assert added.channel_id == VALID_CHANNEL_ID
    assert added.name == "New Channel"
    assert added.group is None
    assert added.languages == ["en"]


def test_add_channel_and_backfill_defaults_name_to_channel_id_when_blank(monkeypatch):
    written, _ = _stub(monkeypatch, existing=[])

    admin.add_channel_and_backfill("folder-id", channel_id=VALID_CHANNEL_ID, name="   ")

    assert written["channels"][0].name == VALID_CHANNEL_ID


def test_add_channel_and_backfill_rejects_blank_channel_id(monkeypatch):
    _stub(monkeypatch, existing=[])

    with pytest.raises(ValueError, match="Channel ID is required"):
        admin.add_channel_and_backfill("folder-id", channel_id="   ", name="Anything")


@pytest.mark.parametrize("bad_id", ["not-a-channel-id", "UCtooshort", "UC" + "x" * 21, "UC" + "x" * 23])
def test_add_channel_and_backfill_rejects_malformed_channel_id(monkeypatch, bad_id):
    """Regression test for the review finding: a typo was previously
    written to channels.json before YouTube ever got a chance to reject
    it. A malformed ID is now rejected before any write happens."""
    written, _ = _stub(monkeypatch, existing=[])
    fetch_calls = []
    monkeypatch.setattr(admin.discovery, "fetch_channel_feed", lambda channel_id: fetch_calls.append(channel_id))

    with pytest.raises(ValueError, match="doesn't look like a YouTube channel ID"):
        admin.add_channel_and_backfill("folder-id", channel_id=bad_id, name="Anything")

    assert "channels" not in written
    assert fetch_calls == []  # never even attempted the preflight fetch


def test_add_channel_and_backfill_rejects_a_channel_whose_feed_cant_be_fetched(monkeypatch):
    """Regression test for the review finding: a well-formed but
    nonexistent/wrong channel ID must be rejected by the RSS preflight
    fetch *before* channels.json is written, not discovered later."""
    written, _ = _stub(monkeypatch, existing=[], feed_ok=False)

    with pytest.raises(ValueError, match="Could not fetch this channel's RSS feed"):
        admin.add_channel_and_backfill("folder-id", channel_id=VALID_CHANNEL_ID, name="Anything")

    assert "channels" not in written


def test_add_channel_and_backfill_rejects_a_duplicate_channel_id(monkeypatch):
    written, _ = _stub(monkeypatch, existing=[Channel(VALID_CHANNEL_ID, "Existing Channel")])

    with pytest.raises(admin.ChannelAlreadyExistsError, match=VALID_CHANNEL_ID):
        admin.add_channel_and_backfill("folder-id", channel_id=VALID_CHANNEL_ID, name="Duplicate")

    assert "channels" not in written


def test_add_channel_and_backfill_defers_when_discovery_lock_is_held(monkeypatch):
    """Regression test for the review finding: backfill must share
    discover_and_process.py's lock (not a lock of its own), and must
    never block waiting for it - if it's held, the channel is still
    added immediately and the backfill is simply deferred."""
    written, release_calls = _stub(monkeypatch, existing=[], lock_token=None)
    backfill_calls = []
    monkeypatch.setattr(admin.discovery, "backfill_new_channels", lambda folder_id: backfill_calls.append(1))

    result = admin.add_channel_and_backfill("folder-id", channel_id=VALID_CHANNEL_ID, name="New Channel")

    assert written["channels"] == [Channel(VALID_CHANNEL_ID, "New Channel")]
    assert result.backfill_deferred is True
    assert result.backfill_report is None
    assert result.backfill_error is None
    assert backfill_calls == []  # never attempted - must not wait for the lock either
    assert release_calls == []  # never acquired, so nothing to release


def test_add_channel_and_backfill_skips_backfill_entirely_for_a_disabled_channel(monkeypatch):
    """Regression test for the review finding: a disabled channel is
    excluded from find_unbackfilled_channels() (and from discovery's
    normal poll) entirely, so attempting a backfill for it would always
    find nothing - must not even try, let alone report a misleading
    "0 videos queued" success."""
    written, release_calls = _stub(monkeypatch, existing=[])
    acquire_calls = []
    monkeypatch.setattr(admin.job_lock, "acquire_lock", lambda folder_id, ttl_seconds: acquire_calls.append(1))
    backfill_calls = []
    monkeypatch.setattr(admin.discovery, "backfill_new_channels", lambda folder_id: backfill_calls.append(1))

    result = admin.add_channel_and_backfill("folder-id", channel_id=VALID_CHANNEL_ID, name="New Channel", enabled=False)

    assert written["channels"] == [Channel(VALID_CHANNEL_ID, "New Channel", False)]
    assert result.channel.enabled is False
    assert result.backfill_report is None
    assert result.backfill_deferred is False
    assert result.backfill_error is None
    assert acquire_calls == []  # never even tried to acquire the lock
    assert backfill_calls == []
    assert release_calls == []


def test_add_channel_and_backfill_reports_a_backfill_error_without_losing_the_added_channel(monkeypatch):
    """Regression test for the review finding: if backfill itself raises
    after channels.json was already written, the caller must be able to
    tell "channel added, backfill failed" apart from total failure -
    not have the exception escape and make a successful write look like
    it never happened."""
    written, release_calls = _stub(monkeypatch, existing=[])

    def _raise(folder_id):
        raise RuntimeError("Drive read failed")

    monkeypatch.setattr(admin.discovery, "backfill_new_channels", _raise)

    result = admin.add_channel_and_backfill("folder-id", channel_id=VALID_CHANNEL_ID, name="New Channel")

    assert written["channels"] == [Channel(VALID_CHANNEL_ID, "New Channel")]
    assert result.backfill_error == "Drive read failed"
    assert result.backfill_report is None
    assert result.backfill_deferred is False
    assert release_calls == ["the-token"]  # still released despite the failure


def _result(*, deferred=False, error=None, report=None, enabled=True):
    return admin.AddChannelResult(
        channel=Channel(VALID_CHANNEL_ID, "New Channel", enabled),
        backfill_report=report,
        backfill_deferred=deferred,
        backfill_error=error,
    )


def test_admin_flash_for_success():
    level, message = admin.admin_flash_for(_result(report=DiscoveryReport(1, 1, 3, 2, 0, [])))
    assert level == "success"
    assert "2 video(s) queued" in message


def test_admin_flash_for_disabled_channel():
    """Regression test for the review finding: adding a disabled channel
    must not read as "queued"/"next run will pick it up" messaging, since
    neither discovery nor backfill ever considers a disabled channel."""
    level, message = admin.admin_flash_for(_result(enabled=False))
    assert level == "info"
    assert "disabled" in message
    assert "no backfill" in message


def test_admin_flash_for_deferred():
    level, message = admin.admin_flash_for(_result(deferred=True))
    assert level == "info"
    assert "currently running" in message


def test_admin_flash_for_backfill_error():
    level, message = admin.admin_flash_for(_result(error="Drive read failed"))
    assert level == "warning"
    assert "Drive read failed" in message


def test_admin_flash_for_surfaces_this_channels_own_feed_failure():
    """Regression test for the review finding: a feed failure for the
    just-added channel must not be silently reported as a success with
    zero videos queued."""
    report = DiscoveryReport(1, 1, 0, 0, 0, [(VALID_CHANNEL_ID, "feed unavailable")])
    level, message = admin.admin_flash_for(_result(report=report))
    assert level == "warning"
    assert "feed unavailable" in message


def test_admin_flash_for_success_ignores_a_different_channels_feed_failure():
    """A feed failure for some *other* still-unbackfilled channel doesn't
    make the just-added channel's own success message inaccurate."""
    report = DiscoveryReport(2, 2, 3, 2, 0, [("UC_other567890123456789", "feed unavailable")])
    level, message = admin.admin_flash_for(_result(report=report))
    assert level == "success"
