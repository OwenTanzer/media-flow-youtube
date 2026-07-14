import pytest

from app.channel_store import Channel
from app.discovery import DiscoveryReport
from vidproc import admin


def test_check_admin_token_accepts_matching_token():
    assert admin.check_admin_token("the-secret", "the-secret") is True


def test_check_admin_token_rejects_wrong_token():
    assert admin.check_admin_token("wrong", "the-secret") is False


def test_check_admin_token_rejects_when_nothing_configured():
    """The panel must never unlock just because someone typed an empty
    string against an empty/unset configured token."""
    assert admin.check_admin_token("", None) is False
    assert admin.check_admin_token("", "") is False


def _stub(monkeypatch, *, existing, backfill_report=None):
    monkeypatch.setattr(admin.channel_store, "read_channels", lambda folder_id: existing)
    written = {}
    monkeypatch.setattr(
        admin.channel_store, "write_channels", lambda folder_id, channels: written.setdefault("channels", channels)
    )
    monkeypatch.setattr(
        admin.discovery,
        "backfill_new_channels",
        lambda folder_id: backfill_report or DiscoveryReport(1, 1, 3, 2, 0, []),
    )
    return written


def test_add_channel_and_backfill_writes_the_new_channel_and_runs_backfill(monkeypatch):
    written = _stub(monkeypatch, existing=[Channel("UC_old", "Old Channel")])

    report = admin.add_channel_and_backfill(
        "folder-id", channel_id="UC_new", name="New Channel", enabled=True, group="Google", languages=["en", "es"]
    )

    assert written["channels"] == [
        Channel("UC_old", "Old Channel"),
        Channel("UC_new", "New Channel", True, ["en", "es"], "Google"),
    ]
    assert report.newly_queued == 2


def test_add_channel_and_backfill_strips_whitespace(monkeypatch):
    written = _stub(monkeypatch, existing=[])

    admin.add_channel_and_backfill(
        "folder-id", channel_id="  UC_new  ", name="  New Channel  ", group="  ", languages=["  en  ", "  "]
    )

    added = written["channels"][0]
    assert added.channel_id == "UC_new"
    assert added.name == "New Channel"
    assert added.group is None
    assert added.languages == ["en"]


def test_add_channel_and_backfill_defaults_name_to_channel_id_when_blank(monkeypatch):
    written = _stub(monkeypatch, existing=[])

    admin.add_channel_and_backfill("folder-id", channel_id="UC_new", name="   ")

    assert written["channels"][0].name == "UC_new"


def test_add_channel_and_backfill_rejects_blank_channel_id(monkeypatch):
    _stub(monkeypatch, existing=[])

    with pytest.raises(ValueError, match="Channel ID is required"):
        admin.add_channel_and_backfill("folder-id", channel_id="   ", name="Anything")


def test_add_channel_and_backfill_rejects_a_duplicate_channel_id(monkeypatch):
    written = _stub(monkeypatch, existing=[Channel("UC_existing", "Existing Channel")])

    with pytest.raises(admin.ChannelAlreadyExistsError, match="UC_existing"):
        admin.add_channel_and_backfill("folder-id", channel_id="UC_existing", name="Duplicate")

    assert "channels" not in written
