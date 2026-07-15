import json

from app import group_store
from app.group_store import Group


def _with_real_drive(monkeypatch, text):
    monkeypatch.setattr(group_store.settings, "dry_run", False)
    monkeypatch.setattr(group_store.drive, "download_text", lambda folder_id, filename: text)


def test_read_groups_parses_valid_registry(monkeypatch):
    _with_real_drive(
        monkeypatch,
        json.dumps(
            {
                "version": 1,
                "groups": [
                    {"name": "Finance", "video_types": ["Post-Market Update", "Thesis Piece"]},
                    {"name": "Google", "video_types": ["Short Showcase", "Tutorial"]},
                ],
            }
        ),
    )

    groups = group_store.read_groups("folder-id")

    assert groups == [
        Group(name="Finance", video_types=["Post-Market Update", "Thesis Piece"]),
        Group(name="Google", video_types=["Short Showcase", "Tutorial"]),
    ]


def test_read_groups_parses_video_type_descriptions(monkeypatch):
    _with_real_drive(
        monkeypatch,
        json.dumps(
            {
                "groups": [
                    {
                        "name": "Google",
                        "video_types": ["Tutorial", "Short Showcase"],
                        "video_type_descriptions": {"Tutorial": "a how-to walkthrough.", "Bogus": 5},
                    }
                ]
            }
        ),
    )

    groups = group_store.read_groups("folder-id")

    assert groups == [
        Group(
            name="Google",
            video_types=["Tutorial", "Short Showcase"],
            video_type_descriptions={"Tutorial": "a how-to walkthrough."},
        )
    ]


def test_read_groups_missing_file_returns_empty(monkeypatch):
    _with_real_drive(monkeypatch, None)
    assert group_store.read_groups("folder-id") == []


def test_read_groups_malformed_json_returns_empty(monkeypatch):
    _with_real_drive(monkeypatch, "not json")
    assert group_store.read_groups("folder-id") == []


def test_read_groups_missing_groups_key_returns_empty(monkeypatch):
    _with_real_drive(monkeypatch, json.dumps({"version": 1}))
    assert group_store.read_groups("folder-id") == []


def test_read_groups_skips_malformed_entries(monkeypatch):
    _with_real_drive(
        monkeypatch,
        json.dumps(
            {
                "groups": [
                    {"no_name": "oops"},
                    {"name": "   "},
                    {"name": "Google", "video_types": ["Tutorial", "", "  ", 5]},
                    "not-a-dict",
                ]
            }
        ),
    )

    groups = group_store.read_groups("folder-id")

    assert groups == [Group(name="Google", video_types=["Tutorial"])]


def test_read_groups_dry_run_returns_empty_without_touching_drive(monkeypatch):
    monkeypatch.setattr(group_store.settings, "dry_run", True)
    called = []
    monkeypatch.setattr(group_store.drive, "download_text", lambda *a, **k: called.append(1))

    assert group_store.read_groups("folder-id") == []
    assert not called


def _capture_upload(monkeypatch):
    written = {}

    def _upload(folder_id, filename, content, **kwargs):
        written["folder_id"] = folder_id
        written["filename"] = filename
        written["content"] = json.loads(content)

    monkeypatch.setattr(group_store.drive, "upload_text_file", _upload)
    return written


def test_write_groups_round_trips_through_read_groups(monkeypatch):
    written = _capture_upload(monkeypatch)
    groups = [
        Group(name="Finance", video_types=["Thesis Piece"]),
        Group(
            name="Google",
            video_types=["Tutorial"],
            video_type_descriptions={"Tutorial": "a how-to walkthrough."},
        ),
    ]

    group_store.write_groups("folder-id", groups)

    assert written["filename"] == group_store.GROUPS_FILENAME
    assert written["folder_id"] == "folder-id"

    monkeypatch.setattr(group_store.settings, "dry_run", False)
    monkeypatch.setattr(group_store.drive, "download_text", lambda folder_id, filename: json.dumps(written["content"]))
    assert group_store.read_groups("folder-id") == groups


def test_write_groups_omits_absent_descriptions(monkeypatch):
    written = _capture_upload(monkeypatch)
    group_store.write_groups("folder-id", [Group(name="Google", video_types=["Tutorial"])])
    assert "video_type_descriptions" not in written["content"]["groups"][0]


def test_get_video_types_returns_the_matching_groups_list():
    groups = [Group(name="Finance", video_types=["Thesis Piece"]), Group(name="Google", video_types=["Tutorial"])]
    assert group_store.get_video_types(groups, "Google", "Finance") == ["Tutorial"]


def test_get_video_types_falls_back_to_the_default_group():
    groups = [Group(name="Finance", video_types=["Thesis Piece"])]
    assert group_store.get_video_types(groups, "SomeUnconfiguredGroup", "Finance") == ["Thesis Piece"]


def test_get_video_types_falls_back_to_the_hardcoded_default_when_nothing_is_configured():
    """Regression test: a deployment with no groups.json at all (or one
    where neither the resolved group nor the default group has anything
    configured) must behave exactly as it did before groups could
    configure their own video_types."""
    assert group_store.get_video_types([], "Finance", "Finance") == group_store.FALLBACK_VIDEO_TYPES


def test_get_video_types_treats_an_empty_configured_list_as_unconfigured():
    groups = [Group(name="Google", video_types=[])]
    assert group_store.get_video_types(groups, "Google", "Finance") == group_store.FALLBACK_VIDEO_TYPES


def test_get_video_type_descriptions_returns_the_matching_groups_descriptions():
    groups = [
        Group(name="Google", video_types=["Tutorial"], video_type_descriptions={"Tutorial": "a how-to walkthrough."})
    ]
    assert group_store.get_video_type_descriptions(groups, "Google", "Finance") == {"Tutorial": "a how-to walkthrough."}


def test_get_video_type_descriptions_returns_empty_when_group_has_none_configured():
    """A group that configures its own video_types without descriptions
    gets no descriptions at all - not the Finance-flavored ones, which
    would describe the wrong categories entirely."""
    groups = [Group(name="Google", video_types=["Tutorial"])]
    assert group_store.get_video_type_descriptions(groups, "Google", "Finance") == {}


def test_get_video_type_descriptions_falls_back_to_hardcoded_defaults_when_nothing_is_configured():
    assert (
        group_store.get_video_type_descriptions([], "Finance", "Finance")
        == group_store.FALLBACK_VIDEO_TYPE_DESCRIPTIONS
    )


def test_get_video_type_descriptions_falls_back_to_the_default_groups_descriptions():
    groups = [
        Group(name="Finance", video_types=["Thesis Piece"], video_type_descriptions={"Thesis Piece": "a deep dive."})
    ]
    assert group_store.get_video_type_descriptions(groups, "SomeUnconfiguredGroup", "Finance") == {
        "Thesis Piece": "a deep dive."
    }
