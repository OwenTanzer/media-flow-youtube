"""Reads/writes the operator-managed group registry (`groups.json`) in
the Drive folder: each dashboard group's video_type classification
taxonomy, used to build the per-group summarization prompt (see
app/summarize.py, app/summary_store.py). A sibling of
app/channel_store.py's channels.json handling - groups themselves are
still just values on Channel.group (see channel_store.py); this only
stores each group's classification categories."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

from . import drive
from .config import settings
from .summarize import FALLBACK_VIDEO_TYPE_DESCRIPTIONS, FALLBACK_VIDEO_TYPES

logger = logging.getLogger("media_flow.groups")

GROUPS_FILENAME = "groups.json"


@dataclass
class Group:
    name: str
    video_types: list[str]
    # Optional {video_type: description} guidance for the model - not
    # editable from the admin panel's create/edit-group forms (those only
    # take names), but can be hand-edited directly in groups.json, same as
    # e.g. channels.json's "languages" field. A category's classification
    # accuracy is meaningfully better with a real definition (see
    # summarize.FALLBACK_VIDEO_TYPE_DESCRIPTIONS for the original, pre-
    # per-group categories' own descriptions) than a bare name alone -
    # this exists so that quality isn't a required regression for groups
    # that want to invest in it.
    video_type_descriptions: dict[str, str] = field(default_factory=dict)


def _group_to_dict(group: Group) -> dict:
    entry: dict = {"name": group.name, "video_types": group.video_types}
    if group.video_type_descriptions:
        entry["video_type_descriptions"] = group.video_type_descriptions
    return entry


def write_groups(folder_id: str, groups: list[Group]) -> None:
    """Overwrites groups.json with exactly this list, in the same
    {"version": 1, "groups": [...]} shape read_groups() parses. Unlocked
    read-modify-write, same as channel_store.write_channels() - groups.json
    changes rarely enough that this hasn't needed advisory-lock treatment."""

    payload = json.dumps({"version": 1, "groups": [_group_to_dict(g) for g in groups]}, indent=2)
    drive.upload_text_file(folder_id, GROUPS_FILENAME, payload, mime_type="application/json")


def read_groups(folder_id: str) -> list[Group]:
    if settings.dry_run:
        return []

    text = drive.download_text(folder_id, GROUPS_FILENAME)
    if text is None:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("groups.json in folder %s was not valid JSON; treating as empty.", folder_id)
        return []

    raw_groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(raw_groups, list):
        logger.warning("groups.json in folder %s has no \"groups\" list; treating as empty.", folder_id)
        return []

    groups: list[Group] = []
    for entry in raw_groups:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
            logger.warning("Skipping malformed groups.json entry: %r", entry)
            continue
        raw_types = entry.get("video_types")
        video_types = (
            [str(t) for t in raw_types if isinstance(t, str) and t.strip()] if isinstance(raw_types, list) else []
        )
        raw_descriptions = entry.get("video_type_descriptions")
        video_type_descriptions = (
            {str(k): str(v) for k, v in raw_descriptions.items() if isinstance(k, str) and isinstance(v, str)}
            if isinstance(raw_descriptions, dict)
            else {}
        )
        groups.append(Group(name=entry["name"], video_types=video_types, video_type_descriptions=video_type_descriptions))
    return groups


def get_video_types(groups: list[Group], group_name: str, default_group: str) -> list[str]:
    """Returns group_name's configured video_types - falling back to
    default_group's configured list if group_name isn't registered (or is
    registered with an empty list), and finally to FALLBACK_VIDEO_TYPES if
    neither has anything configured at all."""

    by_name = {g.name: g.video_types for g in groups if g.video_types}
    return by_name.get(group_name) or by_name.get(default_group) or FALLBACK_VIDEO_TYPES


def get_video_type_descriptions(groups: list[Group], group_name: str, default_group: str) -> dict[str, str]:
    """Returns group_name's configured video_type_descriptions, with the
    same fallback chain as get_video_types() - falls back to
    FALLBACK_VIDEO_TYPE_DESCRIPTIONS (the original categories' own
    definitions) only when neither group_name nor default_group has
    *any* video_types configured at all (i.e. get_video_types() would
    itself be falling back to FALLBACK_VIDEO_TYPES), so a group that
    configures its own video_types without descriptions gets no
    descriptions at all, rather than the Finance-flavored ones by
    accident."""

    types_by_name = {g.name: g.video_types for g in groups if g.video_types}
    if group_name not in types_by_name and default_group not in types_by_name:
        return FALLBACK_VIDEO_TYPE_DESCRIPTIONS

    descriptions_by_name = {g.name: g.video_type_descriptions for g in groups if g.video_types}
    return descriptions_by_name.get(group_name) or descriptions_by_name.get(default_group) or {}


def video_types_fingerprint(video_types: list[str], video_type_descriptions: dict[str, str]) -> str:
    """A short, stable hash of a group's resolved classification config -
    persisted on the summary artifact and compared as part of
    summary_store.needs_summarization()'s idempotency check, so that
    editing a group's video_types/video_type_descriptions later (see
    vidproc/admin.py's update_group_video_types()) makes every video
    previously summarized under a *different* taxonomy eligible for
    re-summarization again, not just a one-time PROMPT_VERSION bump.

    This also closes the launch-sequencing gap where the scheduled job
    runs before groups.json is seeded: videos summarized under the
    fallback taxonomy in that window get a different fingerprint than
    ones summarized after the real categories are configured, so seeding
    afterward still corrects them on the next run instead of leaving them
    permanently marked "current" under the wrong categories."""

    payload = json.dumps({"video_types": video_types, "video_type_descriptions": video_type_descriptions}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
