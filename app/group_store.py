"""Reads/writes the operator-managed group registry (`groups.json`) in
the Drive folder: each dashboard group's video_type classification
taxonomy, used to build the per-group summarization prompt (see
app/summarize.py, app/summary_store.py). A sibling of
app/channel_store.py's channels.json handling - groups themselves are
still just values on Channel.group (see channel_store.py); this only
stores each group's classification categories."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from . import drive
from .config import settings
from .summarize import FALLBACK_VIDEO_TYPES

logger = logging.getLogger("media_flow.groups")

GROUPS_FILENAME = "groups.json"


@dataclass
class Group:
    name: str
    video_types: list[str]


def _group_to_dict(group: Group) -> dict:
    return {"name": group.name, "video_types": group.video_types}


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
        groups.append(Group(name=entry["name"], video_types=video_types))
    return groups


def get_video_types(groups: list[Group], group_name: str, default_group: str) -> list[str]:
    """Returns group_name's configured video_types - falling back to
    default_group's configured list if group_name isn't registered (or is
    registered with an empty list), and finally to FALLBACK_VIDEO_TYPES if
    neither has anything configured at all."""

    by_name = {g.name: g.video_types for g in groups if g.video_types}
    return by_name.get(group_name) or by_name.get(default_group) or FALLBACK_VIDEO_TYPES
