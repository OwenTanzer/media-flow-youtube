"""Pure, Streamlit-independent helper functions for the dashboard's group
tabs and channel filter (issue #8). Deliberately has no Streamlit import
and no session_state access - vidproc_app.py owns st.session_state and
calls these as plain functions, so the filtering/sorting/reset logic stays
unit-testable without a running Streamlit session."""

from __future__ import annotations

from datetime import datetime, timezone

from app.channel_store import Channel
from app.insights_store import DEFAULT_GROUP, UNASSIGNED_CHANNEL_LABEL, VideoInsight, resolve_group

# Sentinel channel_id used in the Level-2 filter for videos whose
# channel_id doesn't resolve to any currently configured channel (missing,
# or belonging to a channel later removed from the registry). Not a real
# channels.json channel_id, so it can never collide with one.
UNASSIGNED_CHANNEL_ID = "__unassigned__"

# datetime.min is naive and can't be compared against aware datetimes in
# the sort key below - this is the aware equivalent, used only to give
# every video a comparable value regardless of whether it has a real date.
_MIN_AWARE_DATETIME = datetime.min.replace(tzinfo=timezone.utc)


def groups_for_channels(channels: list[Channel]) -> list[str]:
    """Every distinct group present among the configured channels, plus
    DEFAULT_GROUP unconditionally (so Finance still gets a tab even if
    every configured channel happens to be Google, since orphaned/
    unmatched videos always fall back to it). Sorted for a stable tab
    order; a real deployment can still control ordering by choosing group
    names, but this repo doesn't need anything fancier for two groups."""

    return sorted({resolve_group(c) for c in channels} | {DEFAULT_GROUP})


def channels_in_group(channels: list[Channel], group: str) -> list[Channel]:
    return [c for c in channels if resolve_group(c) == group]


def validate_channel_selection(selected_channel_ids: list[str], available_channel_ids: list[str]) -> list[str]:
    """Drops any selected ID no longer valid for the active group/channel
    registry (e.g. channels.json was edited between sessions, or the
    group tab just changed). available_channel_ids should include
    UNASSIGNED_CHANNEL_ID when that pseudo-channel is present in the
    active group. Order of the input selection is preserved."""

    available = set(available_channel_ids)
    return [c for c in selected_channel_ids if c in available]


def filter_videos(
    videos: list[VideoInsight],
    group: str,
    selected_channel_ids: list[str] | None,
) -> list[VideoInsight]:
    """selected_channel_ids=None (or empty) means "All channels" - no
    channel-level filtering, only the group scope applies. A non-empty
    list narrows to just those channels (real channel_ids, plus
    UNASSIGNED_CHANNEL_ID for orphaned videos when selected)."""

    in_group = [v for v in videos if v.group == group]
    if not selected_channel_ids:
        return in_group
    selected = set(selected_channel_ids)
    return [v for v in in_group if (v.channel_id or UNASSIGNED_CHANNEL_ID) in selected]


def feed_sort_key(video: VideoInsight) -> tuple[bool, datetime]:
    """Descending order via sorted(..., reverse=True): videos with a real
    video_published_at sort first (most recent first); videos lacking one
    sort after all dated videos, ordered among themselves by the summary
    artifact's own generated_at - the only other timestamp guaranteed
    present - as the defined fallback the issue calls for."""

    published = video.video_published_at
    if published is not None:
        return (True, published)
    return (False, video.generated_at or _MIN_AWARE_DATETIME)


def sorted_feed(videos: list[VideoInsight]) -> list[VideoInsight]:
    return sorted(videos, key=feed_sort_key, reverse=True)


def channel_filter_options(channels: list[Channel], group: str, videos_in_group: list[VideoInsight]) -> list[tuple[str, str]]:
    """(channel_id, display_label) pairs for the Level-2 filter, scoped to
    the active group. Includes UNASSIGNED_CHANNEL_ID only when at least
    one video in this group actually has no resolvable channel - no point
    offering an always-empty filter option otherwise."""

    options = [(c.channel_id, c.name) for c in channels_in_group(channels, group)]
    has_unassigned = any(v.channel_id is None or v.channel_name is None for v in videos_in_group)
    if has_unassigned:
        options.append((UNASSIGNED_CHANNEL_ID, UNASSIGNED_CHANNEL_LABEL))
    return options
