#!/usr/bin/env python3
"""Streamlit dashboard over the transcript-insight archive (issue #8).

Run locally with: streamlit run vidproc_app.py

Deliberately public/unauthenticated - unlike the FastAPI service's
X-API-Key gate, this app has no login, since it's meant to be a
publicly-viewable dashboard at vidproc.moopertonic.net. It fails safe to
a generic "temporarily unavailable" message (no stack trace, no
credential/Drive detail, no internal URLs) if its own backing
configuration or Drive access is broken - see render_unavailable_state()
below.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import streamlit as st

from app.config import ConfigError, settings
from app.insights_store import InsightsSnapshot, load_snapshot
from vidproc.admin import (
    NEW_GROUP_OPTION,
    ChannelAlreadyExistsError,
    add_channel_and_backfill,
    admin_flash_for,
    check_admin_token,
    resolve_group_selection,
)
from vidproc.render import render_detail, render_empty_state, render_feed_card, render_notice
from vidproc.state import (
    channel_filter_options,
    filter_videos,
    groups_for_channels,
    sorted_feed,
    validate_channel_selection,
)
from vidproc.styling import CHROME_CSS

logger = logging.getLogger("media_flow.vidproc_app")

CACHE_TTL_SECONDS = int(os.environ.get("VIDPROC_CACHE_TTL_SECONDS", "300"))
MIN_REFRESH_INTERVAL_SECONDS = int(os.environ.get("VIDPROC_MIN_REFRESH_INTERVAL_SECONDS", "60"))

# st.cache_data is shared across every concurrent Streamlit session in this
# process, not per-viewer - the Refresh button below therefore can't just
# clear it on every click, since this app is public/unauthenticated (see
# module docstring) and an unauthenticated visitor could otherwise mash
# Refresh to force a full Drive read for every other concurrent viewer.
# This tracks the last clear process-wide (not in st.session_state, which
# is per-viewer) so the rate limit actually holds across sessions.
_refresh_lock = threading.Lock()
_last_cache_clear_at = 0.0


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Loading video insights...")
def _load_snapshot_cached(folder_id: str) -> InsightsSnapshot:
    return load_snapshot(folder_id)


def _try_clear_snapshot_cache() -> bool:
    """Clears the shared snapshot cache if MIN_REFRESH_INTERVAL_SECONDS has
    elapsed since the last clear (by any viewer); returns whether it did."""
    global _last_cache_clear_at
    now = time.monotonic()
    with _refresh_lock:
        if now - _last_cache_clear_at < MIN_REFRESH_INTERVAL_SECONDS:
            return False
        _last_cache_clear_at = now
    _load_snapshot_cached.clear()
    return True


def render_unavailable_state() -> None:
    # Does not call st.set_page_config() - main() already calls it exactly
    # once, unconditionally, before either try/except that can reach this
    # function. Streamlit raises if set_page_config() runs twice in one
    # script execution, which would otherwise turn the snapshot-load
    # failure path below into an unhandled exception of its own - exactly
    # what that path exists to prevent.
    st.markdown(
        """<div style="max-width:640px;margin:80px auto;text-align:center;font-family:'Crimson Text',Georgia,serif">
<h2>This dashboard is temporarily unavailable.</h2>
<p style="color:#999">Please check back shortly.</p>
</div>""",
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    st.session_state.setdefault("active_channel_selection", {})  # group -> list[channel_id]
    # Keyed per group, not a single shared value - every tab's body executes
    # on every rerun (Streamlit doesn't lazily skip inactive tabs), so a
    # single shared selection would get cleared by whichever other group's
    # render happens to run afterward and finds the selected video out of
    # its own scope.
    st.session_state.setdefault("selected_video_id_by_group", {})  # group -> video_id
    st.session_state.setdefault("show_minor_points", True)
    # Resets on every new Streamlit session (e.g. a hard page reload opens
    # a fresh session) - there's no persistent auth token/cookie, so the
    # admin token must be re-entered each session. See vidproc/admin.py.
    st.session_state.setdefault("admin_authenticated", False)


def render_header(snapshot: InsightsSnapshot) -> None:
    col1, col2 = st.columns([5, 2])
    with col1:
        st.markdown("<h1 style='font-family:Oxanium,monospace;font-size:1.6em'>VIDEO INSIGHTS</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"<div class='vidproc-meta-text' style='text-align:right;padding-top:10px'>"
            f"updated {snapshot.generated_at.strftime('%b %d, %H:%M UTC')}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Refresh", use_container_width=True):
            if _try_clear_snapshot_cache():
                st.rerun()
            else:
                st.toast("Refresh was rate-limited - already refreshed recently.")

    notices = []
    if snapshot.pending_count:
        notices.append(f"{snapshot.pending_count} video(s) pending summarization.")
    notices.extend(snapshot.load_errors)
    for notice in notices:
        render_notice(notice)


def render_group(group: str, snapshot: InsightsSnapshot) -> None:
    videos_in_group = [v for v in snapshot.videos if v.group == group]
    options = channel_filter_options(snapshot.channels, group, videos_in_group)
    available_ids = [channel_id for channel_id, _ in options]

    per_group_selection = st.session_state.active_channel_selection.setdefault(group, [])
    per_group_selection[:] = validate_channel_selection(per_group_selection, available_ids)

    all_channels = st.checkbox("All channels", value=not per_group_selection, key=f"all-channels-{group}")
    selected_ids: list[str] = []
    if not all_channels and options:
        labels_by_id = dict(options)
        chosen_labels = st.multiselect(
            "Channels",
            options=[label for _, label in options],
            default=[labels_by_id[c] for c in per_group_selection if c in labels_by_id],
            key=f"channel-multiselect-{group}",
            label_visibility="collapsed",
        )
        id_by_label = {label: channel_id for channel_id, label in options}
        selected_ids = [id_by_label[label] for label in chosen_labels]
        st.session_state.active_channel_selection[group] = selected_ids
    else:
        st.session_state.active_channel_selection[group] = []

    scoped_videos = sorted_feed(filter_videos(snapshot.videos, group, selected_ids))

    selected_video_id = st.session_state.selected_video_id_by_group.get(group)
    if selected_video_id is not None:
        selected = next((v for v in scoped_videos if v.video_id == selected_video_id), None)
        if selected is not None:
            if st.button("← Back to feed", key=f"back-{group}"):
                st.session_state.selected_video_id_by_group[group] = None
                st.rerun()
            st.session_state.show_minor_points = st.checkbox(
                "Show minor points", value=st.session_state.show_minor_points, key=f"show-minor-{group}"
            )
            render_detail(selected, show_minor=st.session_state.show_minor_points)
            return
        # The previously selected video fell out of this group's own scope
        # (filter/group changed) - fall through and show the feed instead of
        # a stale detail view.
        st.session_state.selected_video_id_by_group[group] = None

    if not scoped_videos:
        render_empty_state("No insights yet for this selection.")
        return

    for video in scoped_videos:
        if render_feed_card(video, key_prefix=f"card-{group}"):
            st.session_state.selected_video_id_by_group[group] = video.video_id
            st.rerun()


def render_admin_panel(folder_id: str, channels: list) -> None:
    """Password(-token)-gated panel for adding a channel without editing
    channels.json directly in Drive. Only reachable at all if
    VIDPROC_ADMIN_TOKEN is configured - see main()'s tab list below and
    app/config.py. `channels` (the current snapshot's channel list) is
    needed to populate the group selectbox below with every existing
    group - see NEW_GROUP_OPTION."""

    if not st.session_state.admin_authenticated:
        st.markdown(
            "<div class='vidproc-meta-text'>Enter the admin token to manage channels.</div>",
            unsafe_allow_html=True,
        )
        token_input = st.text_input("Admin token", type="password", key="admin-token-input")
        if st.button("Unlock", key="admin-unlock"):
            if check_admin_token(token_input, settings.vidproc_admin_token):
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect token.")
        return

    if st.button("Lock", key="admin-lock"):
        st.session_state.admin_authenticated = False
        st.rerun()

    # Set by a prior "Add channel" click, right before its own st.rerun()
    # below - without stashing it in session_state across that rerun, the
    # result of a successful add would flash and vanish immediately
    # instead of actually being visible on the page the rerun lands on.
    flash = st.session_state.pop("admin_flash", None)
    if flash is not None:
        level, message = flash
        getattr(st, level)(message)

    st.markdown("<div class='vidproc-meta-text'>Add a channel</div>", unsafe_allow_html=True)
    channel_id = st.text_input("Channel ID (UC...)", key="admin-channel-id")
    name = st.text_input("Display name", key="admin-channel-name")
    enabled = st.checkbox("Enabled", value=True, key="admin-channel-enabled")

    # Existing groups only, plus an explicit "create new" option - picking
    # an existing group can never spawn a new tab, so a new one only ever
    # gets created as a conscious choice, not via a free-text typo of an
    # existing group's name.
    group_choice = st.selectbox(
        "Group", options=[*groups_for_channels(channels), NEW_GROUP_OPTION], key="admin-channel-group-choice"
    )
    new_group_name = ""
    if group_choice == NEW_GROUP_OPTION:
        new_group_name = st.text_input("New group name", key="admin-channel-new-group")

    languages_raw = st.text_input("Languages, comma-separated (optional)", key="admin-channel-languages")

    if st.button("Add channel", key="admin-add-channel"):
        languages = languages_raw.split(",") if languages_raw.strip() else None
        try:
            group = resolve_group_selection(group_choice, new_group_name)
            result = add_channel_and_backfill(
                folder_id,
                channel_id=channel_id,
                name=name,
                enabled=enabled,
                group=group,
                languages=languages,
            )
        except (ChannelAlreadyExistsError, ValueError) as exc:
            st.error(str(exc))
        except Exception:  # noqa: BLE001
            # Same public-boundary principle as main()'s snapshot load below -
            # log the real exception server-side, show only a generic message.
            logger.exception("Failed to add channel via admin panel")
            st.error("Something went wrong adding the channel - check the server logs.")
        else:
            st.session_state.admin_flash = admin_flash_for(result)
            _load_snapshot_cached.clear()
            st.rerun()


def main() -> None:
    # Called exactly once, unconditionally, before anything else - both
    # ConfigError below and a later snapshot-load failure route through
    # render_unavailable_state(), which relies on this having already run
    # (see its docstring/comment).
    st.set_page_config(page_title="Video Insights", layout="wide")
    st.markdown(CHROME_CSS, unsafe_allow_html=True)

    try:
        folder_id = settings.require_drive_folder_id()
        settings.require_oauth_credentials()
    except ConfigError:
        render_unavailable_state()
        return

    _init_session_state()

    try:
        snapshot = _load_snapshot_cached(folder_id)
    except Exception:  # noqa: BLE001
        # This is the public application boundary - a real Drive/credential
        # failure (OAuth refresh error, HttpError, timeout, malformed
        # response) must never leak a stack trace or internal detail to an
        # unauthenticated visitor. Log the full exception server-side and
        # render only the generic unavailable state, exactly like the
        # ConfigError case above.
        logger.exception("Failed to load dashboard snapshot")
        render_unavailable_state()
        return

    render_header(snapshot)

    groups = groups_for_channels(snapshot.channels)
    show_admin = bool(settings.vidproc_admin_token)
    tabs = st.tabs([*groups, "Admin"] if show_admin else groups)
    for tab, group in zip(tabs, groups):
        with tab:
            render_group(group, snapshot)
    if show_admin:
        with tabs[-1]:
            render_admin_panel(folder_id, snapshot.channels)


if __name__ == "__main__":
    main()
