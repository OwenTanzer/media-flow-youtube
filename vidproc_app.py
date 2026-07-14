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

import os

import streamlit as st

from app.config import ConfigError, settings
from app.insights_store import InsightsSnapshot, load_snapshot
from vidproc.render import render_detail, render_empty_state, render_feed_card, render_notice
from vidproc.state import (
    channel_filter_options,
    filter_videos,
    groups_for_channels,
    sorted_feed,
    validate_channel_selection,
)
from vidproc.styling import CHROME_CSS

CACHE_TTL_SECONDS = int(os.environ.get("VIDPROC_CACHE_TTL_SECONDS", "300"))


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_snapshot_cached(folder_id: str) -> InsightsSnapshot:
    return load_snapshot(folder_id)


def render_unavailable_state() -> None:
    st.set_page_config(page_title="Video Insights", layout="wide")
    st.markdown(CHROME_CSS, unsafe_allow_html=True)
    st.markdown(
        """<div style="max-width:640px;margin:80px auto;text-align:center;font-family:'Crimson Text',Georgia,serif">
<h2>This dashboard is temporarily unavailable.</h2>
<p style="color:#999">Please check back shortly.</p>
</div>""",
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    st.session_state.setdefault("active_channel_selection", {})  # group -> list[channel_id]
    st.session_state.setdefault("selected_video_id", None)
    st.session_state.setdefault("show_minor_points", True)


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
            _load_snapshot_cached.clear()
            st.rerun()

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

    if st.session_state.selected_video_id is not None:
        selected = next((v for v in scoped_videos if v.video_id == st.session_state.selected_video_id), None)
        if selected is not None:
            if st.button("← Back to feed", key=f"back-{group}"):
                st.session_state.selected_video_id = None
                st.rerun()
            st.session_state.show_minor_points = st.checkbox(
                "Show minor points", value=st.session_state.show_minor_points, key=f"show-minor-{group}"
            )
            render_detail(selected, show_minor=st.session_state.show_minor_points)
            return
        # The previously selected video fell out of this scope (filter/group
        # changed) - fall through and show the feed instead of a stale detail view.
        st.session_state.selected_video_id = None

    if not scoped_videos:
        render_empty_state("No insights yet for this selection.")
        return

    for video in scoped_videos:
        if render_feed_card(video, key_prefix=f"card-{group}"):
            st.session_state.selected_video_id = video.video_id
            st.rerun()


def main() -> None:
    try:
        folder_id = settings.require_drive_folder_id()
        settings.require_oauth_credentials()
    except ConfigError:
        render_unavailable_state()
        return

    st.set_page_config(page_title="Video Insights", layout="wide")
    st.markdown(CHROME_CSS, unsafe_allow_html=True)
    _init_session_state()

    try:
        snapshot = _load_snapshot_cached(folder_id)
    except ConfigError:
        render_unavailable_state()
        return

    render_header(snapshot)

    groups = groups_for_channels(snapshot.channels)
    tabs = st.tabs(groups)
    for tab, group in zip(tabs, groups):
        with tab:
            render_group(group, snapshot)


if __name__ == "__main__":
    main()
