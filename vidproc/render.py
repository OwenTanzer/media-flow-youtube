"""Feed-card and detail-view rendering for the dashboard (issue #8).
Left-accent styled st.markdown divs, matching the oil-crisis-dashboard
MediaFlow reference's card pattern - deliberately not st.dataframe/
st.table (a generic grid) or a card-heavy marketing layout."""

from __future__ import annotations

import html

import streamlit as st

from app.insights_store import InsightPoint, VideoInsight

from .styling import accent_color_for


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _fmt_date(video: VideoInsight) -> str:
    if video.video_published_at is not None:
        return video.video_published_at.strftime("%b %d, %Y")
    return "date unknown"


def render_feed_card(video: VideoInsight, *, key_prefix: str) -> bool:
    """Renders one feed headline. Returns True the run a viewer clicks it
    (a full-width styled button doubles as the clickable headline - native
    Streamlit has no single-element clickable card)."""

    color = accent_color_for(video.video_type)
    channel_label = video.channel_name or video.author or "Unknown channel"
    type_label = video.video_type or "Uncategorized"

    st.markdown(
        f"""<div class="vidproc-card" style="border-left-color:{color}">
<span class="vidproc-meta-text">{_esc(_fmt_date(video))} &nbsp;·&nbsp; {_esc(channel_label)} &nbsp;·&nbsp; {_esc(type_label)}</span>
<div class="vidproc-title">{_esc(video.title)}</div>
<div class="vidproc-summary">{_esc(video.summary)}</div>
</div>""",
        unsafe_allow_html=True,
    )
    return st.button("Open insight →", key=f"{key_prefix}-{video.video_id}")


def render_empty_state(message: str) -> None:
    st.markdown(f'<div class="vidproc-main-text" style="color:#999;padding:20px 0">{_esc(message)}</div>', unsafe_allow_html=True)


def render_notice(text: str) -> None:
    st.markdown(f'<div class="vidproc-notice">{_esc(text)}</div>', unsafe_allow_html=True)


def _render_point(video: VideoInsight, point: InsightPoint) -> None:
    minor_class = " minor" if point.importance == "minor" else ""
    badge_class = "major" if point.importance == "major" else "minor"

    timestamp_html = ""
    if point.timestamp_seconds is not None:
        anchored_url = f"{video.url}&t={point.timestamp_seconds}s"
        label = point.timestamp or f"{point.timestamp_seconds}s"
        timestamp_html = f'<a class="vidproc-timestamp-link" href="{_esc(anchored_url)}" target="_blank">▶ {_esc(label)}</a>'

    st.markdown(
        f"""<div style="margin-bottom:10px">
<span class="vidproc-badge {badge_class}">{point.importance}</span>
&nbsp;{timestamp_html}
<div class="vidproc-main-point{minor_class}">{_esc(point.main_point)}</div>
<div class="vidproc-explanation">{_esc(point.explanation)}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_detail(video: VideoInsight, *, show_minor: bool) -> None:
    channel_label = video.channel_name or video.author or "Unknown channel"
    type_label = video.video_type or "Uncategorized"

    st.markdown(
        f"""<div class="vidproc-meta-text">{_esc(_fmt_date(video))} &nbsp;·&nbsp; {_esc(channel_label)} &nbsp;·&nbsp; {_esc(type_label)}</div>
<div class="vidproc-title" style="font-size:1.5em;margin:4px 0">{_esc(video.title)}</div>
<div class="vidproc-summary">{_esc(video.summary)}</div>""",
        unsafe_allow_html=True,
    )

    links = [f'<a href="{_esc(video.url)}" target="_blank">→ source video</a>']
    if video.drive_file_id:
        drive_link = f"https://drive.google.com/file/d/{video.drive_file_id}/view"
        links.append(f'<a href="{_esc(drive_link)}" target="_blank">→ transcript (Drive)</a>')
    st.markdown(f'<div class="vidproc-link-row">{" &nbsp;&nbsp; ".join(links)}</div>', unsafe_allow_html=True)

    if video.transcript_truncated:
        render_notice("This video's transcript was long enough that only part of it was summarized.")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    points = video.points if show_minor else [p for p in video.points if p.importance == "major"]
    if not points:
        render_empty_state("No points to show with the current filter.")
        return
    for point in points:
        _render_point(video, point)
