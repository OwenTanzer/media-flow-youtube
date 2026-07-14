"""CSS/visual constants for the dashboard, reusing the oil-crisis-dashboard
MediaFlow app's visual framework (issue #8): a sidebar-free, light,
editorial layout with Crimson Text for reading content and Oxanium for
compact metadata/controls. Deliberately not reusing that reference's
background-collector thread, iframe-based hotkey/timezone JS, or
terminal/chat sub-modes - none of those apply to this read-only,
cron-fed dashboard."""

# video_type is the natural per-video categorical field for the feed
# card's left accent - importance (major/minor) is a per-point property,
# used inside the detail view instead, not at the card level.
VIDEO_TYPE_COLORS = {
    "Post-Market Update": "#2980b9",
    "Pre-Market Brief": "#27ae60",
    "Thesis Piece": "#8e44ad",
    "Analytic Overview": "#d35400",
}
DEFAULT_ACCENT_COLOR = "#999999"  # unrecognized/missing video_type - schema-drift safety


def accent_color_for(video_type: str | None) -> str:
    return VIDEO_TYPE_COLORS.get(video_type or "", DEFAULT_ACCENT_COLOR)


CHROME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Oxanium:wght@700&display=swap');
:root { color-scheme: light; }
[data-testid="stAppViewContainer"] { background: #fff; }
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; margin-top: 0 !important; }
.stTabs [data-baseweb="tab"] { padding: 6px 14px; }
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] { font-family: 'Crimson Text', Georgia, serif !important; font-size: 0.98em !important; }
hr { margin: 0.4rem 0 !important; }
h1, h2, h3 { margin-top: 0 !important; margin-bottom: 0.2rem !important; }
body, .stMarkdown, .stCaption, button { font-family: 'Crimson Text', Georgia, serif !important; }
.vidproc-main-text { font-family: 'Crimson Text', Georgia, serif; font-size: 1.05em; line-height: 1.5; }
.vidproc-meta-text { font-family: 'Oxanium', monospace; font-weight: 700; color: #999; font-size: 0.78em; }
.vidproc-card { border-left: 3px solid #999; padding: 8px 14px; margin-bottom: 12px; }
.vidproc-title { font-family: 'Crimson Text', Georgia, serif; font-weight: 600; font-size: 1.2em; }
.vidproc-summary { font-family: 'Crimson Text', Georgia, serif; font-size: 1.0em; margin-top: 4px; }
.vidproc-main-point { font-family: 'Crimson Text', Georgia, serif; font-weight: 700; font-size: 1.08em; }
.vidproc-main-point.minor { font-weight: 400; font-size: 0.95em; color: #555; }
.vidproc-explanation { font-family: 'Crimson Text', Georgia, serif; font-size: 0.95em; color: #333; margin: 2px 0 6px 0; }
.vidproc-timestamp-link { font-family: 'Oxanium', monospace; font-size: 0.8em; text-decoration: none; color: #2980b9; }
.vidproc-badge { font-family: 'Oxanium', monospace; font-size: 0.7em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; padding: 1px 6px; border-radius: 3px; }
.vidproc-badge.major { background: #eef1f5; color: #2c3e50; }
.vidproc-badge.minor { background: #f7f7f7; color: #999; }
.vidproc-link-row { font-family: 'Oxanium', monospace; font-size: 0.78em; color: #999; }
.vidproc-link-row a { color: #999; text-decoration: none; }
.vidproc-notice { font-family: 'Oxanium', monospace; font-size: 0.85em; color: #b9770e; background: #fdf6e8; border-left: 3px solid #b9770e; padding: 6px 10px; margin-bottom: 10px; }
div[data-testid="stButton"] > button {
    font-family: 'Oxanium', monospace !important;
}
</style>
"""
