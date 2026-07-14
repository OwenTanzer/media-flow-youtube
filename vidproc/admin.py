"""Pure logic for the dashboard's password-gated channel-admin panel.
Deliberately has no Streamlit import - vidproc_app.py owns st.session_state
and calls these as plain functions, matching vidproc/state.py's pattern,
so the auth check and add-channel flow stay unit-testable without a
running Streamlit session."""

from __future__ import annotations

import secrets

from app import channel_store, discovery
from app.channel_store import Channel
from app.discovery import DiscoveryReport


class ChannelAlreadyExistsError(ValueError):
    pass


def check_admin_token(entered: str, configured: str | None) -> bool:
    """Constant-time comparison (secrets.compare_digest) to avoid a timing
    side-channel on the token check. Returns False - not an error - when
    no token is configured at all, so the panel simply never unlocks
    rather than crashing when this optional feature isn't set up."""

    if not configured:
        return False
    return secrets.compare_digest(entered, configured)


def add_channel_and_backfill(
    folder_id: str,
    *,
    channel_id: str,
    name: str,
    enabled: bool = True,
    group: str | None = None,
    languages: list[str] | None = None,
) -> DiscoveryReport:
    """Registers a new channel in channels.json, then immediately backfills
    its currently-visible RSS feed (see app/discovery.py's
    backfill_new_channels(), backfill_new_channels.py) so it doesn't have
    to wait for the next discover_and_process.py cron cycle just because
    it was added through this panel instead of a manual Drive edit.

    Raises ValueError for a blank channel_id, or ChannelAlreadyExistsError
    if channel_id is already registered - callers should catch and
    display these, not treat them as an unexpected failure."""

    channel_id = channel_id.strip()
    if not channel_id:
        raise ValueError("Channel ID is required.")
    name = name.strip() or channel_id
    group = group.strip() or None if group else None
    languages = [code.strip() for code in languages if code.strip()] if languages else None

    existing = channel_store.read_channels(folder_id)
    if any(c.channel_id == channel_id for c in existing):
        raise ChannelAlreadyExistsError(f"Channel {channel_id!r} is already registered.")

    new_channel = Channel(channel_id=channel_id, name=name, enabled=enabled, languages=languages, group=group)
    channel_store.write_channels(folder_id, [*existing, new_channel])

    return discovery.backfill_new_channels(folder_id)
