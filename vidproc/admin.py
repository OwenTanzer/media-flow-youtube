"""Pure logic for the dashboard's password-gated channel-admin panel.
Deliberately has no Streamlit import - vidproc_app.py owns st.session_state
and calls these as plain functions, matching vidproc/state.py's pattern,
so the auth check and add-channel flow stay unit-testable without a
running Streamlit session."""

from __future__ import annotations

import logging
import re
import secrets
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

from app import channel_store, discovery, job_lock
from app.channel_store import Channel
from app.config import settings
from app.discovery import DiscoveryReport

logger = logging.getLogger("media_flow.vidproc_admin")

# YouTube's stable "UC..." channel ID: "UC" + 22 URL-safe base64-ish
# characters, 24 total. Not a airtight guarantee the channel exists (the
# RSS preflight fetch below is what actually confirms that), but catches
# an obvious typo/wrong-format paste before it's ever written to
# channels.json.
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")


class ChannelAlreadyExistsError(ValueError):
    pass


@dataclass
class AddChannelResult:
    channel: Channel
    # Exactly one of these is meaningful at a time: a successful backfill
    # sets backfill_report; a failed one sets backfill_error; the main
    # discovery lock being held (discover_and_process.py mid-run) sets
    # backfill_deferred, since this call deliberately never waits for
    # that lock (see add_channel_and_backfill()'s docstring) - the
    # channel is still added immediately either way, just its backfill
    # will happen on the next discover_and_process.py run instead.
    backfill_report: DiscoveryReport | None = None
    backfill_deferred: bool = False
    backfill_error: str | None = None


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
) -> AddChannelResult:
    """Registers a new channel in channels.json, then immediately backfills
    its currently-visible RSS feed (see app/discovery.py's
    backfill_new_channels()) so it doesn't have to wait for the next
    discover_and_process.py cron cycle just because it was added through
    this panel instead of a manual Drive edit.

    Raises ValueError for a blank/malformed channel_id or one whose feed
    can't be fetched (checked *before* writing anything, so a typo is
    never permanently written - see the review finding this addresses),
    or ChannelAlreadyExistsError if channel_id is already registered.
    Callers should catch and display these, not treat them as an
    unexpected failure.

    Backfilling shares discover_and_process.py's own advisory lock
    (app/job_lock.py) rather than a lock of its own, since both write the
    same queue.json - a distinct lock would only serialize this call
    against itself while doing nothing to stop it from interleaving with
    a concurrently-running discover_and_process.py (see job_lock.py's
    module docstring). This call never waits for that lock, though: if
    discover_and_process.py is running right now, the channel is still
    added immediately and AddChannelResult.backfill_deferred is set,
    rather than blocking the admin panel until the lock frees up - that
    run (or the next one) will pick up the new channel on its own."""

    channel_id = channel_id.strip()
    if not channel_id:
        raise ValueError("Channel ID is required.")
    if not _CHANNEL_ID_RE.match(channel_id):
        raise ValueError(
            f"{channel_id!r} doesn't look like a YouTube channel ID - expected \"UC\" followed by "
            "22 characters (find it in the channel page's source, or via a channel-ID lookup tool)."
        )
    name = name.strip() or channel_id
    group = group.strip() or None if group else None
    languages = [code.strip() for code in languages if code.strip()] if languages else None

    existing = channel_store.read_channels(folder_id)
    if any(c.channel_id == channel_id for c in existing):
        raise ChannelAlreadyExistsError(f"Channel {channel_id!r} is already registered.")

    # Preflight: confirm the feed is actually fetchable *before* writing
    # anything, so a bad channel_id is never permanently persisted only
    # to be discovered wrong on the next backfill/cron run.
    try:
        discovery.fetch_channel_feed(channel_id)
    except (requests.RequestException, ET.ParseError) as exc:
        raise ValueError(f"Could not fetch this channel's RSS feed - double-check the channel ID. ({exc})") from exc

    new_channel = Channel(channel_id=channel_id, name=name, enabled=enabled, languages=languages, group=group)
    channel_store.write_channels(folder_id, [*existing, new_channel])

    lock_token = job_lock.acquire_lock(folder_id, settings.discovery_lock_ttl_seconds)
    if lock_token is None:
        return AddChannelResult(channel=new_channel, backfill_deferred=True)

    try:
        try:
            report = discovery.backfill_new_channels(folder_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Immediate backfill failed for newly-added channel %s", channel_id)
            return AddChannelResult(channel=new_channel, backfill_error=str(exc))
    finally:
        job_lock.release_lock(folder_id, lock_token)

    return AddChannelResult(channel=new_channel, backfill_report=report)


def admin_flash_for(result: AddChannelResult) -> tuple[str, str]:
    """Maps an AddChannelResult to an (st.<level> method name, message)
    pair for vidproc_app.py to display. channels.json is always written
    by the time this is called (see add_channel_and_backfill()'s
    docstring), so every branch here is reporting on the immediate-
    backfill step specifically, not whether the channel itself was added."""

    channel_id = result.channel.channel_id
    if result.backfill_deferred:
        return "info", (
            f"Channel {channel_id} added. Immediate backfill was skipped because "
            "discover_and_process.py is currently running - that run (or the next one) will "
            "pick up the new channel automatically."
        )
    if result.backfill_error:
        return "warning", f"Channel {channel_id} added, but its immediate backfill failed: {result.backfill_error}"

    report = result.backfill_report
    own_failure = next((message for cid, message in report.feed_failures if cid == channel_id), None)
    if own_failure:
        return "warning", (
            f"Channel {channel_id} added, but its feed couldn't be fetched just now: {own_failure}. "
            "It'll be retried on the next discover_and_process.py run."
        )
    return "success", f"Channel {channel_id} added - {report.newly_queued} video(s) queued from its current feed."
