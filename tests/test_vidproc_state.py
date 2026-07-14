from datetime import datetime, timezone

from app.channel_store import Channel
from app.insights_store import VideoInsight
from vidproc import state

FINANCE_A = Channel("UC_a", "Finance A", group="Finance")
FINANCE_B = Channel("UC_b", "Finance B")  # group=None -> Finance fallback
GOOGLE = Channel("UC_g", "Google for Developers", group="Google")


def _video(video_id, group, channel_id=None, channel_name=None, published=None, generated=None) -> VideoInsight:
    return VideoInsight(
        video_id=video_id,
        title=f"Title {video_id}",
        author="Some Author",
        url=f"https://www.youtube.com/watch?v={video_id}",
        channel_id=channel_id,
        channel_name=channel_name,
        group=group,
        video_type="Analytic Overview",
        video_published_at=published,
        generated_at=generated,
        summary="Summary.",
        points=[],
        drive_file_id=None,
        transcript_truncated=False,
    )


def test_groups_for_channels_includes_default_group_even_when_absent():
    assert state.groups_for_channels([GOOGLE]) == ["Finance", "Google"]


def test_groups_for_channels_deduplicates_and_sorts():
    assert state.groups_for_channels([FINANCE_A, FINANCE_B, GOOGLE]) == ["Finance", "Google"]


def test_channels_in_group_uses_resolved_group_not_raw_field():
    result = state.channels_in_group([FINANCE_A, FINANCE_B, GOOGLE], "Finance")
    assert result == [FINANCE_A, FINANCE_B]


def test_validate_channel_selection_drops_stale_ids():
    result = state.validate_channel_selection(["UC_a", "UC_stale"], ["UC_a", "UC_b"])
    assert result == ["UC_a"]


def test_validate_channel_selection_preserves_order():
    result = state.validate_channel_selection(["UC_b", "UC_a"], ["UC_a", "UC_b"])
    assert result == ["UC_b", "UC_a"]


def test_filter_videos_all_channels_when_selection_empty():
    videos = [_video("v1", "Finance", "UC_a"), _video("v2", "Google", "UC_g")]
    result = state.filter_videos(videos, "Finance", None)
    assert [v.video_id for v in result] == ["v1"]


def test_filter_videos_narrows_to_selected_channels():
    videos = [
        _video("v1", "Finance", "UC_a"),
        _video("v2", "Finance", "UC_b"),
    ]
    result = state.filter_videos(videos, "Finance", ["UC_a"])
    assert [v.video_id for v in result] == ["v1"]


def test_filter_videos_selecting_unassigned_pseudo_channel():
    videos = [
        _video("v1", "Finance", channel_id=None),
        _video("v2", "Finance", "UC_a"),
    ]
    result = state.filter_videos(videos, "Finance", [state.UNASSIGNED_CHANNEL_ID])
    assert [v.video_id for v in result] == ["v1"]


def test_filter_videos_a_group_cannot_leak_into_another():
    videos = [_video("v1", "Finance", "UC_a"), _video("v2", "Google", "UC_a")]
    # Selecting "UC_a" while scoped to Google should not surface the Finance video.
    result = state.filter_videos(videos, "Google", ["UC_a"])
    assert [v.video_id for v in result] == ["v2"]


def test_feed_sort_key_dated_videos_before_undated():
    dated = _video("dated", "Finance", published=datetime(2026, 1, 1, tzinfo=timezone.utc))
    undated = _video("undated", "Finance", generated=datetime(2026, 6, 1, tzinfo=timezone.utc))
    result = state.sorted_feed([undated, dated])
    assert [v.video_id for v in result] == ["dated", "undated"]


def test_feed_sort_key_dated_videos_most_recent_first():
    older = _video("older", "Finance", published=datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer = _video("newer", "Finance", published=datetime(2026, 6, 1, tzinfo=timezone.utc))
    result = state.sorted_feed([older, newer])
    assert [v.video_id for v in result] == ["newer", "older"]


def test_feed_sort_key_undated_videos_fall_back_to_generated_at():
    older_gen = _video("older_gen", "Finance", generated=datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer_gen = _video("newer_gen", "Finance", generated=datetime(2026, 6, 1, tzinfo=timezone.utc))
    result = state.sorted_feed([older_gen, newer_gen])
    assert [v.video_id for v in result] == ["newer_gen", "older_gen"]


def test_feed_sort_key_handles_missing_generated_at_too():
    """A video with neither video_published_at nor generated_at (shouldn't
    happen in practice - see insights_store - but must not crash sorting)."""
    no_dates = _video("no_dates", "Finance")
    dated = _video("dated", "Finance", published=datetime(2026, 1, 1, tzinfo=timezone.utc))
    result = state.sorted_feed([no_dates, dated])
    assert [v.video_id for v in result] == ["dated", "no_dates"]


def test_channel_filter_options_lists_channels_in_group_only():
    videos = [_video("v1", "Finance", "UC_a", channel_name="Finance A")]
    options = state.channel_filter_options([FINANCE_A, GOOGLE], "Finance", videos)
    assert options == [("UC_a", "Finance A")]


def test_channel_filter_options_includes_unassigned_only_when_present():
    videos_without = [_video("v1", "Finance", "UC_a", channel_name="Finance A")]
    videos_with = [_video("v2", "Finance", channel_id=None)]

    assert state.channel_filter_options([FINANCE_A], "Finance", videos_without) == [("UC_a", "Finance A")]
    result_with = state.channel_filter_options([FINANCE_A], "Finance", videos_with)
    assert (state.UNASSIGNED_CHANNEL_ID, "Unassigned / Other") in result_with
