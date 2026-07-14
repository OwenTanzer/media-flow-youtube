from app import pipeline, youtube
from app.models import VideoResult


def _stub_transcript_and_metadata(monkeypatch, *, status="ok"):
    if status == "ok":
        result = youtube.TranscriptResult(
            "abc123XYZde", "ok", language="English", language_code="en", is_generated=False,
            lines=[(0.0, "hi")],
        )
    else:
        result = youtube.TranscriptResult("abc123XYZde", status, message="nope")
    monkeypatch.setattr(pipeline.youtube, "fetch_transcript", lambda video_id, languages: result)
    monkeypatch.setattr(
        pipeline.youtube, "fetch_video_metadata", lambda video_id: youtube.VideoMetadata("Title", "Author")
    )


def test_process_video_invalid_url_short_circuits():
    result = pipeline.process_video("https://example.com/nope")
    assert result.status == "invalid_url"
    assert result.video_id == ""


def test_process_video_ok_uploads_and_indexes(monkeypatch):
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    uploaded = {}
    monkeypatch.setattr(
        pipeline.drive,
        "upload_text_file",
        lambda folder_id, filename, content, **k: uploaded.setdefault("file_id", "drive-id-123") or "drive-id-123",
    )
    indexed = {}
    monkeypatch.setattr(
        pipeline.drive,
        "update_index_entry",
        lambda folder_id, video_id, entry: indexed.update(entry),
    )

    result = pipeline.process_video("https://www.youtube.com/watch?v=abc123XYZde")

    assert result.status == "ok"
    assert result.drive_file_id == "drive-id-123"
    assert result.message is None
    assert indexed["status"] == "ok"
    assert indexed["video_id"] == "abc123XYZde"


def test_process_video_no_captions_skips_upload(monkeypatch):
    _stub_transcript_and_metadata(monkeypatch, status="no_captions")
    upload_called = []
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: upload_called.append(1))
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda *a, **k: None)

    result = pipeline.process_video("https://www.youtube.com/watch?v=abc123XYZde")

    assert result.status == "no_captions"
    assert result.filename is None
    assert not upload_called


def test_process_video_passes_published_at_to_markdown_and_index(monkeypatch):
    """Regression test: a video's real publish date (only known for
    RSS-discovered videos - see discovery.py) must reach both the
    transcript frontmatter and the _index.json entry, since a future
    visualizer needs to sort by when a video was actually published."""
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    seen_markdown_kwargs = {}
    original_render = youtube.render_transcript_markdown

    def _spy_render(**kwargs):
        seen_markdown_kwargs.update(kwargs)
        return original_render(**kwargs)

    monkeypatch.setattr(pipeline.youtube, "render_transcript_markdown", _spy_render)
    indexed = {}
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda folder_id, video_id, entry: indexed.update(entry))

    pipeline.process_video(
        "https://www.youtube.com/watch?v=abc123XYZde", published_at="2026-07-01T00:00:00+00:00"
    )

    assert seen_markdown_kwargs["published_at"] == "2026-07-01T00:00:00+00:00"
    assert indexed["published_at"] == "2026-07-01T00:00:00+00:00"


def test_process_video_published_at_defaults_to_none(monkeypatch):
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    indexed = {}
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda folder_id, video_id, entry: indexed.update(entry))

    pipeline.process_video("https://www.youtube.com/watch?v=abc123XYZde")

    assert indexed["published_at"] is None


def test_process_video_preserves_existing_published_at_when_reprocessed_without_one(monkeypatch):
    """Regression test for the review finding: update_index_entry()
    replaces the whole stored entry, not just the fields being set. A
    direct/manual reprocess (e.g. via /transcripts or /batch/run with an
    explicit URL, or a queue entry with no known publish date) passes
    published_at=None, which must not erase a publish date an earlier,
    RSS-discovered run of the same video already recorded."""
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.settings, "dry_run", False)
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    monkeypatch.setattr(
        pipeline.drive,
        "read_index",
        lambda folder_id: {"abc123XYZde": {"published_at": "2026-07-01T00:00:00+00:00"}},
    )
    indexed = {}
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda folder_id, video_id, entry: indexed.update(entry))

    pipeline.process_video("https://www.youtube.com/watch?v=abc123XYZde")

    assert indexed["published_at"] == "2026-07-01T00:00:00+00:00"


def test_process_video_does_not_overwrite_published_at_when_a_fresh_one_is_given(monkeypatch):
    """The preserve-existing fallback must only kick in when this
    invocation itself has no publish date - a fresh, real value (e.g. from
    a later RSS discovery run) should still win. channel_id is also given
    fresh here so the lookup is skipped entirely (see the channel_id
    variant of this test below for the case where only one of the two is
    fresh)."""
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.settings, "dry_run", False)
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    read_index_calls = []
    monkeypatch.setattr(
        pipeline.drive,
        "read_index",
        lambda folder_id: read_index_calls.append(1) or {"abc123XYZde": {"published_at": "2026-06-01T00:00:00+00:00"}},
    )
    indexed = {}
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda folder_id, video_id, entry: indexed.update(entry))

    pipeline.process_video(
        "https://www.youtube.com/watch?v=abc123XYZde",
        published_at="2026-07-01T00:00:00+00:00",
        channel_id="UC_a",
    )

    assert indexed["published_at"] == "2026-07-01T00:00:00+00:00"
    assert read_index_calls == []  # no need to even look it up when we already have both


def test_process_video_passes_channel_id_to_index(monkeypatch):
    """channel_id (only known for RSS-discovered videos - see
    discovery.py) must reach the _index.json entry, the same as
    published_at, so a future consumer can join a video back to its
    channels.json entry reliably."""
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    indexed = {}
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda folder_id, video_id, entry: indexed.update(entry))

    pipeline.process_video("https://www.youtube.com/watch?v=abc123XYZde", channel_id="UC_a")

    assert indexed["channel_id"] == "UC_a"


def test_process_video_channel_id_defaults_to_none(monkeypatch):
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    indexed = {}
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda folder_id, video_id, entry: indexed.update(entry))

    pipeline.process_video("https://www.youtube.com/watch?v=abc123XYZde")

    assert indexed["channel_id"] is None


def test_process_video_preserves_existing_channel_id_when_reprocessed_without_one(monkeypatch):
    """Same preserve-on-reprocess fallback as published_at: a direct/manual
    reprocess with no channel_id source must not erase a channel_id an
    earlier, RSS-discovered run already recorded."""
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.settings, "dry_run", False)
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    monkeypatch.setattr(
        pipeline.drive,
        "read_index",
        lambda folder_id: {"abc123XYZde": {"channel_id": "UC_a"}},
    )
    indexed = {}
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda folder_id, video_id, entry: indexed.update(entry))

    pipeline.process_video("https://www.youtube.com/watch?v=abc123XYZde")

    assert indexed["channel_id"] == "UC_a"


def test_process_video_does_not_overwrite_channel_id_when_a_fresh_one_is_given(monkeypatch):
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.settings, "dry_run", False)
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    read_index_calls = []
    monkeypatch.setattr(
        pipeline.drive,
        "read_index",
        lambda folder_id: read_index_calls.append(1) or {"abc123XYZde": {"channel_id": "UC_old"}},
    )
    indexed = {}
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda folder_id, video_id, entry: indexed.update(entry))

    pipeline.process_video(
        "https://www.youtube.com/watch?v=abc123XYZde",
        published_at="2026-07-01T00:00:00+00:00",
        channel_id="UC_new",
    )

    assert indexed["channel_id"] == "UC_new"
    assert read_index_calls == []  # no need to even look it up when we already have both


def test_process_video_looks_up_index_when_only_one_of_published_at_or_channel_id_is_missing(monkeypatch):
    """A partial reprocess (fresh channel_id but no published_at, or vice
    versa) must still consult the existing index entry - the lookup is
    skipped only when *both* are already known."""
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.settings, "dry_run", False)
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    read_index_calls = []
    monkeypatch.setattr(
        pipeline.drive,
        "read_index",
        lambda folder_id: read_index_calls.append(1)
        or {"abc123XYZde": {"published_at": "2026-06-01T00:00:00+00:00", "channel_id": "UC_existing"}},
    )
    indexed = {}
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda folder_id, video_id, entry: indexed.update(entry))

    pipeline.process_video("https://www.youtube.com/watch?v=abc123XYZde", channel_id="UC_new")

    assert len(read_index_calls) == 1
    assert indexed["channel_id"] == "UC_new"  # the fresh value still wins
    assert indexed["published_at"] == "2026-06-01T00:00:00+00:00"  # the preserved value fills the gap


def test_index_failure_does_not_erase_a_successful_archive(monkeypatch):
    """Regression test for the review finding: a failed _index.json write
    must not turn an already-uploaded transcript into a reported failure."""
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")

    def _boom(*a, **k):
        raise RuntimeError("Drive index write failed")

    monkeypatch.setattr(pipeline.drive, "update_index_entry", _boom)

    result = pipeline.process_video("https://www.youtube.com/watch?v=abc123XYZde")

    assert result.status == "ok"
    assert result.drive_file_id == "drive-id-123"
    assert "index update failed" in result.message


def test_safe_process_video_isolates_unexpected_exceptions(monkeypatch):
    """Regression test for the review finding: an unhandled exception from
    anywhere in the pipeline must become an 'error' result, not propagate."""

    def _boom(url_or_id, languages=None, published_at=None, channel_id=None):
        raise RuntimeError("service account credentials are invalid")

    monkeypatch.setattr(pipeline, "process_video", _boom)

    result = pipeline.safe_process_video("https://www.youtube.com/watch?v=abc123XYZde")

    assert isinstance(result, VideoResult)
    assert result.status == "error"
    assert "credentials are invalid" in result.message


def test_safe_process_video_passes_through_normal_results(monkeypatch):
    _stub_transcript_and_metadata(monkeypatch, status="ok")
    monkeypatch.setattr(pipeline.drive, "upload_text_file", lambda *a, **k: "drive-id-123")
    monkeypatch.setattr(pipeline.drive, "update_index_entry", lambda *a, **k: None)

    result = pipeline.safe_process_video("https://www.youtube.com/watch?v=abc123XYZde")

    assert result.status == "ok"
    assert result.drive_file_id == "drive-id-123"
