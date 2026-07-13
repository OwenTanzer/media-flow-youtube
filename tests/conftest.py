import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _base_settings(monkeypatch):
    """Every test gets a known-good baseline config; individual tests
    override specific attributes with monkeypatch so it's reverted automatically."""
    monkeypatch.setattr(settings, "drive_folder_id", "test-folder-id")
    monkeypatch.setattr(settings, "api_key", "test-api-key")
    monkeypatch.setattr(settings, "dry_run", True)
    monkeypatch.setattr(settings, "languages", ["en"])
    monkeypatch.setattr(settings, "enable_scheduler", False)
    monkeypatch.setattr(settings, "schedule_cron", None)
    yield settings
