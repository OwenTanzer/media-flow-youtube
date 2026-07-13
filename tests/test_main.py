import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import VideoResult


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcripts_requires_api_key(client):
    response = client.post("/transcripts", json={"urls": ["https://www.youtube.com/watch?v=abc123XYZde"]})
    assert response.status_code == 401


def test_transcripts_returns_per_video_results(client, monkeypatch):
    import app.main as main_module

    def fake_safe_process_video(url, languages=None):
        return VideoResult(video_id="abc123XYZde", url=url, status="ok", title="T", filename="T.md")

    monkeypatch.setattr(main_module, "safe_process_video", fake_safe_process_video)

    response = client.post(
        "/transcripts",
        headers={"X-API-Key": "test-api-key"},
        json={"urls": ["https://www.youtube.com/watch?v=abc123XYZde"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["status"] == "ok"
    assert body["results"][0]["filename"] == "T.md"


def test_batch_run_returns_processed_count(client, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(
        main_module,
        "run_batch",
        lambda urls, languages: [VideoResult(video_id="x", url="https://y", status="ok")],
    )

    response = client.post("/batch/run", headers={"X-API-Key": "test-api-key"}, json={})
    assert response.status_code == 200
    assert response.json()["processed"] == 1


def test_startup_fails_closed_without_api_key_or_dry_run(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "dry_run", False)

    with pytest.raises(RuntimeError, match="API_KEY is not set"):
        with TestClient(app):
            pass


def test_startup_validates_drive_credentials_when_not_dry_run(monkeypatch):
    from app import drive
    from app.config import settings

    monkeypatch.setattr(settings, "dry_run", False)
    monkeypatch.setattr(drive, "get_drive_service", lambda: object())

    with TestClient(app):
        pass  # should not raise


def test_startup_fails_when_drive_credentials_are_invalid(monkeypatch):
    from app import drive
    from app.config import settings

    monkeypatch.setattr(settings, "dry_run", False)

    def _boom():
        raise RuntimeError("invalid_grant: token has been revoked")

    monkeypatch.setattr(drive, "get_drive_service", _boom)

    with pytest.raises(RuntimeError, match="Failed to validate Google Drive OAuth credentials"):
        with TestClient(app):
            pass


def test_startup_fails_when_drive_folder_id_missing_and_not_dry_run(monkeypatch):
    from app.config import ConfigError, settings

    monkeypatch.setattr(settings, "dry_run", False)
    monkeypatch.setattr(settings, "drive_folder_id", None)

    with pytest.raises(ConfigError):
        with TestClient(app):
            pass
