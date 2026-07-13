import pytest

from app import drive
from app.config import ConfigError


class _FakeCreds:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.refreshed_with = None

    def refresh(self, request):
        self.refreshed_with = request


@pytest.fixture(autouse=True)
def _reset_cached_service(monkeypatch):
    monkeypatch.setattr(drive, "_service", None)
    yield
    monkeypatch.setattr(drive, "_service", None)


def test_get_drive_service_builds_credentials_from_oauth_env(monkeypatch):
    monkeypatch.setattr(drive.settings, "oauth_client_id", "client-id")
    monkeypatch.setattr(drive.settings, "oauth_client_secret", "client-secret")
    monkeypatch.setattr(drive.settings, "oauth_refresh_token", "refresh-token")

    created_creds = {}

    def fake_credentials(**kwargs):
        creds = _FakeCreds(**kwargs)
        created_creds["instance"] = creds
        return creds

    built = {}
    monkeypatch.setattr(drive, "Credentials", fake_credentials)
    monkeypatch.setattr(drive, "build", lambda *a, **k: built.setdefault("service", object()) or built["service"])

    service = drive.get_drive_service()

    creds = created_creds["instance"]
    assert creds.kwargs["refresh_token"] == "refresh-token"
    assert creds.kwargs["client_id"] == "client-id"
    assert creds.kwargs["client_secret"] == "client-secret"
    assert creds.kwargs["token_uri"] == drive.TOKEN_URI
    assert creds.refreshed_with is not None
    assert service is built["service"]


def test_get_drive_service_raises_config_error_when_oauth_env_missing(monkeypatch):
    monkeypatch.setattr(drive.settings, "oauth_client_id", None)
    monkeypatch.setattr(drive.settings, "oauth_client_secret", None)
    monkeypatch.setattr(drive.settings, "oauth_refresh_token", None)

    with pytest.raises(ConfigError):
        drive.get_drive_service()


def test_get_drive_service_caches_the_built_service(monkeypatch):
    monkeypatch.setattr(drive.settings, "oauth_client_id", "client-id")
    monkeypatch.setattr(drive.settings, "oauth_client_secret", "client-secret")
    monkeypatch.setattr(drive.settings, "oauth_refresh_token", "refresh-token")
    monkeypatch.setattr(drive, "Credentials", lambda **kwargs: _FakeCreds(**kwargs))

    build_calls = []
    monkeypatch.setattr(drive, "build", lambda *a, **k: build_calls.append(1) or object())

    drive.get_drive_service()
    drive.get_drive_service()

    assert len(build_calls) == 1
