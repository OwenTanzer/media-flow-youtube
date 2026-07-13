import pytest

from app.config import ConfigError, OAuthCredentials, Settings

_OAUTH_ENV_KEYS = ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_OAUTH_REFRESH_TOKEN")


def _settings_with(monkeypatch, **env):
    for key in (*_OAUTH_ENV_KEYS, "DRIVE_FOLDER_ID", "API_KEY", "TRANSCRIPT_LANGUAGES"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings()


def test_require_oauth_credentials_present(monkeypatch):
    settings = _settings_with(
        monkeypatch,
        GOOGLE_OAUTH_CLIENT_ID="client-id",
        GOOGLE_OAUTH_CLIENT_SECRET="client-secret",
        GOOGLE_OAUTH_REFRESH_TOKEN="refresh-token",
    )
    creds = settings.require_oauth_credentials()
    assert creds == OAuthCredentials(
        client_id="client-id", client_secret="client-secret", refresh_token="refresh-token"
    )


def test_require_oauth_credentials_missing_raises(monkeypatch):
    settings = _settings_with(monkeypatch)
    with pytest.raises(ConfigError, match="GOOGLE_OAUTH_CLIENT_ID"):
        settings.require_oauth_credentials()


def test_require_oauth_credentials_partial_raises(monkeypatch):
    settings = _settings_with(monkeypatch, GOOGLE_OAUTH_CLIENT_ID="client-id")
    with pytest.raises(ConfigError, match="GOOGLE_OAUTH_CLIENT_SECRET"):
        settings.require_oauth_credentials()


def test_require_drive_folder_id_missing_raises(monkeypatch):
    settings = _settings_with(monkeypatch)
    with pytest.raises(ConfigError):
        settings.require_drive_folder_id()


def test_require_drive_folder_id_present(monkeypatch):
    settings = _settings_with(monkeypatch, DRIVE_FOLDER_ID="folder-123")
    assert settings.require_drive_folder_id() == "folder-123"


def test_languages_default_and_parsing(monkeypatch):
    assert _settings_with(monkeypatch).languages == ["en"]
    assert _settings_with(monkeypatch, TRANSCRIPT_LANGUAGES="de, en ,fr").languages == ["de", "en", "fr"]
