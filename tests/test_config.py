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


def test_batch_size_threshold_defaults_to_ten(monkeypatch):
    assert _settings_with(monkeypatch).batch_size_threshold == 10


@pytest.mark.parametrize("value", ["0", "-1", "-100"])
def test_batch_size_threshold_rejects_non_positive_values(monkeypatch, value):
    """Regression test for the review finding: a threshold <= 0 makes
    run_batch() chunk the queue into zero batches, processing nothing and
    silently overwriting queue.json with an empty list."""
    monkeypatch.setenv("BATCH_SIZE_THRESHOLD", value)
    with pytest.raises(ConfigError, match="BATCH_SIZE_THRESHOLD"):
        _settings_with(monkeypatch)


def test_batch_cooldown_seconds_defaults_to_300(monkeypatch):
    assert _settings_with(monkeypatch).batch_cooldown_seconds == 300


@pytest.mark.parametrize("value", ["-1", "-0.5", "nan", "inf"])
def test_batch_cooldown_seconds_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("BATCH_COOLDOWN_SECONDS", value)
    with pytest.raises(ConfigError, match="BATCH_COOLDOWN_SECONDS"):
        _settings_with(monkeypatch)


def test_no_captions_grace_hours_defaults_to_24(monkeypatch):
    assert _settings_with(monkeypatch).no_captions_grace_hours == 24


@pytest.mark.parametrize("value", ["-1", "-0.5", "nan", "inf"])
def test_no_captions_grace_hours_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("NO_CAPTIONS_GRACE_HOURS", value)
    with pytest.raises(ConfigError, match="NO_CAPTIONS_GRACE_HOURS"):
        _settings_with(monkeypatch)
