import base64
import json

import pytest

from app.config import ConfigError, Settings


def _settings_with(monkeypatch, **env):
    for key in ("GOOGLE_SERVICE_ACCOUNT_JSON", "DRIVE_FOLDER_ID", "API_KEY", "TRANSCRIPT_LANGUAGES"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings()


def test_service_account_info_parses_raw_json(monkeypatch):
    payload = {"client_email": "svc@example.com", "type": "service_account"}
    settings = _settings_with(monkeypatch, GOOGLE_SERVICE_ACCOUNT_JSON=json.dumps(payload))
    assert settings.service_account_info == payload


def test_service_account_info_parses_base64_json(monkeypatch):
    payload = {"client_email": "svc@example.com", "type": "service_account"}
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    settings = _settings_with(monkeypatch, GOOGLE_SERVICE_ACCOUNT_JSON=encoded)
    assert settings.service_account_info == payload


def test_service_account_info_missing_raises_config_error(monkeypatch):
    settings = _settings_with(monkeypatch)
    with pytest.raises(ConfigError):
        settings.service_account_info


def test_service_account_info_garbage_raises_config_error(monkeypatch):
    settings = _settings_with(monkeypatch, GOOGLE_SERVICE_ACCOUNT_JSON="not json and not base64 either!!")
    with pytest.raises(ConfigError):
        settings.service_account_info


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
