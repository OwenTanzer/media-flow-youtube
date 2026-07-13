import base64
import json
import os


class ConfigError(RuntimeError):
    pass


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    """Loads and validates all runtime configuration from environment variables."""

    def __init__(self) -> None:
        self.drive_folder_id: str | None = _env("DRIVE_FOLDER_ID")
        self.api_key: str | None = _env("API_KEY")
        self.dry_run: bool = _env_bool("DRY_RUN", False)

        self.languages: list[str] = [
            code.strip() for code in _env("TRANSCRIPT_LANGUAGES", "en").split(",") if code.strip()
        ]

        self.enable_scheduler: bool = _env_bool("ENABLE_SCHEDULER", False)
        self.schedule_cron: str | None = _env("SCHEDULE_CRON")

        self._service_account_raw = _env("GOOGLE_SERVICE_ACCOUNT_JSON")

    @property
    def service_account_info(self) -> dict:
        if not self._service_account_raw:
            raise ConfigError(
                "GOOGLE_SERVICE_ACCOUNT_JSON is not set. Provide the service account "
                "credentials JSON (raw or base64-encoded) as an environment variable."
            )
        raw = self._service_account_raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            decoded = base64.b64decode(raw).decode("utf-8")
            return json.loads(decoded)
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(
                "GOOGLE_SERVICE_ACCOUNT_JSON could not be parsed as JSON or "
                "base64-encoded JSON."
            ) from exc

    def require_drive_folder_id(self) -> str:
        if not self.drive_folder_id:
            raise ConfigError("DRIVE_FOLDER_ID is not set.")
        return self.drive_folder_id


settings = Settings()
