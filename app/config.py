import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass
class OAuthCredentials:
    client_id: str
    client_secret: str
    refresh_token: str


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

        self.oauth_client_id: str | None = _env("GOOGLE_OAUTH_CLIENT_ID")
        self.oauth_client_secret: str | None = _env("GOOGLE_OAUTH_CLIENT_SECRET")
        self.oauth_refresh_token: str | None = _env("GOOGLE_OAUTH_REFRESH_TOKEN")

    def require_oauth_credentials(self) -> OAuthCredentials:
        missing = [
            name
            for name, value in (
                ("GOOGLE_OAUTH_CLIENT_ID", self.oauth_client_id),
                ("GOOGLE_OAUTH_CLIENT_SECRET", self.oauth_client_secret),
                ("GOOGLE_OAUTH_REFRESH_TOKEN", self.oauth_refresh_token),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                f"Missing required OAuth environment variable(s): {', '.join(missing)}. "
                "Run get_refresh_token.py once locally to obtain a refresh token, "
                "then set all three."
            )
        return OAuthCredentials(
            client_id=self.oauth_client_id,
            client_secret=self.oauth_client_secret,
            refresh_token=self.oauth_refresh_token,
        )

    def require_drive_folder_id(self) -> str:
        if not self.drive_folder_id:
            raise ConfigError("DRIVE_FOLDER_ID is not set.")
        return self.drive_folder_id


settings = Settings()
