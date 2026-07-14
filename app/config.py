import math
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

        self.discovery_lock_ttl_seconds: int = int(_env("DISCOVERY_LOCK_TTL_SECONDS", "1800"))
        self.no_captions_grace_hours: float = float(_env("NO_CAPTIONS_GRACE_HOURS", "24"))
        if self.no_captions_grace_hours < 0 or not math.isfinite(self.no_captions_grace_hours):
            raise ConfigError(f"NO_CAPTIONS_GRACE_HOURS must be a non-negative, finite number of hours, got {self.no_captions_grace_hours}.")

        self.batch_size_threshold: int = int(_env("BATCH_SIZE_THRESHOLD", "10"))
        if self.batch_size_threshold < 1:
            raise ConfigError(
                f"BATCH_SIZE_THRESHOLD must be a positive integer, got {self.batch_size_threshold}. "
                "A value <= 0 makes run_batch() chunk the queue into zero batches, silently "
                "processing nothing and overwriting queue.json with an empty list."
            )

        self.batch_cooldown_seconds: float = float(_env("BATCH_COOLDOWN_SECONDS", "0"))
        if self.batch_cooldown_seconds < 0 or not math.isfinite(self.batch_cooldown_seconds):
            raise ConfigError(f"BATCH_COOLDOWN_SECONDS must be a non-negative, finite number of seconds, got {self.batch_cooldown_seconds}.")

        self.oauth_client_id: str | None = _env("GOOGLE_OAUTH_CLIENT_ID")
        self.oauth_client_secret: str | None = _env("GOOGLE_OAUTH_CLIENT_SECRET")
        self.oauth_refresh_token: str | None = _env("GOOGLE_OAUTH_REFRESH_TOKEN")

        self.transcript_fetch_max_attempts: int = int(_env("TRANSCRIPT_FETCH_MAX_ATTEMPTS", "3"))
        if self.transcript_fetch_max_attempts < 1:
            raise ConfigError(
                f"TRANSCRIPT_FETCH_MAX_ATTEMPTS must be a positive integer, got {self.transcript_fetch_max_attempts}."
            )

        self.youtube_proxy_type: str | None = _env("YOUTUBE_PROXY_TYPE")
        self.webshare_proxy_username: str | None = _env("WEBSHARE_PROXY_USERNAME")
        self.webshare_proxy_password: str | None = _env("WEBSHARE_PROXY_PASSWORD")
        self.webshare_proxy_locations: list[str] = [
            code.strip().upper() for code in _env("WEBSHARE_PROXY_LOCATIONS", "").split(",") if code.strip()
        ]
        self.youtube_proxy_http_url: str | None = _env("YOUTUBE_PROXY_HTTP_URL")
        self.youtube_proxy_https_url: str | None = _env("YOUTUBE_PROXY_HTTPS_URL")

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
