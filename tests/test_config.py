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


def test_batch_cooldown_seconds_defaults_to_zero(monkeypatch):
    """Regression test: an earlier default of 300s assumed a cooldown was
    needed to let a degraded proxy pool recover. Controlled testing
    disproved that (see README's egress proxy section), so chunking's only
    remaining job is checkpointing - which doesn't require sleeping."""
    assert _settings_with(monkeypatch).batch_cooldown_seconds == 0


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


def test_transcript_fetch_max_attempts_defaults_to_three(monkeypatch):
    assert _settings_with(monkeypatch).transcript_fetch_max_attempts == 3


@pytest.mark.parametrize("value", ["0", "-1"])
def test_transcript_fetch_max_attempts_rejects_non_positive_values(monkeypatch, value):
    monkeypatch.setenv("TRANSCRIPT_FETCH_MAX_ATTEMPTS", value)
    with pytest.raises(ConfigError, match="TRANSCRIPT_FETCH_MAX_ATTEMPTS"):
        _settings_with(monkeypatch)


def test_summary_settings_defaults(monkeypatch):
    settings = _settings_with(monkeypatch)
    assert settings.summary_model == "claude-haiku-4-5"
    assert settings.summary_max_output_tokens == 4096
    assert settings.summary_max_transcript_chars == 400000
    assert settings.summary_max_total_tokens_per_run == 500000
    assert settings.summary_max_cost_usd_per_run == 2.0


@pytest.mark.parametrize(
    "env_var",
    ["SUMMARY_MAX_OUTPUT_TOKENS", "SUMMARY_MAX_TRANSCRIPT_CHARS", "SUMMARY_MAX_TOTAL_TOKENS_PER_RUN"],
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_summary_positive_int_settings_reject_non_positive_values(monkeypatch, env_var, value):
    monkeypatch.setenv(env_var, value)
    with pytest.raises(ConfigError, match=env_var):
        _settings_with(monkeypatch)


@pytest.mark.parametrize("value", ["-1", "-0.5", "nan", "inf"])
def test_summary_max_cost_usd_per_run_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("SUMMARY_MAX_COST_USD_PER_RUN", value)
    with pytest.raises(ConfigError, match="SUMMARY_MAX_COST_USD_PER_RUN"):
        _settings_with(monkeypatch)


def test_vidproc_admin_token_defaults_to_none(monkeypatch):
    monkeypatch.delenv("VIDPROC_ADMIN_TOKEN", raising=False)
    assert _settings_with(monkeypatch).vidproc_admin_token is None


def test_vidproc_admin_token_reads_from_env(monkeypatch):
    assert _settings_with(monkeypatch, VIDPROC_ADMIN_TOKEN="some-secret").vidproc_admin_token == "some-secret"


def test_summary_bulk_read_max_workers_defaults_to_eight(monkeypatch):
    assert _settings_with(monkeypatch).summary_bulk_read_max_workers == 8


@pytest.mark.parametrize("value", ["0", "-1", "33", "100"])
def test_summary_bulk_read_max_workers_rejects_out_of_range_values(monkeypatch, value):
    """Regression test for the review finding: 0 or negative survives
    import but makes ThreadPoolExecutor raise when the dashboard loads a
    snapshot; an unreasonably large value would open that many
    simultaneous Drive connections/OAuth-refreshed clients at once."""
    monkeypatch.setenv("SUMMARY_BULK_READ_MAX_WORKERS", value)
    with pytest.raises(ConfigError, match="SUMMARY_BULK_READ_MAX_WORKERS"):
        _settings_with(monkeypatch)


def test_summary_bulk_read_max_workers_rejects_malformed_value(monkeypatch):
    """Same behavior as every other int-parsed setting in this module
    (e.g. SUMMARY_MAX_OUTPUT_TOKENS) - a non-integer fails at import via
    int()'s own ValueError, before the range check below ever runs."""
    monkeypatch.setenv("SUMMARY_BULK_READ_MAX_WORKERS", "not-a-number")
    with pytest.raises(ValueError):
        _settings_with(monkeypatch)


def test_summary_model_with_unknown_pricing_raises(monkeypatch):
    """Regression test for the review finding: estimate_cost_usd() silently
    returns None for an unrecognized model, so SUMMARY_MAX_COST_USD_PER_RUN
    would stop enforcing any real limit without anyone noticing."""
    monkeypatch.setenv("SUMMARY_MODEL", "some-future-model-not-in-pricing-table")
    with pytest.raises(ConfigError, match="SUMMARY_MODEL"):
        _settings_with(monkeypatch)


def test_summary_max_attempts_per_video_defaults_to_three(monkeypatch):
    assert _settings_with(monkeypatch).summary_max_attempts_per_video == 3


@pytest.mark.parametrize("value", ["0", "-1"])
def test_summary_max_attempts_per_video_rejects_non_positive_values(monkeypatch, value):
    monkeypatch.setenv("SUMMARY_MAX_ATTEMPTS_PER_VIDEO", value)
    with pytest.raises(ConfigError, match="SUMMARY_MAX_ATTEMPTS_PER_VIDEO"):
        _settings_with(monkeypatch)


def test_summary_retry_backoff_seconds_defaults_to_900(monkeypatch):
    assert _settings_with(monkeypatch).summary_retry_backoff_seconds == 900


@pytest.mark.parametrize("value", ["-1", "-0.5", "nan", "inf"])
def test_summary_retry_backoff_seconds_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("SUMMARY_RETRY_BACKOFF_SECONDS", value)
    with pytest.raises(ConfigError, match="SUMMARY_RETRY_BACKOFF_SECONDS"):
        _settings_with(monkeypatch)
