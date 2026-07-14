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


class _FakeExecutable:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeFilesResource:
    def __init__(self, list_result, create_result):
        self._list_result = list_result
        self._create_result = create_result
        self.create_calls = []

    def list(self, **kwargs):
        return _FakeExecutable(self._list_result)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _FakeExecutable(self._create_result)


class _FakeService:
    def __init__(self, list_result, create_result=None):
        self._files = _FakeFilesResource(list_result, create_result)

    def files(self):
        return self._files


def test_get_or_create_folder_returns_existing_folder_id(monkeypatch):
    fake_service = _FakeService({"files": [{"id": "existing-folder-id", "name": "summaries"}]})
    monkeypatch.setattr(drive, "get_drive_service", lambda: fake_service)

    folder_id = drive.get_or_create_folder("parent-id", "summaries")

    assert folder_id == "existing-folder-id"
    assert fake_service.files().create_calls == []


def test_get_or_create_folder_creates_when_missing(monkeypatch):
    fake_service = _FakeService({"files": []}, create_result={"id": "new-folder-id"})
    monkeypatch.setattr(drive, "get_drive_service", lambda: fake_service)

    folder_id = drive.get_or_create_folder("parent-id", "summaries")

    assert folder_id == "new-folder-id"
    assert len(fake_service.files().create_calls) == 1
    body = fake_service.files().create_calls[0]["body"]
    assert body["name"] == "summaries"
    assert body["parents"] == ["parent-id"]
    assert body["mimeType"] == "application/vnd.google-apps.folder"


class _FakePaginatedFilesResource:
    """Serves list() across two pages, so list_files() is exercised for
    pagination rather than just the single-page case."""

    def __init__(self, pages):
        self._pages = pages
        self.list_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        page_token = kwargs.get("pageToken")
        index = 0 if page_token is None else self._pages_by_token[page_token]
        return _FakeExecutable(self._pages[index])


class _FakePaginatedService:
    def __init__(self, pages):
        self._files = _FakePaginatedFilesResource(pages)

    def files(self):
        return self._files


def test_list_files_returns_name_to_id_mapping_across_pages(monkeypatch):
    pages = [
        {"nextPageToken": "page-2", "files": [{"id": "id-1", "name": "vid1.json"}]},
        {"files": [{"id": "id-2", "name": "vid2.json"}]},
    ]
    resource = _FakePaginatedFilesResource(pages)
    resource._pages_by_token = {"page-2": 1}
    fake_service = _FakePaginatedService(pages)
    fake_service._files = resource
    monkeypatch.setattr(drive, "get_drive_service", lambda: fake_service)

    result = drive.list_files("folder-id")

    assert result == {"vid1.json": "id-1", "vid2.json": "id-2"}
    assert len(resource.list_calls) == 2
    assert resource.list_calls[0]["pageToken"] is None
    assert resource.list_calls[1]["pageToken"] == "page-2"


def test_list_files_returns_empty_dict_for_empty_folder(monkeypatch):
    fake_service = _FakeService({"files": []})
    monkeypatch.setattr(drive, "get_drive_service", lambda: fake_service)

    assert drive.list_files("folder-id") == {}


def test_download_text_by_id_reads_file_content(monkeypatch):
    class _FakeDownloader:
        def __init__(self, buffer, request):
            self._buffer = buffer
            self._request = request

        def next_chunk(self):
            self._buffer.write(self._request)
            return None, True

    class _FakeFilesResourceForDownload:
        def get_media(self, fileId):
            return f"content-for-{fileId}".encode()

    class _FakeServiceForDownload:
        def files(self):
            return _FakeFilesResourceForDownload()

    monkeypatch.setattr(drive, "get_drive_service", lambda: _FakeServiceForDownload())
    monkeypatch.setattr(drive, "MediaIoBaseDownload", _FakeDownloader)

    text = drive.download_text_by_id("file-id-1")

    assert text == "content-for-file-id-1"
