import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status

from . import youtube
from .batch import run_batch
from .config import ConfigError, settings
from .models import BatchRequest, BatchResponse, TranscriptRequest, TranscriptResponse, VideoResult
from .pipeline import safe_process_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("media_flow.main")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not settings.api_key and not settings.dry_run:
        raise RuntimeError(
            "API_KEY is not set. Refusing to start with an unauthenticated public endpoint. "
            "Set API_KEY, or set DRY_RUN=true for local testing without auth."
        )
    try:
        youtube.build_proxy_config()
    except ConfigError as exc:
        raise RuntimeError(f"Invalid YouTube proxy configuration: {exc}") from exc

    if settings.dry_run:
        logger.warning("DRY_RUN is enabled - no files will actually be written to Google Drive.")
    else:
        from . import drive

        settings.require_drive_folder_id()
        try:
            drive.get_drive_service()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to validate Google Drive OAuth credentials at startup: {exc}"
            ) from exc
    if settings.enable_scheduler:
        from .scheduler import start_scheduler

        start_scheduler()
    yield
    if settings.enable_scheduler:
        from .scheduler import stop_scheduler

        stop_scheduler()


app = FastAPI(title="media-flow-youtube", version="1.0.0", lifespan=lifespan)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key header.")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {
        "service": "media-flow-youtube",
        "endpoints": ["/healthz", "/transcripts", "/batch/run"],
    }


@app.post("/transcripts", response_model=TranscriptResponse, dependencies=[Depends(require_api_key)])
def create_transcripts(request: TranscriptRequest) -> TranscriptResponse:
    try:
        settings.require_drive_folder_id()
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results: list[VideoResult] = [safe_process_video(url, request.languages) for url in request.urls]
    return TranscriptResponse(results=results)


@app.post("/batch/run", response_model=BatchResponse, dependencies=[Depends(require_api_key)])
def batch_run(request: BatchRequest) -> BatchResponse:
    try:
        results = run_batch(request.urls, request.languages)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return BatchResponse(processed=len(results), results=results)
