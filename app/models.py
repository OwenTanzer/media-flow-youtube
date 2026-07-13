from pydantic import BaseModel, Field


class TranscriptRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, description="YouTube video URLs or bare video IDs.")
    languages: list[str] | None = Field(
        None, description="Preferred transcript language codes, in priority order. Defaults to server config."
    )


class VideoResult(BaseModel):
    video_id: str
    url: str
    status: str
    title: str | None = None
    filename: str | None = None
    drive_file_id: str | None = None
    message: str | None = None


class TranscriptResponse(BaseModel):
    results: list[VideoResult]


class BatchRequest(BaseModel):
    urls: list[str] | None = Field(
        None, description="Explicit URLs to process. If omitted, the Drive-hosted queue file is used instead."
    )
    languages: list[str] | None = None


class BatchResponse(BaseModel):
    processed: int
    results: list[VideoResult]
