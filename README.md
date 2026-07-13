# media-flow-youtube

A lightweight backend that fetches YouTube transcripts on demand or on a
schedule, and archives them to a Google Drive folder as clean Markdown
files (with metadata frontmatter) plus a JSON index. Built to run as a
single small service on Railway.

## How it works

- `POST /transcripts` — give it one or more YouTube URLs (or bare video
  IDs), it fetches each caption track via `youtube-transcript-api`, looks
  up the title/channel via YouTube's public oEmbed endpoint, and writes a
  Markdown file per video to your Drive folder.
- `POST /batch/run` — either processes an explicit list of URLs, or (if
  none given) reads `queue.json` from the same Drive folder, processes
  everything in it, and rewrites the queue with only the entries that
  failed transiently. Add new videos to the queue any time by editing
  `queue.json` directly in Drive.
- Every fetch (success or failure) is recorded in `_index.json` in the
  folder, keyed by video ID, so you always know what's been tried and
  where to find it.
- Missing/disabled captions, unavailable videos, and IP blocks are all
  caught and reported as a `status` field — the service never crashes on
  a single bad video.

Because everything lives in one Drive folder as plain Markdown + JSON,
Claude (or you) can browse, search, and read the transcript archive
directly from Drive at any time — no database, no extra API to query.

## Transcript file format

Each video becomes `{title} [{video_id}].md`:

```markdown
---
video_id: dQw4w9WgXcQ
title: "Never Gonna Give You Up"
url: https://www.youtube.com/watch?v=dQw4w9WgXcQ
channel: Rick Astley
fetched_at: 2026-07-13T10:41:55+00:00
language: English (en)
auto_generated: false
---

[00:00] We're no strangers to love
[00:05] You know the rules and so do I
...
```

## Setup

### 1. Create a Google Cloud service account

1. In the [Google Cloud Console](https://console.cloud.google.com/), create
   (or reuse) a project and enable the **Google Drive API**.
2. Create a service account, then create and download a JSON key for it.
3. In Google Drive, create the folder you want transcripts archived to,
   and **share it with the service account's `client_email`** (Editor
   access). The service account has no access of its own — this share is
   what grants it.
4. Copy the folder's ID from its URL (`https://drive.google.com/drive/folders/<THIS_PART>`).

### 2. Configure environment variables

See [`.env.example`](.env.example) for the full list. At minimum:

| Variable | Description |
|---|---|
| `DRIVE_FOLDER_ID` | The Drive folder ID from step 1. |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | The service account key JSON — raw or base64-encoded (`base64 -w0 key.json`). Base64 is recommended since some hosting UIs mangle multi-line env vars. |
| `API_KEY` | Shared secret required in the `X-API-Key` header. The app refuses to start without it unless `DRY_RUN=true`, since the deployed URL is otherwise public and unauthenticated. |

### 3. Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in values, or export them directly
export $(grep -v '^#' .env | xargs)   # or use a tool like direnv/foreman
uvicorn app.main:app --reload
```

Set `DRY_RUN=true` to test the API and transcript pipeline without a real
Drive folder or credentials — Drive writes are logged instead of sent.

### 4. Deploy to Railway

1. Push this repo to GitHub and create a new Railway project from it (or
   `railway up` from the CLi).
2. Set the environment variables from step 2 in the Railway service
   settings.
3. Railway will detect `railway.toml` / `Procfile` and run
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT` automatically.
4. Once deployed, Railway gives you a public URL — that's your endpoint,
   e.g. `https://your-app.up.railway.app/transcripts`.

**Note on IP blocks:** `youtube-transcript-api` fetches captions directly
from YouTube, which rate-limits/blocks known cloud provider IP ranges
(including Railway's) more aggressively than residential IPs. If you see
`"status": "blocked"` responses, see the
[project's guidance on residential proxies](https://github.com/jdepoix/youtube-transcript-api?tab=readme-ov-file#working-around-ip-bans-requestblocked-or-ipblocked-exception).

## Usage

```bash
curl -X POST https://your-app.up.railway.app/transcripts \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]}'
```

```bash
curl -X POST https://your-app.up.railway.app/batch/run \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Response shape (per video):

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "status": "ok",
  "title": "Never Gonna Give You Up",
  "filename": "Never Gonna Give You Up [dQw4w9WgXcQ].md",
  "drive_file_id": "1AbC...",
  "message": null
}
```

`status` is one of: `ok`, `no_captions`, `unavailable`, `blocked`,
`invalid_url`, `error`.

## Scheduled batch processing (optional)

Two ways to run `POST /batch/run` on a schedule — pick one:

**A. In-process scheduler (simplest, one Railway service)**

Set `ENABLE_SCHEDULER=true` and `SCHEDULE_CRON` (a standard 5-field
crontab expression, UTC) on the web service. It runs the queue batch job
on that schedule inside the same process, no extra deployment needed.

**B. Separate Railway Cron Job service**

Deploy `batch_runner.py` as its own Railway service using the
[Cron Job](https://docs.railway.com/guides/cron-jobs) deployment type,
with start command `python batch_runner.py`, pointed at the same
environment variables. Railway runs it on your schedule and shuts it down
between runs — no idle web process needed.

## Project layout

```
app/
  main.py          FastAPI app: /healthz, /transcripts, /batch/run
  pipeline.py       process_video(): the shared fetch-and-archive path
  batch.py          run_batch(): queue-driven or explicit-list batch runs
  youtube.py        URL parsing, transcript fetch, oEmbed title lookup, Markdown rendering
  drive.py          Drive upload/read + _index.json maintenance
  queue_store.py    queue.json read/write helpers
  scheduler.py       optional in-process APScheduler wiring
  config.py         environment variable loading/validation
  models.py         request/response schemas
batch_runner.py     standalone entrypoint for a separate Railway Cron Job service
```
