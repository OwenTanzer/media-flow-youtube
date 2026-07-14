# media-flow-youtube

![CI](https://github.com/OwenTanzer/media-flow-youtube/actions/workflows/ci.yml/badge.svg)

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
- `discover_and_process.py` — optionally, poll a set of YouTube channels
  (configured in `channels.json`, also in Drive) for new uploads via
  their public RSS feeds, queue the ones not seen before, and process the
  queue, all in one scheduled run. See "Scheduled channel discovery" below.
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

### 1. Get OAuth credentials for your Google account

The app authenticates to Drive as **your own Google account**, not a
service account — service accounts have zero Drive storage quota and
can't create new files in a personal ("My Drive") folder, only edit
files someone else already owns. Since a personal Gmail account can't
use Shared Drives or domain-delegated service accounts (those require
Google Workspace), OAuth user credentials are the fix.

1. In the [Google Cloud Console](https://console.cloud.google.com/), create
   (or reuse) a project and enable the **Google Drive API**.
2. Under **APIs & Services → Credentials**, create an **OAuth client ID**
   of type **Desktop app**. Note the client ID and client secret.
3. Locally, run the one-time authorization script to mint a refresh token:
   ```bash
   pip install google-auth-oauthlib
   python get_refresh_token.py
   ```
   This opens a browser for you to sign in and grant Drive access, then
   prints `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, and
   `GOOGLE_OAUTH_REFRESH_TOKEN` — set all three wherever the app runs.
4. In Google Drive, create the folder you want transcripts archived to.
   No sharing step is needed — you own it, since the app now acts as you.
5. Copy the folder's ID from its URL (`https://drive.google.com/drive/folders/<THIS_PART>`).

Treat the client secret and refresh token like passwords: set them only
in Railway's environment variables (or a local, gitignored `.env`) —
never commit them, and never paste them into a public place.

**Before deploying long-term**, check your OAuth consent screen's
publishing status under **APIs & Services → OAuth consent screen**. Left
in **Testing**, Google expires refresh tokens after 7 days — fine for a
local smoke test, but it'll silently break the deployed service a week
in. Move it to **Production** (no Google verification review is required
for personal, single-user use like this) so the refresh token doesn't
expire. If a token does expire, just rerun `get_refresh_token.py` and
update the env var. See
[Google's docs on refresh token expiration](https://developers.google.com/identity/protocols/oauth2#expiration).

### 2. Configure environment variables

See [`.env.example`](.env.example) for the full list. At minimum:

| Variable | Description |
|---|---|
| `DRIVE_FOLDER_ID` | The Drive folder ID from step 1. |
| `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET` / `GOOGLE_OAUTH_REFRESH_TOKEN` | From `get_refresh_token.py` above. |
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

### Testing

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

The test suite mocks YouTube/Drive/oEmbed calls throughout, so it runs
offline with no real credentials. It also covers the failure-isolation
behavior called out below (a bad video can't 500 a whole request, and a
failed index update can't erase a successful archive). GitHub Actions
(`.github/workflows/ci.yml`) runs both on every push and pull request.

### 4. Deploy to Railway

1. Push this repo to GitHub and create a new Railway project from it (or
   `railway up` from the CLi).
2. Set the environment variables from step 2 in the Railway service
   settings.
3. Railway will detect `railway.toml` / `Procfile` and run
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT` automatically.
4. Once deployed, Railway gives you a public URL — that's your endpoint,
   e.g. `https://your-app.up.railway.app/transcripts`.

### 5. Working around IP blocks (optional, but likely needed)

`youtube-transcript-api` fetches captions directly from YouTube, which
blocks known cloud-provider IP ranges (Railway's included) far more
aggressively than residential IPs. If you see `"status": "blocked"` in
responses, route requests through a proxy via `YOUTUBE_PROXY_TYPE` (see
[`.env.example`](.env.example) for every variable) - no code or API
changes needed, just configuration:

| `YOUTUBE_PROXY_TYPE` | Use when | Required variables |
|---|---|---|
| unset (default) | Requests direct, no proxy | — |
| `webshare` | Using [Webshare](https://www.webshare.io/), which the library has built-in retry/rotation support for. It has a permanent free tier (10 datacenter IPs, 1GB/month) worth trying first - though datacenter IPs may still get blocked the same way Railway's do. Their paid "Residential" tier is the reliable fix. | `WEBSHARE_PROXY_USERNAME`, `WEBSHARE_PROXY_PASSWORD` (from the [Webshare dashboard](https://dashboard.webshare.io/proxy/settings)) |
| `generic` | Any other HTTP/HTTPS proxy, including a self-hosted tunnel back to your own residential IP (e.g. via Cloudflare Tunnel or Tailscale) | `YOUTUBE_PROXY_HTTP_URL` and/or `YOUTUBE_PROXY_HTTPS_URL`, e.g. `http://user:pass@host:port` |

The app validates this configuration at startup — an invalid or
incomplete proxy setup fails the deployment immediately rather than
surfacing later as per-video `blocked`/`error` results. A working but
rejected proxy still produces the existing safe behavior: affected
videos come back as `blocked` and stay queued for retry, exactly as
without a proxy.

**Rotating/updating credentials:** proxy credentials are read fresh from
environment variables on every YouTube request rather than cached, so
rotating them is just updating the Railway variables and letting the
service restart (Railway does this automatically on a variable change).
No redeploy of code is required. As with all secrets in this project,
set proxy credentials only in Railway's environment variables (or a
local, gitignored `.env`) - never commit them.

**Note on `WEBSHARE_PROXY_USERNAME`:** use the *bare* username from your
Webshare dashboard, not a sticky-session variant (e.g. one with a
`-<country>-<n>` suffix already appended). `youtube-transcript-api`
appends its own `-rotate` suffix to whatever username you provide, so a
username that already has a session suffix baked in produces an invalid
combined string and proxy auth fails. A bare username lets it correctly
draw a fresh IP from the rotating pool on every request - verified by
hitting an IP-echo endpoint through the proxy repeatedly and confirming
the IP actually changes each time, rather than staying fixed on one
sticky IP.

**A single rotating-proxy draw has a fixed chance of landing on an
already-flagged exit IP - and this doesn't improve by waiting.** An
earlier investigation into a large live run's rising failure rate
initially assumed the pool was being "worn out" by sustained request
volume and needed a cooldown to recover. Controlled follow-up testing
disproved that: a fresh, completely unpaced single request against a
video that had just failed in production succeeded at roughly the same
~80-85% rate whether tested immediately, mid-run, or after a 5-minute
gap - and independent draws against Webshare's own IP-echo endpoint
showed the rotating pool handing out distinct, healthy IPs throughout,
even while the "degraded" run was still failing. So the actual
per-request failure rate looks like a fairly constant property of the
pool at any given moment (some fraction of exit IPs are already flagged
by YouTube), not something that accumulates with our own volume or is
fixed by resting.

What actually fixes it: `fetch_transcript()` (see `app/youtube.py`)
retries a blocked or network-flaky attempt up to
`TRANSCRIPT_FETCH_MAX_ATTEMPTS` times (default `3`), each attempt using a
brand-new `YouTubeTranscriptApi` instance - a fresh connection, and thus
another independent draw from the rotating pool - rather than relying on
the library's own internal same-session retry (which testing showed
doesn't reliably escape a blocked IP) or giving up after a single bad
draw. At a ~80% single-shot success rate, 2-3 independent draws pushes
effective success into the high 90s%; this was confirmed directly by
retrying videos that had actually failed in a live production run, most
of which succeeded on the very next fresh attempt.

`run_batch()` (see `app/batch.py`) still processes queued entries in
chunks of `BATCH_SIZE_THRESHOLD` (default `10`) with a
`BATCH_COOLDOWN_SECONDS` (default `300`) pause between chunks, and
checkpoints `queue.json` after every chunk - but this is now understood
as a crash-safety / not-hammering-the-pool measure, not the fix for the
failure rate itself. Both settings must be positive/non-negative - the
app refuses to start with an invalid value, since a threshold `<= 0`
would otherwise chunk the queue into zero batches and silently overwrite
`queue.json` with an empty list. Chunking only applies to the
queue-driven path (`discover_and_process.py`/`batch_runner.py`) - an
explicit URL list (e.g. a live `POST /batch/run` request) is never
paced, since a large one would otherwise hold the HTTP connection open
for the full cooldown duration. Because a large queue's total cooldown
time can exceed `DISCOVERY_LOCK_TTL_SECONDS`, `discover_and_process.py`
renews its lock lease after every chunk (`job_lock.renew_lock()`) so a
healthy long run doesn't look crashed to a concurrent invocation; if the
lease is ever found to belong to someone else mid-run, the run aborts
immediately rather than continuing to write against a lock it no longer
holds.

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

## Scheduled channel discovery (optional)

Instead of (or as well as) manually editing `queue.json`, you can point
the app at a set of YouTube channels and have it discover new uploads on
its own via each channel's public RSS feed.

### 1. Create `channels.json` in the Drive folder

```json
{
  "version": 1,
  "channels": [
    {
      "channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw",
      "name": "Google for Developers",
      "enabled": true
    },
    {
      "channel_id": "UCsomeSpanishChannelId",
      "name": "Some Spanish-language channel",
      "enabled": true,
      "languages": ["es", "en"]
    }
  ]
}
```

- `channel_id` is the stable `UC…` channel ID (not the `@handle`) — find
  it in a channel's page source, or via any "channel ID lookup" tool.
- `enabled: false` skips a channel without deleting its entry.
- `languages` is optional; when set, it overrides `TRANSCRIPT_LANGUAGES`
  for videos discovered from that channel only.

No deployment is needed to add, remove, enable, or disable a channel —
just edit `channels.json` in Drive, same as `queue.json`.

### 2. Deploy `discover_and_process.py` as a Railway Cron Job

One job does both steps every run: discover new uploads from enabled
channels, queue the ones not already in `_index.json` or `queue.json`,
then process the queue exactly like `batch_runner.py` does. Deploy it as
its own [Cron Job](https://docs.railway.com/guides/cron-jobs) service
with start command `python discover_and_process.py`, pointed at the same
environment variables (plus `channels.json` in the same Drive folder).

**Concurrency invariant:** `queue.json` and `_index.json` use unlocked
read-modify-write Drive operations that only tolerate one writer at a
time. Discovery and queue processing must therefore run as one
serialized job, never as independent, potentially-overlapping ones — do
not *also* enable `ENABLE_SCHEDULER` or deploy `batch_runner.py` on a
schedule alongside `discover_and_process.py`; pick one queue-processing
path. `discover_and_process.py` additionally takes out an advisory
Drive-based lock (`_discovery_lock.json` in the same folder) for its own
duration, so a second overlapping invocation of *itself* (e.g. a manual
run while a scheduled one is still going) exits immediately instead of
racing it. `DISCOVERY_LOCK_TTL_SECONDS` (default `1800`) controls how old
that lock must be before a new run assumes the previous one crashed and
proceeds anyway.

This lock is deliberately advisory, not a true distributed
compare-and-swap (that redesign - Drive revision/ETag preconditions plus
retry, or a transactional shared store - is explicitly out of scope for
this feature). Its acquire is a check-then-write that isn't atomic across
processes, and Drive additionally permits duplicate filenames in one
folder, so two near-simultaneous invocations could otherwise both believe
they hold the lock. Two things narrow that window: every lock carries a
random ownership token, so a run can only ever delete or take over a
lease it actually recognizes (a slow/crashed run can't steal or clear a
*different* run's active lock); and immediately after writing its lock,
a run re-checks that exactly one lock file exists and it's the one that
run just wrote, backing off if a concurrent writer is detected. Real
overlap (e.g. Railway Cron drift plus a manual run) is caught; a
sub-second true race at the Drive API level is not fully eliminated.

**Recovery:** if a run crashed and you're certain nothing is actually in
flight, you don't have to wait out the TTL — just delete
`_discovery_lock.json` from the Drive folder and the next run will
acquire the lock immediately.

### Livestreams

YouTube's RSS feed lists a livestream as soon as it starts, well before
it ends or captions exist for it. Without special handling, discovery
would try to fetch its transcript immediately, get "no_captions", and -
since that's normally a terminal status - drop the video for good,
losing it even after the stream ends and captions become available.

To avoid that, every video discovery queues carries a `first_seen_at`
timestamp, and a "no_captions" result for such a video is retried on
every subsequent run rather than treated as final, until it's older than
`NO_CAPTIONS_GRACE_HOURS` (default `24`) - long enough to cover even a
multi-hour livestream plus YouTube's post-stream caption-processing
delay. Past that window, it's treated as genuinely caption-less and
dropped, same as any other video. Videos added manually to `queue.json`
(no `first_seen_at`) get no grace period - "no_captions" is terminal for
them immediately, exactly as before this existed.

## Project layout

```
app/
  main.py          FastAPI app: /healthz, /transcripts, /batch/run
  pipeline.py       process_video(): the shared fetch-and-archive path
  batch.py          run_batch(): queue-driven or explicit-list batch runs
  youtube.py        URL parsing, transcript fetch, oEmbed title lookup, Markdown rendering
  drive.py          Drive upload/read + generic Drive file helpers
  queue_store.py    queue.json read/write helpers (plain URLs or {"url","languages","first_seen_at"} entries)
  channel_store.py  channels.json read helper (the discovery source registry)
  discovery.py      RSS feed fetch/parse + discover_and_enqueue(): queues unseen uploads
  job_lock.py       Drive-based advisory lock preventing overlapping discovery runs
  scheduler.py       optional in-process APScheduler wiring
  config.py         environment variable loading/validation
  models.py         request/response schemas
batch_runner.py     standalone entrypoint for a separate Railway Cron Job service
discover_and_process.py  standalone entrypoint: discover channel uploads, then process the queue
get_refresh_token.py  one-time local script to mint the Drive OAuth refresh token
```
