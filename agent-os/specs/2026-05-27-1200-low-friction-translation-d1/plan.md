# Phase 2 finalization — low-friction translation UX (D1)

## Context

Phase 2.x landed the quality work (multi-voice + emotion + voice cloning + loudness match) and `quality-issues.md` shows no active issues as of 2026-05-27. The remaining gap is **friction**: today the only way to dub a video is `patmir dub <file>` on the CLI. The user wants a path that's closer to "click a button on a YouTube tab and a few minutes later the dub plays in the page."

Constraints forced the design:

- **Real-time / streaming is not feasible** with the current pipeline shape. The runner is strictly sequential (`patmir/pipeline/runner.py:67-100`), `separate` (Demucs) is single-threaded per job, and end-to-end throughput on the most recent run is **~0.64× realtime** (300 s of video → 469 s of wall-clock, `separate` 178 s + `tts` 145 s dominate). A 30 s initial buffer doesn't begin to cover the ~280 s before the first TTS sentence lands.
- **DRM streams (Netflix / Prime / Disney+) are out of scope.** Browser extensions cannot read EME-protected MSE buffers; the only way to dub DRM is system-audio capture, which is a different product.
- **Pipeline-parallel rewrite (D3) is too big** to bundle here — it would unwind Phase 2.x assumptions (global diarization, global ref-clip picking, sentence boundaries) and is a multi-week phase of its own.

Chosen design — **D1: Pre-process + cache**:

1. Local FastAPI server (extends the existing `patmir/ui/app.py`) accepts a YouTube URL or local file, runs the current batch pipeline unchanged, and persists the finished dub in a global, URL-keyed cache.
2. Chrome MV3 extension on YouTube tabs detects `<video>` elements, queries the cache, and either shows "Translate this" (kicks off a job, shows progress) or "Play dub" (cache hit — plays the cached `dub.wav` in-page over Web Audio, muting the original).
3. macOS Quick Action / `patmir submit <file>` CLI handles local files outside the browser, hitting the same server.

The UX is "first view: queue + walk away; rewatch: instant in-page dub." That matches the son's actual viewing pattern (repeat-watched cartoons).

## Scope

**In Phase 2 finalization:**
- Local FastAPI server gains `POST /jobs`, `GET /cache/lookup`, `GET /jobs/{id}`, `GET /cache/{key}/dub.wav`, `GET /cache/{key}/metadata.json`. Existing `/run/{id}/progress` SSE is reused.
- Loopback-only bind + bearer-token auth (token in `~/.patmir/secret`, chmod 600).
- Global URL-keyed cache at `~/.patmir/cache/<hash>/` keyed by source-id + pipeline-config hash.
- yt-dlp URL ingest. Cheapest path: **pre-download in the `/jobs` handler** to a temp file, then call the existing `run_job` with that path. Adds a new ingest **candidate** later if/when slicing matters; not blocking for D1.
- Chrome MV3 extension (popup + content script + service worker + Web Audio sync).
- `patmir submit <file>` CLI subcommand + macOS Finder Quick Action that pipes a file path to the server.

**Out of scope (defer):**
- Real-time / streaming pipeline (would be Phase 3).
- DRM-protected sources.
- Cache eviction policy (LRU). Cache grows; user clears manually with `patmir cache clear`.
- Subtitle overlay from `timing.json`.
- Music-bed-only / dub-only playback toggles.
- Cross-machine sharing of cache.

## Architecture

```
┌──────────────────────────────┐         ┌──────────────────────────────────┐
│ Chrome MV3 extension         │         │ Local FastAPI (127.0.0.1:8765)   │
│  - content.js (on youtube)   │ HTTP+SSE│  - patmir/ui/app.py (existing)   │
│  - sw.js (token, SSE relay)  │◀───────▶│  - patmir/ui/jobs.py     (new)   │
│  - popup.{html,js}           │ bearer  │  - patmir/ui/cache.py    (new)   │
│  - audio_sync.js (Web Audio) │ token   │  - patmir/ui/auth.py     (new)   │
└──────────────────────────────┘         └─────────────┬────────────────────┘
       │                                               │
       │ play dub.wav synced to <video>.currentTime    │ run_job (existing)
       ▼                                               ▼
┌──────────────────────────────┐         ┌──────────────────────────────────┐
│ Page <video> element (muted) │         │ Existing batch pipeline          │
│ + Web Audio AudioBuffer       │         │ (ingest → ... → output)           │
└──────────────────────────────┘         └─────────────┬────────────────────┘
                                                       │ on success
                                                       ▼
                                         ┌──────────────────────────────────┐
                                         │ ~/.patmir/cache/<hash>/           │
                                         │   dub.wav, metadata.json, ...    │
                                         └──────────────────────────────────┘
```

## Code changes

### Server (new files under `patmir/ui/`)

- **`patmir/ui/auth.py`** (new) — `require_token` FastAPI dependency. Reads token from `~/.patmir/secret` (auto-generated on first server start, 32 hex bytes, chmod 600). Accepts `Authorization: Bearer <t>` header **or** `?token=<t>` query (SSE-friendly because `EventSource` can't set headers). Also rejects requests with non-loopback `Origin`.
- **`patmir/ui/cache.py`** (new) — single source of truth for cache layout + key derivation (see "Cache" section below). Functions: `cache_key(source_id, cfg) -> str`, `lookup(key) -> Path | None`, `materialize(key, job_dir) -> None` (copy curated artifacts out of `runs/<id>/` into the cache dir), `lookup_by_url(canonical_url) -> str | None`.
- **`patmir/ui/jobs.py`** (new) — in-process `asyncio.Queue` with concurrency=1 (Demucs + MPS + Scribe rate limits make parallel runs unsafe). Worker pops a `JobSpec`, runs `run_job` in `asyncio.to_thread`, then `cache.materialize` on success. Tracks `JobRecord { job_id, cache_key, source_url|source_path, status: queued|running|done|failed }` in memory + flushes to `~/.patmir/jobs.json` for crash recovery.
- **`patmir/ui/app.py`** (modify) — factory now takes `_create_app(runs_root, cache_root)`. New endpoints (all gated by `auth.require_token`):

  | Method | Path | Purpose |
  |---|---|---|
  | `GET` | `/cache/lookup?url=<canonical>` | Fast cache-hit probe on every YouTube tab load. Returns `{cached, cache_key?, duration?, created_at?}`. |
  | `POST` | `/jobs` | Body `{source_url?, source_path?, source_lang?}`. Returns `{job_id, cache_key, status}`. If already cached, returns `done` immediately without re-running. |
  | `GET` | `/jobs/{job_id}` | `{status, cache_key, stage_states: {...}}`. |
  | `GET` | `/run/{job_id}/progress` | (existing SSE — unchanged) |
  | `GET` | `/cache/{key}/dub.wav` | Bytes; `Range:` supported; `Cache-Control: immutable`. |
  | `GET` | `/cache/{key}/metadata.json` | Inspection. |

  CORS middleware allows `chrome-extension://*` (extension origin known after we publish/install). No general HTTPS origins. Token auth means CORS is belt-and-suspenders.

### Pipeline (small, optional in slice 1)

- **`POST /jobs` handler downloads URLs with yt-dlp before calling `run_job`.** Saves to `<tempdir>/<urlhash>.mp4`, passes that `Path` to `run_job`. Existing `ffmpeg` ingest candidate works unchanged. **No new ingest candidate needed for slice 1.**
- **`patmir/pipeline/stages/ingest/yt_dlp.py`** (new, **slice 2/3 only**) — second ingest candidate. Reads URL from `ctx.extra["source_url"]`, downloads to `<job>/ingest/source.mp4`, then re-uses the existing ffmpeg PCM-extract logic via a small `_audio.py` helper factored out of `ffmpeg.py:22-40`. Plumbing change to runner: in `run_job`, if `source_url` is passed, set `ctx.extra["source_url"]` before the stage loop. Slice 1 deliberately skips this — the pre-download path in the HTTP handler is simpler and unblocks the demo.

### CLI

- **`patmir/cli.py`** (modify) — add `patmir submit <video> [--config ...]` subcommand. Reads token from `~/.patmir/secret`, POSTs to `127.0.0.1:8765/jobs` with `source_path`, prints `job_id`. Also add `patmir cache ls` / `patmir cache clear` (slice 3).
- **`patmir/ui/__init__.py`** — `serve(port, runs_root, cache_root)` signature update.

### Chrome extension (new top-level `extension/`)

- **`extension/manifest.json`** — MV3. Permissions: `activeTab`, `scripting`, `notifications`, `storage`. `host_permissions: ["http://127.0.0.1:8765/*", "https://*.youtube.com/*"]`. Service worker + popup + content script registered.
- **`extension/sw.js`** — service worker; owns the bearer token (set via `options.html`, stored in `chrome.storage.local`). All `fetch` + `EventSource` to the local server originate here (avoids HTTPS↔HTTP mixed-content blocks in the YouTube page context — see "Mixed-content gotcha" below). Relays SSE events to popup/content via `chrome.runtime.sendMessage`. Fires `chrome.notifications.create` on job completion.
- **`extension/content.js`** — runs on `youtube.com`. On `DOMContentLoaded`: extract canonical URL (`yt:<videoId>`), check for DRM (host blocklist + `video.mediaKeys` probe — if DRM, popup shows "DRM-protected, cannot dub"), ask SW for cache status. Owns the page-context Web Audio graph: mutes the `<video>`, plays the dub via `AudioBufferSourceNode`, listens to `video.{play,pause,seeking,ratechange,timeupdate}`.
- **`extension/audio_sync.js`** — encapsulates the sync algorithm (see below).
- **`extension/popup.{html,js}`** — three states: "Translate this" / "Translating… <stage> <pct>" / "Play dub" toggle. Subscribes to progress via SW relay.
- **`extension/options.html`** — one-time bearer-token entry.

### macOS drop-target

- **`patmir/macos/quick-action.workflow`** (new bundle, slice 2) — Automator Quick Action that runs `patmir submit "$@"` on a selected file in Finder. Zero new app shell. User installs by double-clicking in `~/Library/Services/`.

## Cache key + layout

**Key derivation** (in `patmir/ui/cache.py`):

```
source_id = (
    canonicalize_url(url)            # e.g. "yt:<videoId>" — strips all tracking params
    if url else
    f"file:{size}:{sha256(first_1MiB + last_1MiB).hex()}"
)
config_hash = sha256(json.dumps({
    "stages": {n: cfg.entries[n].id for n in STAGE_ORDER},
    "models": {n: cfg.entries[n].model_id_fields() for n in STAGE_ORDER},
    "target_lang": "hy",
}, sort_keys=True)).hexdigest()[:8]

cache_key = sha256(f"{source_id}|{config_hash}").hexdigest()[:16]
```

Audio-tuning options (`music_bed_gain`, `voice_gain_db`, `sentence_afade_seconds`) are deliberately **excluded** so ear-test tweaks don't bust the cache. Stage selectors + model ids are included so a model swap invalidates correctly.

**Layout**:

```
~/.patmir/
  secret                        # 32-byte hex, chmod 600
  jobs.json                     # in-flight job records, for crash recovery
  cache/
    by-url.json                 # {canonical_url: cache_key} — fast lookup
    <cache_key>/
      dub.wav                   # copied (not symlinked) from runs/<job>/dub.wav
      metadata.json             # {cache_key, source_id, source_url?, pipeline, created_at, job_id, duration}
      transcript.json           # copy of runs/<job>/translated.json (future overlay use)
```

`runs/<job_id>/` remains the canonical workspace and stays ephemeral; `~/.patmir/cache/` is the stable, curated export.

## Web Audio sync algorithm (`audio_sync.js`)

Goal: play the cached `dub.wav` such that audio position tracks `<video>.currentTime` within ±80 ms.

1. **Load**: SW fetches `dub.wav` bytes (Authorization header — `fetch` in SW origin sidesteps mixed-content). Transfers the `ArrayBuffer` to the content script via `chrome.runtime.sendMessage`. Content script's `AudioContext.decodeAudioData` → `AudioBuffer`.
2. **Mute** the page video: `video.muted = true` (don't touch `video.volume` — leave that to the user / page UI).
3. **Drive playback off video state**:
   - `play`: create a fresh `AudioBufferSourceNode`, set `playbackRate.value = video.playbackRate`, `source.start(0, video.currentTime)`. Record `audioStartCtxTime = audioCtx.currentTime`.
   - `pause`: `source.stop()`; null it out.
   - `seeking` → `seeked`: stop the source; create a new one at the new `currentTime`.
   - `ratechange`: `source.playbackRate.setValueAtTime(video.playbackRate, audioCtx.currentTime)`.
4. **Drift correction** (1 s `setInterval`):
   - `expected = video.currentTime`
   - `actual = audioStartOffset + (audioCtx.currentTime - audioStartCtxTime) * playbackRate`
   - If `|drift| > 80 ms` → hard-resync (stop + restart at `video.currentTime`).
   - If `30 ms < |drift| ≤ 80 ms` → soft-correct: nudge `playbackRate` by ±0.5% for one second.

Duration mismatch (dub vs source): mux's per-sentence stretch keeps total dub duration within ~ε of source, so we don't need pad / truncate logic beyond letting the source EOF naturally.

## Mixed-content gotcha (critical)

YouTube is HTTPS; `fetch('http://127.0.0.1:8765/...')` from a content script's page context is **blocked by mixed-content policy**. Two viable workarounds:

- **Preferred**: do **all** `fetch` + `EventSource` calls in the **service worker** (`sw.js`). The SW runs in `chrome-extension://<id>` origin, which is exempt from page-mixed-content. Transfer the audio bytes to the content script via `chrome.runtime.sendMessage`.
- **Alternative**: explicit loopback `host_permissions` lifts the block for content-script-initiated requests in some Chrome versions. Worth a one-hour spike to confirm; if it works, the audio path simplifies (no big `ArrayBuffer` over `sendMessage`).

Start with the SW-relay path. If the spike confirms the alternative, switch to it.

## Phasing inside D1

**Slice 1 — minimum end-to-end demo on YouTube (the "wow" milestone)**
- `auth.py` + `cache.py` + `jobs.py`.
- `POST /jobs`, `GET /cache/lookup`, `GET /jobs/{id}`, `GET /cache/{key}/dub.wav`.
- yt-dlp pre-download in the `/jobs` handler (no new ingest candidate yet).
- Extension: `manifest.json`, `sw.js`, `content.js`, `popup.{html,js}`, `audio_sync.js`, `options.html`. Polling `/jobs/{id}` once a second for slice 1.
- Demo path: open YouTube → popup says "Translate this" → click → wait → popup says "Play dub" → click → in-page Web Audio overlay plays dub synced to video.

**Slice 2 — production UX**
- SSE in popup (replaces polling, reuses existing `/run/{id}/progress`).
- `chrome.notifications` on completion.
- DRM detection in `content.js`.
- Drift-correction loop in `audio_sync.js`.
- `patmir submit` CLI + macOS Quick Action.

**Slice 3 — polish**
- `yt_dlp` ingest candidate (cleaner than handler-side pre-download for very large videos).
- `patmir cache ls / clear / show` CLI.
- Subtitle overlay from `transcript.json`.
- Music-bed-only / dub-only toggles.

## Maintenance contract impact (per CLAUDE.md)

This is a delivery / UX milestone, not a model swap. Doc updates required during implementation:

- **`pipeline.md`** — add an "Ingest sources" subsection noting that ingest now accepts both file paths and (via the HTTP handler) URLs.
- **`current-models.md`** — add an `ingest` candidate id `yt_dlp` row when slice 3 lands.
- **`decisions/2026-05-27-low-friction-translation-d1.md`** — new ADR documenting the D1 choice + the rejection of D3 (real-time) and DRM (system-audio bridge), with the throughput data as evidence.
- **`decisions.md`** — one-line entry pointing at the new ADR.
- **`quality-issues.md`** — no change (this isn't a quality issue).

## Verification

End-to-end manual:

1. `patmir ui --port 8765` (modified to also create `~/.patmir/secret` on first start).
2. Load `extension/` as unpacked in `chrome://extensions`; paste the secret into the options page.
3. Open a short public YouTube video (≤5 min).
4. Popup shows "Translate this". Click. Status switches to "Translating… <stage>".
5. Wait the expected ~7-8 min wall-clock.
6. Popup switches to "Play dub". Click. Video mutes; Armenian dub plays in-page, synced to `<video>.currentTime`.
7. Pause / seek / 1.25× playback — dub follows within 80 ms drift.
8. Reload the page. Popup shows "Play dub" immediately (cache hit).
9. Run `patmir submit ~/Movies/test.mkv`. Job appears in cache; second submit returns cached job_id immediately.

Programmatic:

- Pytest covering `cache.cache_key` invariance under unrelated `[pipeline.options]` changes and inversion when a model id changes.
- `cache.lookup_by_url` round-trip.
- `auth.require_token` rejects missing / wrong tokens with 401.

## Tasks

**Task 1 — Save spec documentation**

Create `agent-os/specs/2026-05-27-1200-low-friction-translation-d1/`:

- `plan.md` — this file.
- `shape.md` — shaping notes (scope, decisions log, rejected alternatives D2/D3/DRM).
- `references.md` — points at `patmir/ui/app.py`, `patmir/pipeline/runner.py:30-100`, `patmir/pipeline/stages/tts/adapter.py:101-151`, `patmir/pipeline/stages/ingest/ffmpeg.py`, `patmir/pipeline/context.py:58-74`.
- `visuals/` — empty (no mockups provided).

**Task 2 — Slice 1: server + extension (demo end-to-end)**

Implement the slice 1 list above. Acceptance: YouTube clip → in-page dub.

**Task 3 — Slice 2: production UX**

SSE, notifications, DRM detection, drift correction, drop-target.

**Task 4 — Slice 3: polish**

yt-dlp ingest candidate, cache CLI, subtitle overlay, dub/bed toggles.

**Task 5 — Documentation per maintenance contract**

ADR + `decisions.md` entry + `pipeline.md` ingest note.
