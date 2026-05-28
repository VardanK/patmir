# References — Low-friction translation UX (D1)

## Server extension points

### Existing FastAPI app
- **Location:** `patmir/ui/app.py`
- **Relevance:** D1 extends this file, doesn't replace it.
- **Key patterns to keep:** `_create_app(runs_root)` factory shape (now takes a second `cache_root`); the `/run/{job_id}/progress` SSE endpoint stays unchanged and is reused for job progress (the SSE works against `progress.jsonl` which the existing pipeline writes — no change needed).
- **Patterns to borrow:** `_safe_job_dir(job_id)` path-escape guard (`app.py:30-36`) — apply the same pattern to `_safe_cache_dir(key)`.

### Pipeline programmatic entry
- **Location:** `patmir/pipeline/runner.py:30-100`
- **Relevance:** `run_job(source: Path, cfg, runs_root, resume_job_dir, resume_from, source_lang) -> Manifest` is callable directly from `jobs.py`'s worker. No refactor needed for slice 1.
- **Key fact:** the `source` parameter is a `Path`. For URL jobs, the `/jobs` handler pre-downloads to a tempfile and passes that path.

### TTS per-sentence WAV emission
- **Location:** `patmir/pipeline/stages/tts/adapter.py:101-151` (writes each sentence WAV at line 135 as it completes)
- **Relevance:** Not used in D1 directly (we wait for the full mux). Will become load-bearing if D3 / streaming is ever built.

### Per-job caching example
- **Location:** `patmir/pipeline/stages/asr/elevenlabs_scribe.py:68-105`
- **Relevance:** Shows the existing per-job cache pattern (`ctx.job_dir / "scribe" / "result.json"`). The new D1 cache is **global** (`~/.patmir/cache/`), not per-job; this reference is just the conceptual prior art.

### Pipeline config / candidate entries
- **Location:** `patmir/pipeline/config.py:42-66`
- **Relevance:** `PipelineConfig.entries: dict[str, dict[str, Any]]` is what the cache key derivation hashes. Each entry includes `id`, `module`, and any model-specific knobs (e.g., `model_id`, `voice`, `language_code`). Hashing the dict captures everything that affects output.

### Job context shape
- **Location:** `patmir/pipeline/context.py:58-74`
- **Relevance:** `JobContext.extra: dict[str, Any]` is the cross-stage hand-off bag. Currently unused by D1 (we pre-download URLs in the handler), but the slice 3 `yt_dlp` ingest candidate will read `ctx.extra["source_url"]` from here.

## Pipeline ingest

### File-only ffmpeg ingest
- **Location:** `patmir/pipeline/stages/ingest/ffmpeg.py`
- **Relevance:** Stays unchanged in slice 1. Slice 3's `yt_dlp.py` candidate factors lines 22-40 (the ffmpeg PCM-extract) into a shared `_audio.py` helper.

## CLI

### Existing CLI entry
- **Location:** `patmir/cli.py:73-117` (the `patmir dub` subcommand)
- **Relevance:** Add a sibling `patmir submit <video>` subcommand that POSTs to the local server.

## Stage order constant
- **Location:** `patmir/pipeline/stage.py` (`STAGE_ORDER` constant)
- **Relevance:** Cache key derivation iterates `STAGE_ORDER` to hash candidate entries in a stable order.

## Decision log this spec touches

- `decisions/2026-05-24-local-first-strict.md` — D1 honors local-first; server binds 127.0.0.1.
- `decisions/2026-05-24-eastern-armenian-only.md` — target lang hardcoded in cache key.
- `quality-issues.md` — no active issues impacted; this spec doesn't touch quality.

## Mocks / visuals

None provided.
