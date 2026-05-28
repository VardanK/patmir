# Low-friction translation UX (D1) — Shaping notes

## Scope

Close the *friction* gap left at the end of Phase 2.x. The pipeline already produces good Eastern Armenian dubs; the only way to trigger one is `patmir dub <file>` on the CLI. We want one-click translation from a YouTube tab and a drag-target for local files, both backed by a single local FastAPI server.

## Decisions

- **D1 chosen over D2/D3.** D1 = pre-process + cache. D3 (real-time / streaming pipeline rewrite) rejected because the current runner is strictly sequential, `separate` is single-threaded per job, and end-to-end throughput is ~0.64× realtime on a 5-minute clip — a 30 s initial buffer cannot cover the ~280 s before the first TTS sentence lands. D3 would also unwind Phase 2.x quality assumptions (global diarization, global ref-clip picking). D2 (stream existing TTS WAVs without cache) was rejected as a half-step that doesn't deliver dub-ahead and adds no rewatch value.
- **DRM (Netflix / Prime / Disney+) is explicitly out of scope.** A Chrome extension cannot read EME-protected MSE audio buffers. The only path to dub DRM is system-audio capture (BlackHole / ScreenCaptureKit) — a different product, deferred indefinitely.
- **Up to 30 s initial buffering accepted** but moot now: D1 isn't dub-ahead. First view waits minutes; rewatch is instant from cache.
- **Pre-download in the `/jobs` handler (slice 1)** instead of building a yt-dlp ingest candidate. Cheapest path that keeps the existing `ffmpeg` ingest candidate unchanged. yt-dlp-as-ingest-candidate becomes slice 3 work for very large videos where progressive download matters.
- **Concurrency = 1.** Demucs + MPS + ElevenLabs / OpenRouter rate limits make parallel runs unsafe. Queue, don't parallelize.
- **Loopback-only bind + bearer token.** Server binds 127.0.0.1; auth dependency rejects non-loopback `Origin` and requires `Authorization: Bearer <token>` (or `?token=` query for SSE). Token in `~/.patmir/secret`, chmod 600, auto-generated on first start.
- **Global cache lives outside `runs/`** at `~/.patmir/cache/<key>/`. `runs/` is the canonical ephemeral workspace; the cache is the curated, stable export keyed by content + pipeline config.
- **Cache key = source-id + selected-candidate-ids + per-candidate config dict.** Audio-tuning options (`music_bed_gain`, `voice_gain_db`, `sentence_afade_seconds`) deliberately excluded so ear-test tweaks don't bust the cache.
- **Mixed-content workaround = service-worker relay.** YouTube is HTTPS; `fetch` from a content script's page context to `http://127.0.0.1` is blocked. SW runs in `chrome-extension://<id>` origin, exempt from page mixed-content; SW relays the audio bytes to the content script.
- **Drift bound ±80 ms** for Web Audio sync. Hard-resync above that; soft `playbackRate` nudge between 30–80 ms.

## Context

- **Visuals provided:** None.
- **Reference implementations in the codebase:**
  - `patmir/ui/app.py` — existing FastAPI server (read-only progress UI). Extension point: `_create_app(runs_root)` factory + `/run/{job_id}/progress` SSE endpoint stays unchanged.
  - `patmir/pipeline/runner.py:30-100` — `run_job(source, cfg, ...)` is the clean programmatic entry into the pipeline.
  - `patmir/pipeline/stages/tts/adapter.py:101-151` — TTS already emits per-sentence WAVs incrementally (not needed for D1 but useful context).
  - `patmir/pipeline/stages/ingest/ffmpeg.py` — current ingest, file-only.
  - `patmir/pipeline/context.py:58-74` — `JobContext` shape including `extra: dict[str, Any]` for cross-stage hand-offs.
  - `patmir/pipeline/stages/asr/elevenlabs_scribe.py:68-105` — example of per-job caching (Scribe result at `<job>/scribe/result.json`).
- **Product alignment:** Phase 2 is the multi-voice / emotion-transfer milestone. The roadmap puts low-friction UX as the natural finalization once quality is good (`quality-issues.md` has no active issues as of 2026-05-27). This spec is the friction half of the same phase.

## Standards Applied

None of the indexed `agent-os/standards/` standards apply directly. The work is bounded by existing project conventions:

- `CLAUDE.md` "Maintenance contract": after this lands, update `pipeline.md` (ingest sources note), `current-models.md` (yt_dlp candidate row in slice 3), and add a new ADR `decisions/2026-05-27-low-friction-translation-d1.md`.
- Local-first policy from `decisions/2026-05-24-local-first-strict.md`: this spec does not alter the per-stage cloud-opt-in posture — the new HTTP server is loopback-only.
- `decisions/2026-05-24-eastern-armenian-only.md`: target lang stays hardcoded `hy` in the cache key derivation.

## Rejected alternatives

| ID | Description | Why rejected |
| -- | ----------- | ------------ |
| D2 | Stream existing TTS WAVs as they're written, no cache | Doesn't deliver dub-ahead (throughput is 0.64× realtime → first sentence lands ~280 s in). No rewatch benefit. Half-step. |
| D3 | Pipeline-parallel + VAD-aligned chunking, dub-ahead with 30 s buffer | Multi-week rewrite. Unwinds Phase 2.x global ref-clip picking + global diarization. Quality regresses. Better as Phase 3. |
| DRM-via-system-audio | macOS ScreenCaptureKit / BlackHole capture for Netflix/Prime/Disney+ | Different product. macOS permission + audio-bridge complexity. Defer indefinitely. |
| Per-job cache (no global cache) | Just use `runs/<id>/` | Rewatch wouldn't work; no content-hash dedup; doesn't address friction. |
| Cache key includes all `[pipeline.options]` | More precise invalidation | Would bust the cache on every `music_bed_gain` ear-test tweak. Bad ergonomics. |
| Pre-download via yt-dlp ingest candidate from day one | Cleaner abstraction | Forces a small runner refactor (URL on `ctx.extra` or new `JobContext` field) before any UX ships. Deferred to slice 3. |
