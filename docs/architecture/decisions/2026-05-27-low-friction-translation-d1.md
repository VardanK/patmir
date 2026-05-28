# 2026-05-27 — Low-friction translation UX uses D1 (pre-process + cache)

## Status

Accepted, 2026-05-27. Implemented in slice 1 (server + Chrome extension + tests). Slices 2–3 (SSE, drift correction, macOS Quick Action, dedicated yt-dlp ingest candidate, cache CLI) are deferred but planned in `agent-os/specs/2026-05-27-1200-low-friction-translation-d1/`.

## Context

Phase 2.x landed the quality work (multi-voice + emotion + voice cloning + per-sentence loudness match). `quality-issues.md` shows no active issues as of 2026-05-27. The remaining gap is **friction**: the only way to dub a video is `patmir dub <file>` on the CLI.

The user asked for "a Chrome plugin that detects video playing on a web page and automatically translates by routing the video stream to our server and replaces the audio stream with translated one." That implied real-time. A buffer of up to 30 s was acceptable.

## Constraints surfaced during exploration

- The pipeline runner is **strictly sequential** (`patmir/pipeline/runner.py:67-100`). Each stage blocks the next.
- End-to-end throughput on the most recent run (Fallout S02E01-5min) is **~0.64× realtime**: 300 s of video → 469 s of wall-clock. `separate` (Demucs, 178 s) + `tts` (ElevenLabs, 145 s) together are 72% of that.
- Sequential structure means no TTS WAV appears until **~280 s into wall-clock** (everything before TTS must finish). A 30 s buffer covers ~10% of that.
- Sentence boundaries cross any fixed time-window chunking (translate / TTS fragments produce garbled Armenian). Mitigations exist (VAD-aligned chunking, lookahead) but require pipeline-parallel execution to be worth doing.
- **DRM streams** (Netflix / Prime / Disney+) cannot be intercepted by a browser extension — EME-protected MSE buffers are opaque. The only path is system-audio capture (BlackHole / ScreenCaptureKit), which is a different product.

## Decision

Three designs evaluated; **D1 chosen**.

| ID | Description | First-dub latency | Steady-state lag | Engineering | Verdict |
| -- | ----------- | ----------------- | ---------------- | ----------- | ------- |
| **D1** | Pre-process + cache. Click extension → server runs current batch pipeline → cached `dub.wav` plays in-page on completion (and instantly on rewatch). | ~5–10 min for a 22-min episode | 0 (cached) | Small — server + extension + cache; no pipeline changes | **Accepted** |
| **D2** | Stream existing per-sentence TTS WAVs as they're written, no cache. | ~5–10 min before first WAV | Falls further behind at 0.64× | Small | Rejected — half-step; no rewatch benefit |
| **D3** | Pipeline-parallel runner + VAD-aligned chunking + parallel TTS. Dub-ahead with 30 s buffer. | ~30 s if throughput hits ≥1.0× | Stable, leads playback | **Large** — runner rewrite, every stage candidate accepts chunked input, Demucs replacement or skip-on-dialogue heuristic, parallel TTS, mux refactor | Rejected — multi-week rewrite that unwinds Phase 2.x global-diarization + global-ref-clip-picking assumptions, with material quality regression vs the current batch output |
| DRM-via-system-audio | macOS ScreenCaptureKit / BlackHole capture for Netflix etc. | n/a | n/a | Large (macOS permission + audio bridge) | Deferred indefinitely — different product |

## Why D1 is the right finalization of Phase 2

- It makes the current high-quality pipeline low-friction without throwing away the Phase 2.x quality work.
- The son's actual viewing pattern is repeat-watched cartoons — cache-on-rewatch is the dominant value driver.
- It composes: D3 can be tried later as a Phase 3 R&D bet without invalidating any code shipped here.

## What changed

- New `patmir/ui/{cache,auth,jobs}.py` and extended `patmir/ui/app.py` with `POST /jobs`, `GET /cache/lookup`, `GET /jobs/{id}`, `GET /cache/{key}/dub.wav`, `GET /cache/{key}/metadata.json`. Loopback-only bind + bearer-token auth (`~/.patmir/secret`, chmod 600).
- Global URL-keyed cache at `~/.patmir/cache/<hash>/` keyed by source-id + pipeline-config hash. Audio-tuning options (`music_bed_gain`, `voice_gain_db`, `sentence_afade_seconds`) deliberately excluded from the key so ear-test tweaks don't bust the cache.
- yt-dlp URL ingest implemented as **pre-download in the `/jobs` handler** to a known dir (`~/.patmir/downloads/<cache_key>/`). The existing `ffmpeg` ingest candidate runs unchanged. A proper `yt_dlp` ingest candidate is deferred to slice 3.
- `patmir submit <video | --url URL>` CLI subcommand.
- `extension/` — MV3 Chrome extension (popup + content script + service worker + options page). SW-relay pattern works around the YouTube HTTPS → loopback HTTP mixed-content block.

## Revisit triggers

- A frontier-quality streaming Armenian TTS appears AND streaming Demucs/diarize models close the quality gap → reopen D3.
- Cache size becomes unmanageable on disk → ship LRU eviction (`patmir cache clear`, slice 3 polish).
- An EA-capable system-audio bridge becomes interesting for DRM content → score the DRM-via-system-audio path as a separate product.
- The son's actual usage shows almost no rewatch (single-view content) → the cache value drops; reconsider whether the wait UX needs more work, or pre-cache-overnight queues.
