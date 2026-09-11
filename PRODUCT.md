# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

FastAPI serving Jinja templates, with HTMX for partial updates and polling. No build step, no `node_modules` in a Python repo. Chosen by the author over both plain vanilla JS and an SPA; consistent with `agent-os/product/tech-stack.md`, which specifies a "lightweight local web UI (FastAPI + a minimal HTMX or vanilla-JS front end)" and "no web framework beyond the minimal local UI."

Server-Sent Events are already wired for live job progress (`GET /run/{job_id}/progress`) and remain the transport for streaming updates; HTMX covers the rest.

## Users

**One user: the author**, operating the Patmir pipeline on his own workstation. He is the person who runs the models, reads the architecture docs, files the quality issues, and performs the ear-tests. He is technically expert in this system — it is his — so the interface owes him density and precision, not explanation.

His son — the product's actual beneficiary, an Eastern-Armenian-speaking child who wears hearing aids — **does not use this interface**. He watches finished dubs through the Chrome extension on YouTube, or as MKV files. Nothing in this surface is designed for him.

Two situations, both the same person:

- **At the workstation**, desktop browser, alongside a terminal and the running pipeline. The primary scene.
- **On another laptop over the LAN**, checking on a long job without walking to the machine.

## Product Purpose

Patmir translates foreign-language video into an Eastern Armenian (`hy_AM`) dub, so that a child who cannot follow the original audio can watch it. See `agent-os/product/mission.md` for the full product mission.

**This surface is the operator console for the Patmir server** (`patmir ui`, FastAPI on port 8765). Today that server ships two bare Jinja pages — a run list and a run detail page with a raw progress scroll — while the real operating surface is the CLI plus a Chrome extension. Substantial state the server already holds is visible nowhere: which job holds the GPU, what is queued behind it, what is cached and at which config hash, how a watch-along stream's buffer is faring against the viewer's playhead.

Success is that the author can answer "what is my machine doing, and does the output sound right?" from a browser, without a terminal and without reconstructing state from files on disk after the fact.

## Positioning

Not a product position — this is an internal tool with one user. What it must be true to instead:

The console is the **live** face of a system whose knowledge already lives in `docs/architecture/`. Those docs are the record; this is the instrument panel. It shows what is happening now and what is on disk now, and it does not restate what a document already explains better.

## Operating Context

**The server.** `patmir ui` runs a FastAPI app bound to loopback. It holds a `JobManager` (batch/quality jobs) and a `StreamManager` (watch-along), which **share one process-wide warm model cache and one `asyncio.Lock`** — only one pipeline fits on a 24 GB card, so a job submitted during a stream reports status `waiting` until the accelerator frees (ADR `2026-08-25-one-accelerator-at-a-time`). Models are prewarmed at boot.

**The two execution modes.** Quality mode (D1) runs all 9 stages in order over the whole video and caches one `dub.wav`. Watch-along (D3) produces VAD-snapped sections streamed behind a short buffer while the viewer watches. They have different records, different failure modes, and different diagnostics.

**How work arrives today.** The Chrome extension (`extension/`), the CLI (`patmir dub`, `patmir submit`), and `POST /jobs`. The console adds a fourth door.

**The ear-test loop, which is the project's real bottleneck.** A large share of the rows in `docs/architecture/quality-issues.md` sit at "awaiting an ear verdict." The loop today is: run the pipeline, `open runs/<id>/out.mkv` or `afplay runs/<id>/dub.wav`, listen, note a timestamp, hand-transcribe the finding into markdown. Every step of that happens outside any interface.

**Storage roots**, all configurable via `[storage]` in `patmir.toml` and resolved by `patmir/paths.py`: `runs_root` (live working dirs, one per job, all intermediates), `cache_root` (finished dubs keyed by source + config hash), `downloads_root` (yt-dlp), `state_root` (bearer token + job index), optional `library_root` (published deliverables).

**Machine profiles.** One committed `patmir.toml` serves an Apple Silicon laptop and a WSL2 + RTX 3090 workstation, selected by `PATMIR_PROFILE`. Which candidate is live for a stage depends on the profile, and the console runs on the CUDA box.

## Capabilities and Constraints

### Confirmed scope

All of the following are in. Each is backed by state the server already holds.

**Operations**
- Current jobs, job history, and cross-job error feed (`JobRecord`: status, stage states, live stage/pct/msg, download pct, error, timings).
- **GPU queue and live load** — who holds the accelerator lock, who is `waiting`, plus live VRAM / power / temperature / throttle state. `patmir/telemetry.py` already has an NVML reader with an `nvidia-smi` fallback; it is currently used only by `patmir bench`.
- **Submit from the browser** — URL or local path, mode (`quality` / `watch-along`), audience level, force-rerun. `POST /jobs` already accepts all of these.
- **Per-stage timeline** — the 9 stages with wall time, build time and RTF per stage, from `manifest.json` (`StageStatus.started_at` / `finished_at` / `build_s`), replacing today's raw progress scroll.
- **Processing time details** — the breakdown behind the timeline: warm vs. cold build cost, per-stage RTF, `other_s` (wall minus the stage sum), `cpu_s` including child processes. `patmir/pipeline/probe.py` and `patmir/bench.py` already produce this shape; `docs/architecture/performance.md` is its written form.
- **Active pipeline config panel** — live profile, candidate id per stage, `config_hash`, resolved storage roots. `HANDOFF.md` currently instructs the reader to verify this by pasting a Python snippet into a shell.

**Streams and cache**
- **Watch-along stream monitor** — buffer-ahead vs. playhead, per-section RTF, time to first audio, the speaker registry. Data exists in `StreamRecord` (`first_audio_s`, `download_s`, `prelude_s`, `opening_dub_s`, `sections_done`, `dubbed_s`, `duration`, `playhead`, `speakers`) and in each `sections.json` entry (per-stage walls, `build_s`, `rtf`, `buffer_ahead_s`, `other_s`), read through the shared reader `patmir/pipeline/streaming/timing.py`.
- **Cache browser** — every cached dub with source, key, size, config hash, mode, audience, build time; play it, reveal its run, or invalidate it. `cache.invalidate()` exists and is reachable from no interface.
- **Stale-config flags** — mark cache entries built under a config hash that no longer matches the live one. `CACHE_VERSION` moved 1 → 2 on 2026-09-04, so this question is live.

**Ear-test loop**
- **Artifact browser with inline audio** — a run's files with players for `dub.wav`, `vocals.wav`, and each per-sentence `tts/*.wav`. `GET /run/{job_id}/file/{rest}` already serves anything under a job dir.
- **Transcript viewer** — per sentence: source text, Armenian target, speaker id, emotion bin, timing, each row playable.
- **Run comparison (A/B)** — two runs of the same source side by side, sentence for sentence. The workflow that `runs/emotion-ab/` and `runs/ab-redraw/` were assembled by hand to serve.
- **Ear-test note capture** — record a verdict against a run or a timestamp, persisted. This is a **new persistence surface**; nothing in the server stores it today, and its storage location is an open decision.

### Constraints

- **Auth is one shared bearer token** (`patmir/ui/auth.py`), auto-generated on first `patmir ui` and printed to stderr. **No login form is in scope** — the author has confirmed the console will not be port-forwarded for now. Loopback and LAN only.
- **"Connected clients" is thinner than the phrase suggests.** With a single shared token the server cannot tell one client from another. What is honestly knowable: last-seen per stream (`StreamRecord.last_seen_at`), request IP, user agent. This is a **connections list, not identities**. Real per-client identity would require a token-per-client change, which is not decided.
- **Only one pipeline runs at a time.** The console must make a `waiting` job legible rather than looking hung — the extension's older fallback rendered this state as "Running… (0/0)".
- **Job history is capped** at `_INDEX_MAX_ENTRIES = 200` in the persisted index.
- **The console is not the extension.** It does not play dubbed video over YouTube; that is `extension/`'s job and stays there.
- **A polished consumer UI is explicitly not a product goal** (`mission.md`). Density and diagnostic precision are what this surface owes; decoration is not.

### Terminology

Stage names (`ingest`, `separate`, `diarize`, `emotion`, `asr`, `translate`, `tts`, `mux`, `output`), `RTF` (real-time factor; wall / source seconds, >1 = slower than realtime), `config_hash`, `cache_key`, quality mode vs. watch-along, section, prelude, speaker registry, audience level, machine profile, accelerator lock. Defined in `docs/architecture/glossary.md` and `data-contracts.md`.

### Open decisions

- Where ear-test notes persist (job dir sidecar, `state_root`, or a small database) and whether they feed `quality-issues.md`.
- Whether per-client identity is worth a token-per-client change.

## Brand Commitments

- **Name:** Patmir (Armenian **պատմիր**, "tell the story").
- **Interface language: English throughout** for this surface. This is deliberate and differs from the Chrome extension, which uses Eastern Armenian for the labels the son and family read (Որակ, Զուգահեռ, Չեղարկել) with English for operator hints. The console has one reader, and he reads the logs and the docs in English.
- **Armenian text is content, not chrome.** Translated output, transcripts and speaker names render in Armenian script and must be typeset legibly — the interface around them is English.
- No logo, wordmark, or brand palette exists. Nothing here is a binding visual constraint.

## Evidence on Hand

- **A running server with real endpoints** — `patmir/ui/app.py` (routes), `jobs.py`, `stream.py`, `cache.py`, `auth.py`, `logs.py`.
- **Two incumbent Jinja templates** — `patmir/ui/templates/index.html` and `run.html`. Minimal, unstyled, and the honest starting point.
- **A shipped Chrome extension** with real UI decisions already made — `extension/popup.html`, `options.html`, `overlay.js`.
- **A live architecture knowledge base** — `docs/architecture/` (pipeline, current-models, decisions, quality-issues, performance, runs, data-contracts, glossary), actively maintained under a contract in `CLAUDE.md`.
- **Real telemetry code** — `patmir/telemetry.py`, `patmir/pipeline/probe.py`, `patmir/bench.py`.
- **Real run artifacts on disk** under `runs/` (gitignored) and cached dubs under `~/.patmir/cache/`.

**Absences future work must not fabricate:** there are no other users, no customers, no testimonials, no pricing, no deployment story, no uptime record, and no usage metrics. Patmir is one person's workstation tool with an audience of one child.

## Product Principles

1. **Show the live machine; let the docs explain it.** `docs/architecture/` is the system of record and is excellent. The console shows current state — what is running, what is queued, what is on disk — and links out rather than restating.
2. **Make invisible state visible.** The recurring failure in this project's history is state that existed but was surfaced nowhere: a `waiting` job that looked hung, a stream stall reconstructed from disk days later, a progress bar whose field was never emitted. Every view should reduce that class of failure.
3. **Serve the ear-test.** Quality here is judged by listening, and the listening loop currently runs entirely outside any interface. Shortening it is the highest-leverage thing this surface can do.
4. **Density over decoration.** One expert reader, at a workstation, wanting numbers. Precision, alignment and typographic clarity earn their keep; ornament does not.
5. **Never claim what is not measured.** This codebase's documentation is scrupulous about the difference between a measurement and an assumption. The interface must hold the same line — an unknown reads as unknown, not as zero.

## Accessibility & Inclusion

No product-specific standard has been established for this surface, and its single user has not stated a requirement. Ordinary web accessibility hygiene applies — real focus states, keyboard reachability, sufficient contrast at workstation viewing distance, and correct rendering of Armenian script in transcript and output views.
