# Phase 1 — MVP pipeline (Mode A on local files)

## Context

Phase 0 closed with a **soft go on Phase 1, scoped to cartoons** (see `agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/verdict.md`). The local Eastern-Armenian stack — `nllb_200_distilled_600m` for translation, `piper_hye` (`hy_AM-gor-medium`) for TTS — cleared the "watchable for hours?" bar on 20/20 utterances. Idiom translation and Russian-source narration are known weak spots; both are expected to improve once the GPU re-eval runs (`nllb_200_3.3b`, `madlad_400`, `xtts_v2`, `bark`) but neither blocks Phase 1.

Phase 1 builds the actual MVP per `agent-os/product/roadmap.md` §"Phase 1": end-to-end offline batch translation for **local video files** with the six logical stages as swappable modules behind typed interfaces, plus a minimal local UI. Definition of done is the author's son finding a 20-minute cartoon watchable end-to-end after the pipeline runs on a local video file.

### Decisions (from shaping)

| Decision | Value | Why |
|---|---|---|
| GPU re-eval | Deferred; plumb now on Phase 0 winners | Stages are swappable — re-eval becomes a `patmir.toml` config change. Ships sooner. |
| Hardware | Mac (MPS); device set per stage via config | Phase 0 ran on MPS; CUDA workstation still pending. Phase 1 code must stay device-agnostic. |
| Source langs | EN + RU from day 1 | Whisper auto-detects language; NLLB-600M handles both (RU narration weak but acceptable for cartoons). |
| ASR | `faster-whisper` running `whisper-large-v3-turbo`, int8 | Turbo is `large-v3` pruned to 4 decoder layers — much faster, minor quality loss. Same runtime as `large-v3`, so switching is a `model_id` change in `patmir.toml`. `large-v3` registered as a deferred fallback for quality-critical jobs. |
| Alignment | Sentence-level: pad with silence + `pyrubberband` time-stretch capped at ±20% | Roadmap: "rough alignment is fine, lip sync is not a goal." Heavy stretching wrecks TTS audio — capped. |
| Container | MKV now, output stage designed so MP4 is a near-future swap | MKV cleanly handles AR-default + original 2nd + SRT. MP4 = same stage, second backend. |
| UI | CLI + minimal FastAPI localhost page | Matches `tech-stack.md` §"Local UI". |
| Job dir | `runs/<YYYYMMDD-HHMMSS>-<source-stem>/` | Mirrors existing `eval_runs/<ts>/` convention; preserves history. |

### What's reused from Phase 0

The Phase 0 candidate-module pattern (`patmir/eval/candidates/...`) already exposes the right entrypoints:

- Translation candidates have `class XxxTranslator: def translate(self, text: str, source_lang: str) -> str`. Loader: `build(config: dict) -> Translator`.
- TTS candidates have `class XxxTTS: def synthesize(self, text: str) -> bytes` (WAV bytes). Loader: `build(config: dict) -> TTS`.
- Both are loaded by `patmir/eval/_loader.py::load_candidates(config_path, stage)` from `[[eval.<stage>.candidates]]` tables, filtered by `enabled = true`.

Phase 1 will **reuse these exact candidate modules** — no rewrites. The pipeline picks one candidate per stage by id from the existing `[[eval.*.candidates]]` tables via a new `[pipeline]` selector section.

### Open weakness to accept

NLLB-600M idiom translation is unusable (1.3/1.3/1.5 in Phase 0). Phase 1 will not try to fix idioms — they'll come out wrong on the dogfood episode and that's fine. Idiom quality is a GPU-re-eval problem.

### Modern HuggingFace candidates (post-Phase 0) — registered, not enabled

A fresh scan after Phase 0 surfaced new candidates worth tracking in `patmir.toml`. None change Phase 1's v1 plumbing — they're added as **disabled** entries so the GPU re-eval picks them up:

- **Translation:** `Gen2B/HyGPT-10b-it` (July 2025, Gemma-2-9b base, 10B Armenian tokens, supports EN/RU/hy translation, Flores 79.33%, ~20 GB VRAM at fp16, license allows personal local use). Dialect not labeled but training corpus is Armenia-centric — must be ear-judged. Likely the strongest local translator candidate ever for this stack. Add as `hygpt_10b_it`, `device = "cuda"`, `enabled = false`.
- **Translation (fast path):** `Helsinki-NLP/opus-mt-en-hy` and `Helsinki-NLP/opus-mt-ru-hy` — 2020 Helsinki-NLP / OPUS-MT, explicitly Eastern Armenian (hye), ~80 MB each, MPS-friendly, BLEU 16.6 on Tatoeba en→hy. Modest quality but tiny and fast. Add as `opus_mt_en_hy` / `opus_mt_ru_hy`, `device = "mps"`, `enabled = false` — could be flipped on for a quick A/B vs NLLB-600M without GPU.
- **ASR:** `large-v3` registered as a non-default candidate `faster_whisper_large_v3` for jobs where turbo's quality loss matters. Default stays `whisper-large-v3-turbo`.
- **TTS:** No new viable local Eastern Armenian model since Phase 0. Piper `hy_AM-gor-medium` stays v1. Notes on the search:
  - **XTTS-v2 does NOT support Armenian.** Its current model card lists exactly 17 languages (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi). Armenian is not among them. The Phase 0 `tech-stack.md` and `patmir.toml` overstated this — *correction:* XTTS-v2 is **not** a Phase 1 candidate without full fine-tuning, which is out of scope. **Decision:** keep the `xtts_v2` entry in `patmir.toml` (disabled) with an inline comment explaining the language gap, so the choice is documented and future readers don't re-discover it. Tech-stack.md gets a matching correction note.
  - **Bark** — **decision:** stays deferred to the GPU re-eval pass alongside the translation candidates. Not in Phase 1 plumbing scope. Bark's dialect handling for Armenian is unverified and the eval cost isn't worth the schedule slip on Phase 1.
  - **F5-TTS / MeloTTS / OpenVoice v2 / Parler-TTS** — none claim Armenian. Skipped.
  - **`facebook/mms-tts-hye`** still 404. Only `mms-tts-hyw` (Western, wrong dialect) exists.
  - **`davit312/Armenian-speech-hy` dataset** (Nov 2024) exists — useful for a future Piper or VITS fine-tune as a Phase 2 voice-variety track, but not Phase 1.
  - **AUA Jun 2025 paper** on Armenian STT/TTS (`cse.aua.am/files/2025/06/Arm2Eng-STT-TTS_2025.pdf`) is referenced in search results but the PDF returned 404; no public model release found.

Not added (intentionally): `Chillarmo/whisper-large-v3-turbo-armenian` — Armenian-only ASR, useless for transcribing EN/RU source. `ArthurMalajyan/seamless-m4t-v2-large-asr-hyw` — Western Armenian, wrong dialect.

---

## Task 1 — Save spec documentation

Create `agent-os/specs/<YYYY-MM-DD-HHMM>-phase-1-mvp-pipeline/` with:

- `plan.md` — this file copied in.
- `shape.md` — scope, decisions table, references, "open weakness" note.
- `standards.md` — `agent-os/standards/index.yml` is currently empty, so this file just records "no standards applied yet" with a one-liner pointer to add them later.
- `references.md` — pointers to:
  - `patmir/eval/_loader.py` (candidate loader contract)
  - `patmir/eval/candidates/translation/nllb_200_distilled_600m.py` (Translator interface, reused)
  - `patmir/eval/candidates/tts/piper_hye.py` (TTS interface, reused)
  - `agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/verdict.md` (winners + audio samples)
- `visuals/` — empty; no mockups for a CLI/FastAPI MVP.

---

## Task 2 — Stage protocols, job context, manifest, runner

New module `patmir/pipeline/` with:

- `stage.py` — `@runtime_checkable Protocol` per stage. Each protocol declares the one method the stage runs (e.g. `IngestStage.extract(self, video_path: Path, ctx: JobContext) -> AudioTrack`). Six protocols: `IngestStage`, `ASRStage`, `TranslateStage`, `TTSStage`, `MuxStage`, `OutputStage`. Typed dataclasses for the inter-stage contracts: `AudioTrack`, `Transcript` (word-level `Word` + sentence-grouping helper), `TranslatedTranscript`, `SynthesizedAudio` (per-sentence WAV paths), `MuxedAudio` (single rendered AR track), `OutputArtifacts` (paths).
- `context.py` — `JobContext` carrying `job_dir: Path`, `source_path: Path`, `source_lang: str | None` (set by ASR), a `progress(stage: str, pct: float, msg: str)` callback, and `cfg: PipelineConfig`. Progress writes a row to `<job_dir>/progress.jsonl`.
- `manifest.py` — `Manifest` dataclass with status per stage (`pending`/`running`/`done`/`failed`), timestamps, artifact paths. Persisted as `<job_dir>/manifest.json`; the FastAPI page reads this.
- `runner.py` — `run_job(source: Path, cfg: PipelineConfig, resume_from: str | None) -> Manifest`. Creates `runs/<ts>-<stem>/`, instantiates each stage from config, runs in sequence, writes the manifest after each step. `resume_from` lets a stage be re-run by reading prior artifacts from disk.

The runner does **not** wrap a stage's typed input/output around the candidate's raw `translate`/`synthesize` calls — that wrapping lives in stage adapters (Tasks 5 and 6).

---

## Task 3 — `[pipeline]` section in `patmir.toml` + loader

Append to `patmir.toml`:

```toml
[pipeline]
# Single source of truth for which candidate runs each stage.
# Each value is a candidate `id` defined in the corresponding [[eval.<stage>.candidates]] table.
ingest    = "ffmpeg"
asr       = "faster_whisper_large_v3_turbo"
translate = "nllb_200_distilled_600m"
tts       = "piper_hye"
mux       = "ffmpeg_pad_stretch"
output    = "mkv_default"

[pipeline.options]
stretch_cap = 0.20      # ±20% time-stretch cap
asr_int8    = true
keep_intermediate = true
```

Also extend the existing `[[eval.translation.candidates]]` and `[[eval.asr.candidates]]` tables with the post-Phase-0 candidates (all `enabled = false` — they're for the GPU re-eval, not Phase 1 plumbing):

```toml
[[eval.translation.candidates]]
id        = "hygpt_10b_it"
enabled   = false
module    = "patmir.eval.candidates.translation.hygpt"
model_id  = "Gen2B/HyGPT-10b-it"
device    = "cuda"

[[eval.translation.candidates]]
id        = "opus_mt_en_hy"
enabled   = false
module    = "patmir.eval.candidates.translation.opus_mt"
model_id  = "Helsinki-NLP/opus-mt-en-hy"
source_lang = "en"
device    = "mps"

[[eval.translation.candidates]]
id        = "opus_mt_ru_hy"
enabled   = false
module    = "patmir.eval.candidates.translation.opus_mt"
model_id  = "Helsinki-NLP/opus-mt-ru-hy"
source_lang = "ru"
device    = "mps"

[[eval.asr.candidates]]
id        = "faster_whisper_large_v3"
enabled   = false
module    = "patmir.pipeline.stages.asr.faster_whisper"
model_id  = "large-v3"
device    = "auto"
compute_type = "int8"
```

The corresponding candidate modules (`patmir/eval/candidates/translation/hygpt.py`, `opus_mt.py`) ship as small stubs in Phase 1 so the table loads cleanly — the actual eval runs happen during the GPU re-eval, not in Phase 1.

Add an inline comment to the existing `xtts_v2` entry in `patmir.toml` explaining that XTTS-v2's current model card does not include Armenian in its 17 supported languages — keeps the entry as a documented dead end rather than a viable deferred candidate. Apply the same correction to `agent-os/product/tech-stack.md` §"Stage 4 — Eastern-Armenian TTS".

The two **new** stages with their own candidate registries (ingest, ASR, mux, output) get `[[eval.<stage>.candidates]]`-style tables too, even though they aren't evaluated like translation/TTS — same loader, same `build(config)` contract:

```toml
[[eval.asr.candidates]]
id        = "faster_whisper_large_v3_turbo"
enabled   = true
module    = "patmir.pipeline.stages.asr.faster_whisper"
model_id  = "large-v3-turbo"
device    = "auto"      # auto → mps if available, else cpu
compute_type = "int8"
```

Same for `ingest`, `mux`, `output`. This keeps **one** loading mechanism across all six stages.

New helper `patmir/pipeline/config.py::load_pipeline_config(toml_path) -> PipelineConfig` that:
1. Reads `[pipeline]` to get the chosen candidate id per stage.
2. Calls the existing `load_candidates(toml_path, stage)` for that stage, filters by id, returns the single built object.
3. Aborts loudly if the chosen id isn't enabled in its `[[eval.<stage>.candidates]]` table.

---

## Task 4 — Stage 1: Ingest

New module `patmir/pipeline/stages/ingest/ffmpeg.py`:

- `build(config) -> FFmpegIngest`
- `FFmpegIngest.extract(video_path, ctx) -> AudioTrack` — calls `ffmpeg -i <video> -vn -ac 1 -ar 16000 -c:a pcm_s16le <ctx.job_dir>/audio.wav`. Returns `AudioTrack(path=..., sample_rate=16000, duration=...)`.
- Reads source duration via `ffprobe` for progress reporting.

`ffmpeg` and `ffprobe` invoked via `subprocess.run`; no `ffmpeg-python` dependency needed for these calls.

---

## Task 5 — Stage 2: ASR

New module `patmir/pipeline/stages/asr/faster_whisper.py`:

- `build(config) -> FasterWhisperASR` — loads `WhisperModel(model_id, device=resolved_device, compute_type=config["compute_type"])`.
- `FasterWhisperASR.transcribe(audio_track, ctx) -> Transcript` — runs `model.transcribe(audio_track.path, word_timestamps=True, language=None)`. Sets `ctx.source_lang = detected_language` (the runner uses this to route to the right translator language pair).
- `Transcript` carries word-level timing; provides `to_sentences(max_gap_s=0.6) -> list[Sentence]` for downstream stages. Sentence grouping: split on `.!?` in word texts, or on a gap > `max_gap_s` between consecutive words.
- Writes `<job_dir>/transcript.json` (full word list) and `<job_dir>/sentences.json` (grouped) for resume + UI inspection.

Add `faster-whisper>=1.0` to the `local` extra in `pyproject.toml`.

---

## Task 6 — Stage 3: Translate (adapter over Phase 0 candidate)

New module `patmir/pipeline/stages/translate/adapter.py`:

- `build(config) -> SentenceTranslator` — `config` carries the already-built `Translator` from `patmir/eval/candidates/translation/...` (the `[pipeline]` loader hands it in). No new candidate code.
- `SentenceTranslator.translate(transcript, ctx) -> TranslatedTranscript` — iterates `transcript.to_sentences()`, calls `inner.translate(sentence.text, source_lang=ctx.source_lang)` for each, attaches the AR text + original timing to each entry.
- Writes `<job_dir>/translated.json`.

Note: NLLB-600M is single-utterance; batching is left for a future GPU swap. Each sentence is one call.

---

## Task 7 — Stage 4: TTS (adapter over Phase 0 candidate)

New module `patmir/pipeline/stages/tts/adapter.py`:

- `build(config) -> SentenceTTS` — same pattern as translate adapter; wraps the built `piper_hye` candidate.
- `SentenceTTS.synthesize(translated, ctx) -> SynthesizedAudio` — for each translated sentence, calls `inner.synthesize(ar_text) -> bytes`, writes `<job_dir>/tts/<idx>.wav`, returns a list of `(sentence, wav_path, wav_duration_s)`.

WAV duration is read back from the file header (already 22.05 kHz mono PCM from Piper) — no decode needed.

---

## Task 8 — Stage 5: Mux (pad + time-stretch)

New module `patmir/pipeline/stages/mux/ffmpeg_pad_stretch.py`:

- `FFmpegPadStretch.mux(synth, ctx) -> MuxedAudio` — for each sentence:
  - Compute target window = `sentence.end - sentence.start` from ASR timing.
  - Compute required stretch ratio = `wav_duration / target_window`. Clamp to `[1 - cap, 1 + cap]` where `cap = ctx.cfg.options.stretch_cap` (default 0.20).
  - If ratio is in `[0.98, 1.02]`, skip stretching.
  - Apply via `pyrubberband.time_stretch` (preserves pitch).
  - If, after stretch, the WAV is still shorter than the window: pad with silence after it.
  - If, after stretch, it's still longer than the window: let it overrun (lip sync is not a goal); the next sentence's window starts when the prior one ends, not at the source timing — drift is acceptable.
- Concatenates all sentence WAVs into `<job_dir>/dub.wav` at the source's sample rate (resample on write).

Add `pyrubberband>=0.3` to the `local` extra. (`rubberband-cli` must be installed system-side — document in `README` / `pyproject` notes.)

---

## Task 9 — Stage 6: Output (MKV first, MP4-ready)

New module `patmir/pipeline/stages/output/mkv.py`:

- `MKVOutput.write(muxed, source_video, ctx) -> OutputArtifacts` — runs ffmpeg:
  ```
  ffmpeg -i <source_video> -i <dub.wav>
         -map 0:v -map 1:a -map 0:a
         -c:v copy -c:a:0 aac -c:a:1 copy
         -metadata:s:a:0 language=hye -metadata:s:a:0 title="Armenian (Patmir)"
         -metadata:s:a:1 language=<src> -metadata:s:a:1 title="Original"
         -disposition:a:0 default -disposition:a:1 0
         <job_dir>/out.mkv
  ```
- Also writes an Armenian SRT from the translated transcript and muxes it as the default subtitle track.
- Returns `OutputArtifacts(video=out.mkv, subtitles=out.hy.srt, manifest=manifest.json)`.

Designed so `MP4Output` becomes a sibling module later with the same `OutputStage` protocol — only the ffmpeg flags change (`-c:a:0 aac`, mp4 sub track gotchas handled separately).

---

## Task 10 — CLI

`patmir/cli.py` + register in `pyproject.toml` `[project.scripts]` as `patmir = "patmir.cli:main"`.

- `patmir dub <video> [--resume-from <stage>] [--keep-intermediate / --no-keep-intermediate]` — runs the pipeline, prints a per-stage progress bar (one line per stage with `[░░░░░░░░░░]` ASCII and pct), prints final paths.
- `patmir runs ls` — lists `runs/*/` with status and source.
- `patmir runs show <ts>` — prints the manifest for a run.

Progress comes from `JobContext.progress(...)` writing to `progress.jsonl`; the CLI tails that file in a thread.

---

## Task 11 — FastAPI local UI

`patmir/ui/app.py`:

- FastAPI app, binds `127.0.0.1:8765` only.
- `GET /` — lists `runs/` directory entries, status from each manifest.
- `GET /run/<ts>` — shows the manifest, per-stage progress, links to intermediate artifacts (`transcript.json`, `translated.json`, per-sentence TTS WAVs, final `out.mkv`).
- `GET /run/<ts>/progress` — Server-Sent Events stream tailing `progress.jsonl` for live updates.

Single Jinja template; vanilla JS for the SSE stream. No build step. Launched via `patmir ui` CLI subcommand.

Add `fastapi`, `uvicorn`, `jinja2` to a new `ui` extra in `pyproject.toml`.

---

## Task 12 — Dogfood test (definition of done)

- Pick a 20-minute English-source cartoon episode the son already wants to watch.
- Run `patmir dub <episode>.mkv`.
- Watch the resulting `out.mkv` with the son.
- **Pass = son watches the full episode in Armenian audio without asking to switch back.**
- Capture: failure modes (sentences that dropped sync, words Piper mispronounced, sentences NLLB mangled). Write findings into the spec folder as `dogfood.md` — feeds into Phase 2 priorities and the GPU re-eval target list.

---

## Verification

End-to-end on a short clip (~2 min) before the 20-min dogfood:

1. `uv pip install -e ".[local,ui]"` (and `brew install rubberband ffmpeg` if not present).
2. `patmir dub samples/clip-en-2min.mkv` — should produce `runs/<ts>-clip-en-2min/out.mkv`.
3. Open `out.mkv` in mpv: confirm AR audio is default, original is selectable, AR subtitle track is present.
4. `patmir ui`, browse `http://127.0.0.1:8765`, confirm run listed with all six stages green.
5. Stage-isolation check: run the same input but with `--resume-from translate` (delete `translated.json` first) — should rerun translate, TTS, mux, output and skip ingest+ASR.
6. RU-source clip: same flow on a 2-min Russian-source clip; confirm ASR detects `ru`, translator gets `source_lang="ru"`, output is Armenian.
7. Stretch-cap sanity: pick a sentence where Piper output is 1.6× the source window; confirm dub.wav has that sentence stretched by exactly the cap (1.2×) and the rest is overrun (not chopped).

The full 20-min cartoon dogfood (Task 12) is the actual definition-of-done; the verification steps above are pre-checks to catch obvious breakage before the long run.
