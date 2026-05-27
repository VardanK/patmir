# Cloud ASR migration — OpenRouter GPT-4o-transcribe (with forced-alignment fallback)

## Context

This is the next step in a per-stage cloud escalation: translate already moved to OpenRouter Gemini 2.5 Flash, TTS to OpenRouter Gemini 3.1 Flash TTS. ASR is the remaining major local stage. The goal is to set a stronger quality baseline — local Whisper Large-v3 (MLX) is good but still hallucinates on laughter/scream regions and needs gap-filler + blocklist scaffolding to be usable on Masha-style cartoons.

Constraints from the user:
- **OpenRouter only** for the active cloud route. Soniox / Gladia / Speechmatics / Deepgram (mentioned in deep-research) are off the table because they're not on OR.
- **Not enterprise-grade.** Single-user pipeline; concurrency, SLA, 20k-batch-job scenarios are irrelevant.
- **ElevenLabs Scribe** must be documented but not wired up yet — tier-gated by project policy (mirrors how ElevenLabs Multilingual v2 TTS is shelved at `enabled = false`).

Decisions (taken in the planning session):
- **Active default:** `openai/gpt-4o-transcribe` via OpenRouter. Best published Russian WER; native `timestamp_granularities=["word"]`; clean break from the Qwen3-ASR family we already tried locally and shelved over chain-of-thought interleaving (commit `c1b02c5`, doc `stages/asr/qwen3-asr-mlx.md`).
- **Alternates registered, disabled:** `qwen/qwen3-asr-flash`, `google/chirp-3`, `mistralai/voxtral-mini-transcribe`. Same module, different model id (mirrors the openrouter_claude_opus / openrouter_gpt5 pattern on translate).
- **Timestamp gap → forced alignment.** If a cloud transcript comes back without word timings (or with attention-drift), run `ctc-forced-aligner` locally on the vocals to recover precise word boundaries.
- **Local Whisper-MLX stays as fallback.** Keep `whisper_large_v3_mlx_fp16` registered for offline runs.
- **No new pipeline stage.** Forced alignment runs inside the ASR candidate when needed (not as its own stage), so the existing `stage.py` Protocol stays unchanged.

## Implementation plan

### 1. New candidate module — `patmir/pipeline/stages/asr/openrouter_stt.py`

Mirrors the structure of `patmir/eval/candidates/translation/openrouter.py` (the precedent for OpenRouter HTTP candidates).

- `class OpenRouterSTT` exposing `id`, constructor with `model`, `api_key_env`, `base_url`, `timeout_seconds`, `chunk_seconds`, optional `language` hint.
- `httpx.Client` instantiated once per `build()` (matches translate + TTS).
- `transcribe(audio: AudioTrack, ctx: JobContext) -> Transcript` does:
  1. Read `audio.path` (16 kHz mono WAV produced by `ingest` or vocals from `separate`).
  2. Chunk into ≤10-minute / ≤25 MB segments using `ctx.extra["vad_intervals"]` for clean split points when available; fixed-length with 1 s overlap otherwise.
  3. For each chunk, POST multipart `/v1/audio/transcriptions` to OpenRouter with `model`, `response_format=verbose_json`, `timestamp_granularities=["word"]`, optional `language=ctx.source_lang or self.language`.
  4. Handle HTTP / response-shape errors with the same `RuntimeError`-with-500-char-truncated-payload pattern used in the translate candidate.
  5. Stitch chunk results, offsetting word `start`/`end` by chunk start time.
  6. **Timestamp gap path:** if the parsed response is missing `words[*].start/end` for any chunk, invoke `_aligner.align_words(vocals_path, text, language)` and use its output. Log which path was taken per chunk.
  7. Group words into `Sentence` objects via the existing `patmir/pipeline/stages/asr/_sentence.py` helpers.
  8. Persist via `write_transcript(ctx.job_dir, transcript)`.
- `build(config: dict[str, Any])` factory function — mirrors translate.
- **No `transcribe_slice()` helper.** The runner's gap-filler will skip (acceptable — cloud model is stronger on laughter regions; hallucination blocklist still applies inside `_sentence.py`).

### 2. Forced-alignment helper — `patmir/pipeline/stages/asr/_aligner.py`

- Function `align_words(audio_path: Path, text: str, language: str) -> list[Word]`.
- Backed by `ctc-forced-aligner` (MMS-FA; multilingual including Russian). Lazy-loaded.

### 3. `patmir.toml` updates

- Flip `[pipeline].asr = "openrouter_gpt4o_transcribe"`.
- Add four new `[[stages.asr.candidates]]` blocks (active + three `enabled = false`).
- Add a fifth disabled block for `elevenlabs_scribe_v2` (documentation-only stub; module not implemented).
- Keep existing local entries untouched as offline fallbacks.

### 4. Dependency

- Add `ctc-forced-aligner` to `pyproject.toml`.

### 5. Documentation updates (Maintenance Contract)

- `docs/architecture/current-models.md` — flip ASR row.
- `docs/architecture/stages/asr/current.md` — replace config block; document forced-alignment fallback and absence of `transcribe_slice()`.
- New `docs/architecture/stages/asr/openrouter-stt.md` — model card.
- New `docs/architecture/stages/asr/elevenlabs-scribe.md` — documented-but-not-used card.
- `docs/architecture/stages/asr/whisper-mlx-large-v3.md` — status: local fallback.
- New ADR `docs/architecture/decisions/2026-05-26-cloud-asr-via-openrouter.md`.
- One-line entry in `docs/architecture/decisions.md`.
- `docs/architecture/quality-issues.md` — update Whisper-hallucination row.

## Critical files

- New: `patmir/pipeline/stages/asr/openrouter_stt.py`, `patmir/pipeline/stages/asr/_aligner.py`
- Edits: `patmir.toml`, deps file, the six docs above
- Reused: `patmir/pipeline/stages/asr/_sentence.py` (sentence grouping + hallucination blocklist), `patmir/runner.py` (drop-in `ASRStage`)
- Pattern reference: `patmir/eval/candidates/translation/openrouter.py`

## Verification

1. **Smoke probe.** curl `https://openrouter.ai/api/v1/audio/transcriptions` with `model=openai/gpt-4o-transcribe`, `response_format=verbose_json`, `timestamp_granularities=["word"]` on a short Russian clip. Confirm `words[]` array shape. If absent, forced-alignment becomes mandatory for every call.
2. **Chunking math** on a synthetic long WAV.
3. **End-to-end:** `OPENROUTER_API_KEY=<key> patmir dub <masha-2-5min-clip> --resume-from asr`. Inspect `runs/<job>/asr/transcript.json`.
4. **Downstream regression:** continue through translate → tts → mux → output.
5. **Fallback toggle test:** point `[pipeline].asr` at `openrouter_qwen3_asr_flash`; confirm clean behavior.
6. **Ear-test active config** on Masha clip.

## Out of scope

- Implementing the ElevenLabs Scribe candidate.
- A/B harness for the disabled alternates.
- Touching diarization or emotion stages.
- Cost-tracking telemetry.
