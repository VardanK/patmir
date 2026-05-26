# References for Multi-voice (diarization + per-speaker cloning)

No prior diarization or speaker-identification code exists in this repository (`grep -ri "diariz\|pyannote\|speaker_embed" patmir/` returns nothing). The work below references existing pieces of Phase-0 / Phase-1 we lean on.

## Similar Implementations

### OmniVoice MLX TTS (proves EA cloning works per-call)

- **Location:** `patmir/eval/candidates/tts/omnivoice_mlx.py:48–75`
- **Relevance:** `model.generate(...)` already accepts `ref_audio` and `ref_text` per call. The single-voice usage today only stores them at constructor time, but the underlying API is fully per-call. Per-sentence voice switching is a one-line change to `synthesize`.
- **Key patterns to borrow:** how `mlx-audio` consumes ref clips (mono WAV, any sample rate the model accepts) and how generation chunks are concatenated into PCM bytes.

### Demucs separate + Silero VAD masking (the clean stem we slice refs from)

- **Location:** `patmir/pipeline/stages/separate/htdemucs.py:55–245` (especially `_split_vocals_with_vad` at 163–245).
- **Relevance:** Produces `speech_vocals` — Demucs vocal stem with Silero VAD applied per-sample plus 20 ms cosine fades at boundaries. Music-free *and* non-speech-vocalization-free. Exactly the input we want for reference-clip slicing.
- **Key patterns to borrow:** the per-sample-mask + fade approach for any future "slice this region cleanly" need; resampling-via-numpy.interp pattern for 16 kHz mono.

### Runner ctx.extra plumbing (how diarization reaches ASR)

- **Location:** `patmir/pipeline/runner.py:131–177` (the `_run_stage` `"asr"` branch, especially `ctx.extra["vad_intervals"]` at line 157).
- **Relevance:** Phase 1 already established the pattern of "stage A produces a sidecar; runner attaches it to `ctx.extra`; stage B reads it." We reuse the same pattern verbatim for diarization → ASR.
- **Key patterns to borrow:** the post-stage hook (run additional logic after `stage_obj.transcribe` returns, before storing the result) — that's where `apply_speaker_tags` and `backfill_ref_text` hang.

### Stage data contracts + protocol (where `speaker_id` lives)

- **Location:** `patmir/pipeline/stage.py:54–179`.
- **Relevance:** Defines the typed dataclasses passed between stages and the `STAGE_ORDER` tuple. We extend it with the new diarization dataclasses, the `DiarizeStage` Protocol, and a `speaker_id` field on `Sentence`/`TranslatedSentence`.
- **Key patterns to borrow:** `@runtime_checkable` Protocol shape for stages; `field(default_factory=list)` for optional collections.

### Pipeline config / candidate loader (how to register the new stage)

- **Location:** `patmir/pipeline/config.py:29–37, 88–129`.
- **Relevance:** `_STAGE_NAMESPACE` is the registry. Adding one line registers `"diarize"` as a stages-namespace stage with `[[stages.diarize.candidates]]`.
- **Key patterns to borrow:** the `module.build(config)` entry-point convention; lazy stage instantiation via `build_stage`.

## External references (libraries, not code)

- **pyannote-audio 3.1** — `pyannote/speaker-diarization-3.1` pipeline (HF). Returns an `Annotation` object with speaker turns plus an `OverlapDetection` track when configured.
- **mlx-audio** — already in the dep tree; OmniVoice model loader.
