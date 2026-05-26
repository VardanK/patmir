# Phase 2 — Multi-voice (diarization + per-speaker cloning)

## Context

Phase 1 ships a working English→Eastern-Armenian dub but uses a **single global voice** for the whole episode. OmniVoice MLX (the active TTS) is a zero-shot voice cloner — `model.generate(text, ref_audio, ref_text, ...)` already takes the reference per call (`omnivoice_mlx.py:52`), so EA voice cloning works today. What's missing is **who spoke when** in the source, and **which reference clip** to use per sentence.

This phase adds:

1. **Diarization** of the source's Demucs vocal stem → per-speaker turn boundaries (with overlap detection) via `pyannote-audio 3.1`.
2. **Speaker-tagged ASR sentences** — each `Sentence` carries a `speaker_id`; sentences spanning two speakers are split at the boundary when feasible, else assigned to the dominant speaker.
3. **Automatic per-speaker reference clip extraction** — for each detected speaker, pick the longest clean turn, slice from the VAD-masked vocal stem, derive `ref_text` from the ASR word stream inside that window. Save as `<job_dir>/diarize/refs/<spk>.wav`.
4. **Per-sentence TTS conditioning** — `omnivoice_mlx.synthesize(text, ref_audio, ref_text)` swaps reference per sentence based on `speaker_id`.

Outcome: drop a cartoon, get each character cloned into a distinct Armenian voice automatically.

## Decisions (locked)

- **Diarizer:** `pyannote-audio 3.1`. SOTA accuracy + native overlap detection. One-time HF token + license accept on `pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1`; all inference local after.
- **Reference clip selection:** fully automatic. No `patmir.toml` override knobs this phase.
- **Sentences spanning speakers:** split at boundary if both halves contain ≥1 ASR word; else assign dominant.
- **No noise / RMS floor checks on reference clips** — Demucs vocal stem + Silero VAD masking already give us clean speech (the user explicitly confirmed background bleeding is not a concern). Trust the existing artifacts.
- **Local-first strict** — pyannote inference is fully local after one-time model download.
- **Eastern Armenian only** — diarization is language-agnostic; nothing changes.

## Audio inputs (important detail from existing pipeline)

The separate stage (`stages/separate/htdemucs.py`) emits **two relevant stems**:

- `SeparatedAudio.vocals` — full Demucs vocal stem (preserves soft-onset speech; also contains screams/laughs).
- `SeparatedAudio.speech_vocals` — same stem with Silero VAD applied as a per-sample mask + 20 ms cosine fades at boundaries. Music-free *and* non-speech-vocalization-free.
- `SeparatedAudio.vad_intervals` — `(start, end)` list of speech windows.

**Diarize on `vocals`** — mirrors the existing ASR decision (`runner.py:139–158`): the full stem preserves soft onsets pyannote needs for accurate turn boundaries. Post-filter pyannote turns against `vad_intervals` to drop ghost-speaker segments inside scream/laugh regions.

**Slice reference clips from `speech_vocals`** — cleanest possible audio (Demucs music removal + VAD non-speech removal). Trim ~100 ms inward from each end of the chosen turn so the cosine-fade edges never land in the ref clip.

## Architecture

New pipeline order (inserts one stage):

```
ingest → separate → diarize → asr → translate → tts → mux → output
                       │
                       └── writes diarize/turns.json + diarize/refs/<spk>.wav
```

The runner attaches the diarization result to `ctx.extra["diarization"]` (same pattern as the existing `ctx.extra["vad_intervals"]`). ASR candidates stay unchanged; speaker tagging happens in the runner immediately after `stage_obj.transcribe(...)` returns.

## Critical files

### New files

- `patmir/pipeline/stages/diarize/__init__.py`
- `patmir/pipeline/stages/diarize/adapter.py` — Stage entry; writes `diarize/turns.json` + `diarize/refs.json` + ref WAVs; coordinates pyannote candidate + ref picker.
- `patmir/pipeline/stages/diarize/pyannote.py` — pyannote 3.1 candidate. Loads `pyannote/speaker-diarization-3.1` with HF token from env, runs on `SeparatedAudio.vocals`, returns `(turns, overlaps)`. Same `device = "auto"` MPS/CUDA/CPU resolution as `htdemucs.py:25–36`.
- `patmir/pipeline/stages/diarize/ref_picker.py` — picks one reference clip per speaker. Algorithm:
  1. For each speaker, list their turns.
  2. Drop turns that intersect any `OverlapInterval`.
  3. Drop turns shorter than `min_ref_seconds` (default 3.0).
  4. Pick the longest remaining turn. Trim 100 ms inward each side.
  5. Slice the window out of `speech_vocals` → `diarize/refs/<spk>.wav` (mono, 16 kHz — resample from Demucs's 44.1 kHz).
  6. Build `ref_text` by joining `Word.text` for words whose midpoint falls inside the window. *(Requires ASR to have run first — see "Ordering note" below.)*
- `patmir/pipeline/stages/asr/speaker_tag.py` — `apply_speaker_tags(transcript: Transcript, diarization: Diarization) -> Transcript`. Lives next to ASR because it operates on `Transcript`. Algorithm per sentence:
  1. Compute overlap-duration with each speaker turn.
  2. If one speaker covers ≥85% of the sentence duration → tag with that speaker.
  3. Else find the boundary time. Split words at the first word whose `start` ≥ boundary. If both halves contain ≥1 word, emit two re-indexed `Sentence`s with text = `" ".join(w.text)` per half. Else fall back to dominant speaker.

### Ordering note

Reference clip extraction needs ASR words for `ref_text`. The `diarize` stage runs **before** ASR (it has to, so ASR sentences can be tagged). So we split the ref-picker into two passes:
- **At diarize time:** pick the per-speaker ref *window* (start/end) and save the sliced WAV.
- **After ASR + speaker tagging:** the runner backfills `ref_text` for each speaker by joining the ASR words inside that window. Persist `diarize/refs.json` at this point.

This keeps the stage order clean (diarize → asr) and avoids a circular dependency.

### Modified files

- `patmir/pipeline/stage.py`
  - New dataclasses: `SpeakerTurn{start, end, speaker_id}`, `OverlapInterval{start, end}`, `SpeakerRef{speaker_id, ref_audio: Path, ref_text: str, source_start: float, source_end: float}`, `Diarization{turns: list[SpeakerTurn], overlaps: list[OverlapInterval], refs: dict[str, SpeakerRef], sample_rate: int}`.
  - Add `speaker_id: str | None = None` to `Sentence` and `TranslatedSentence`.
  - Add `DiarizeStage` Protocol: `diarize(separated: SeparatedAudio, ctx) -> Diarization` (with `refs` initially having empty `ref_text` — backfilled post-ASR).
  - Insert `"diarize"` into `STAGE_ORDER` between `"separate"` and `"asr"`.

- `patmir/pipeline/config.py` — add `"diarize": ("stages", "diarize")` to `_STAGE_NAMESPACE` (line 29). No other changes; lazy build-on-demand works as-is.

- `patmir/pipeline/runner.py`
  - New branch in `_run_stage` (after `"separate"`, before `"asr"`): `if name == "diarize": return stage_obj.diarize(state["separate"], ctx)`.
  - In the `"asr"` branch: before calling `stage_obj.transcribe`, set `ctx.extra["diarization"] = state.get("diarize")`. After `transcribe` returns, call `apply_speaker_tags(transcript, diar)` and then `_backfill_ref_text(diar, transcript)` which mutates `diar.refs[*].ref_text` and rewrites `diarize/refs.json`.
  - Add a `"diarize"` branch to `_load_resume_artifact` that reads `diarize/turns.json` + `diarize/refs.json`.

- `patmir/pipeline/stages/tts/adapter.py` — for each `TranslatedSentence`, look up `ctx.extra["diarization"].refs[s.speaker_id]`. Call `self.inner.synthesize(s.target_text, ref_audio=ref.ref_audio, ref_text=ref.ref_text)`. If `speaker_id` is `None` or missing from `refs` (rare — e.g. a sentence outside all diarized turns), fall back to the inner candidate's constructor-time `ref_audio`/`ref_text` (the existing single-voice config). Adapter writes per-sentence WAVs as today.

- `patmir/pipeline/stages/tts/__init__.py` (and `eval/candidates/tts/omnivoice_mlx.py:48`) — extend `synthesize` signature to `synthesize(text: str, *, ref_audio: Path | None = None, ref_text: str | None = None) -> bytes`. When the optional args are `None`, use the constructor defaults (preserves Piper / global-voice usage with no behavior change). Pass straight through to `model.generate(...)` — OmniVoice already supports per-call refs.

- `patmir/pipeline/stages/translate/adapter.py` (and any translate candidate that constructs `TranslatedSentence`) — pass `speaker_id` through from input `Sentence` to output `TranslatedSentence`. Mechanical change.

- `patmir.toml`
  - `[pipeline]` — add `diarize = "pyannote_3_1"`.
  - New section:
    ```toml
    [[stages.diarize.candidates]]
    id = "pyannote_3_1"
    module = "patmir.pipeline.stages.diarize.pyannote"
    model_id = "pyannote/speaker-diarization-3.1"
    device = "auto"
    min_speakers = 0      # 0 = let pyannote estimate
    max_speakers = 8
    min_ref_seconds = 3.0
    hf_token_env = "HF_TOKEN"
    ```
  - Keep the global `ref_audio` / `ref_text` on the TTS candidate as the fallback.

- `patmir/pipeline/manifest.py` — surface diarization in `runs show`: speaker count, per-speaker ref-clip duration. The stage's artifacts dict (`turns.json`, `refs.json`, ref WAVs) is already captured by `_record_artifacts` if we extend `Diarization` with a `path` attribute that points to the diarize subdir, or write them explicitly in the diarize adapter.

- `pyproject.toml` — add `pyannote.audio>=3.1`. Pulls torchaudio + pytorch-lightning (most deps already present via torch).

- `README.md` (Phase-1 setup section) — append HF token note:
  - Create an HF token at `huggingface.co/settings/tokens`.
  - Visit `huggingface.co/pyannote/segmentation-3.0` and `huggingface.co/pyannote/speaker-diarization-3.1`, click "Agree".
  - `export HF_TOKEN=hf_...` before first run.

## Reusable existing utilities

- `SeparatedAudio.vocals`, `.speech_vocals`, `.vad_intervals` (`stage.py:46–51`) — diarize-on-full / slice-from-clean / post-filter ghost turns.
- The `ctx.extra[...]` plumbing pattern (`runner.py:157`) — reused verbatim to hand diarization to ASR.
- Cache-clear / drop-stage-reference pattern (`runner.py:97–128`) — applies identically to pyannote; pyannote's model frees on the same `gc.collect()` + `torch.mps.empty_cache()` path.
- Device resolution helper `_resolve_device` (`htdemucs.py:25`) — copy/adapt for pyannote candidate.
- `_load_resume_artifact` JSON-roundtrip pattern (`runner.py:191`) — mirror for `Diarization`.

## Verification

End-to-end on the existing multi-speaker sample (`samples/<masha-clip>.mp4`, referenced in `agent-os/specs/2026-05-25-1745-test-pipeline-masha-clip/`):

1. `uv run patmir dub samples/<masha-clip>.mp4` runs all eight stages without error.
2. `runs/<latest>/diarize/turns.json` — ≥2 distinct speaker labels (Masha + Bear minimum). Total turn duration roughly matches `vad_intervals` total duration.
3. `runs/<latest>/diarize/refs/` — one `<spk>.wav` per speaker, each ≥3 s, audibly clean (verify by ear: no music, no second voice, no scream/laugh leakage).
4. `runs/<latest>/diarize/refs.json` — each entry has a non-empty `ref_text` (Armenian text from ASR-then-translate? — no, source-language text from ASR, since the clip is from the source) and points to the corresponding WAV.
5. `runs/<latest>/transcript.json` — every sentence has a non-null `speaker_id`. Split-sentence pairs (if any) carry sequential indices and shared time-range coverage.
6. Listen to `runs/<latest>/output/*.mkv` — each character has a distinguishable Armenian voice; no character bleed into another's voice.
7. `patmir runs show <id>` — summary lists speaker count and per-speaker ref-clip durations.

Mid-build smoke tests (kotlin.test-style unit tests in `tests/`):

- `test_speaker_tag.py`: synthetic `Transcript` + `Diarization` →
  - dominant-speaker tagging when overlap ≥85%
  - sentence split when both halves have ≥1 word
  - fallback to dominant when split would orphan one half
- `test_ref_picker.py`: synthetic turn list + overlap intervals → picks longest non-overlapping turn ≥`min_ref_seconds` per speaker; skips speakers with no qualifying turn.

## Risks

- **pyannote MPS coverage is partial.** Some ops fall back to CPU. Expected: ~1–3 min on a 20-min cartoon. The `device = "auto"` config knob lets users force `"cpu"` if they hit MPS bugs.
- **Speakers with no clean turn** (rare — e.g. a character who only ever interjects during overlap). Ref picker returns no entry; TTS adapter falls back to the global `ref_audio`/`ref_text` for those sentences. Log a warning in the manifest.
- **HF token gate.** Hard-fail with an actionable error message if `HF_TOKEN` is unset *and* the pyannote model isn't already in the HF cache. Document in README.
- **OmniVoice ref-encoding cost per sentence.** Untested today since the global ref was set once. If profiling shows it's a real cost (e.g. >50 ms/sentence), batch by `speaker_id` to amortize — but only if measurement shows it matters.
- **`agent-os/standards/index.yml` is empty** — no project standards to apply yet; nothing to enforce.

---

## Tasks

### Task 1 — Save spec documentation

Create `agent-os/specs/2026-05-26-<HHMM>-multi-voice-diarization-cloning/` with:

- `plan.md` — copy of this file
- `shape.md` — scope, the five locked decisions, "no visuals", and the audio-input call-out (diarize on `vocals`, slice refs from `speech_vocals`)
- `standards.md` — note that `agent-os/standards/index.yml` is empty; nothing applies
- `references.md` — cite `omnivoice_mlx.py:48–75` (proves EA cloning works per-call), `htdemucs.py:163–245` (the VAD masking pattern we lean on), `runner.py:131–177` (the `ctx.extra` plumbing pattern we copy)

### Task 2 — Data contracts + stage protocol

Edit `patmir/pipeline/stage.py`: add `SpeakerTurn`, `OverlapInterval`, `SpeakerRef`, `Diarization`, `DiarizeStage` Protocol; add `speaker_id: str | None = None` to `Sentence` and `TranslatedSentence`; insert `"diarize"` into `STAGE_ORDER` between `"separate"` and `"asr"`. Edit `config.py:29` to add `"diarize": ("stages", "diarize")`.

### Task 3 — pyannote candidate

Implement `stages/diarize/pyannote.py`. Load `pyannote/speaker-diarization-3.1` with `os.environ[hf_token_env]`. Run on `SeparatedAudio.vocals`. Post-filter turns against `vad_intervals` (drop any turn segment fully outside any VAD window). Return `Diarization` with empty `refs` (filled by adapter + post-ASR backfill). Persist `diarize/turns.json`.

### Task 4 — Reference clip picker (window pass)

Implement `stages/diarize/ref_picker.py:pick_windows(turns, overlaps, speech_vocals_path, min_ref_seconds) -> dict[spk, (window_path, start, end)]`. Slice from `speech_vocals`, resample to 16 kHz mono, trim 100 ms inward. Write `diarize/refs/<spk>.wav`. Stage adapter assembles `Diarization.refs` with empty `ref_text` for now.

### Task 5 — ASR speaker tagging + sentence split

Implement `stages/asr/speaker_tag.py:apply_speaker_tags(transcript, diarization)`. Runner calls it inside the `"asr"` branch right after `stage_obj.transcribe(...)` returns. Then runner calls `backfill_ref_text(diarization, transcript)` to fill `SpeakerRef.ref_text` from ASR words inside each ref window; rewrites `diarize/refs.json`.

Unit tests: dominant tag, clean split, orphan-half fallback.

### Task 6 — Translate passthrough

Update `TranslatedSentence` construction in `stages/translate/adapter.py` and any translate candidate to pass `speaker_id` through. Grep `TranslatedSentence(` to confirm coverage.

### Task 7 — OmniVoice per-sentence conditioning

Extend `omnivoice_mlx.py:synthesize` to `(text, *, ref_audio=None, ref_text=None)` — pass through to `model.generate(...)` overriding the stored defaults when args are set. Update `Piper` candidate to accept and ignore the new kwargs (so the adapter signature is uniform). Update `tts/adapter.py` to look up the per-speaker `SpeakerRef` from `ctx.extra["diarization"].refs` and pass it; fall back to inner-default when missing.

### Task 8 — Config + CLI + manifest

Add `[stages.diarize]` and `[[stages.diarize.candidates]]` to `patmir.toml`. Extend `Manifest` to surface speaker count + ref-clip durations in `patmir runs show`. Add a `"diarize"` branch to `_load_resume_artifact` in `runner.py`.

### Task 9 — Deps + docs

Add `pyannote.audio>=3.1` to `pyproject.toml`; `uv sync`. Append HF-token setup to `README.md`.

### Task 10 — End-to-end verification

Run on the masha sample; complete the 7 verification steps above; spot-check a second multi-speaker clip if available.
