# References

## Prior specs (mined for historical context)

- `agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/` — verdict, winning candidates per stage, quality floors, RU-narration / idiom failure modes, GPU re-eval queue, cloud fallback ordering.
- `agent-os/specs/2026-05-24-1704-phase-1-mvp-pipeline/` — Phase 1 model choices, post-Phase-0 candidate scan (XTTS-v2 correction, Bark deferral), dogfood plan.
- `agent-os/specs/2026-05-25-1745-test-pipeline-masha-clip/` — masha clip introduced as the canonical multi-speaker test sample.
- `agent-os/specs/2026-05-26-0215-multi-voice-diarization-cloning/` — Phase 2 plan (diarization, per-speaker cloning, ASR gap-filler, hallucination blocklist). Phase 2.x extends this.

## Product docs

- `agent-os/product/mission.md` — Eastern-Armenian-only scope, local-first principle, hard non-goals.
- `agent-os/product/roadmap.md` — phase ordering, Phase 2 quality-of-life list (preset pool, voice cloning stretch).
- `agent-os/product/tech-stack.md` — per-stage candidate landscape, hardware-sizing requirement, dialect-ambiguity rule.

## Code (cited by line)

- `patmir/pipeline/stage.py:18-211` — every typed dataclass + Protocol the pipeline talks through. Phase 2.x adds `Emotion`, `EmotionPrediction`, `EmotionStage` here.
- `patmir/pipeline/runner.py:131-213` — the `_run_stage` dispatch + `ctx.extra` plumbing pattern. Phase 2.x adds a new `"emotion"` branch + moves the `ref_picker.pick_refs` call out of diarize into a runner-level post-emotion step.
- `patmir/pipeline/config.py:29-38` — `_STAGE_NAMESPACE` registry for stage candidates. Phase 2.x adds `"emotion"`.
- `patmir/pipeline/stages/diarize/pyannote.py:117-194` — diarize stage today. Phase 2.x strips the `pick_refs` call here.
- `patmir/pipeline/stages/diarize/ref_picker.py` — current per-speaker `pick_refs`. Phase 2.x reworks to per-`(speaker, emotion)`.
- `patmir/pipeline/stages/asr/speaker_tag.py` — algorithm shape for sentence-speaker tagging. Phase 2.x's `emotion_tag.py` mirrors this.
- `patmir/pipeline/stages/asr/_sentence.py:154-181` — sentence grouping from word stream with start/end timing. Used to derive sentence-windowed emotion lookups.
- `patmir/pipeline/stages/asr/gap_filler.py` — Phase 2 re-ASR pass over non-lexical-dominated turns. Documented; not modified by this spec.
- `patmir/pipeline/stages/separate/htdemucs.py:102-242` — vocal-stem writing, VAD masking, cosine-fade boundaries. The cleaner `speech_vocals.wav` is what ref_picker slices from.
- `patmir/eval/candidates/tts/omnivoice_mlx.py:48-74` — the TTS API surface we route emotion through. Phase 2.x extends `synthesize` to accept `guidance_scale: float | None = None`.
- `patmir/eval/candidates/tts/piper_hye.py:29` — fallback TTS; signature broadens to accept (and ignore) the new emotion kwargs.
- `patmir/eval/candidates/translation/qwen3_instruct_mlx.py:21-72` — the LLM-style translator. Phase 2.x injects an emotion tag into the user-message template when active.
- `patmir/eval/candidates/translation/nllb_200_distilled_600m.py` — shared loader used by both `nllb_200_distilled_600m` and `nllb_200_3.3b` candidates.
- `patmir/pipeline/stages/tts/adapter.py:43-91` — adapter loop where per-sentence ref lookup happens. Phase 2.x changes the lookup key.
- `patmir/pipeline/stages/mux/ffmpeg_pad_stretch.py` — where the voice-leak `afade` lands (issue #4).
- `patmir.toml` — `[pipeline]` selectors + `[[eval.<stage>.candidates]]` registries + (Phase 2.x) `[[stages.emotion.candidates]]` + `[emotion_bins]` + `[emotion_guidance]`.

## External

- **F5-TTS** — `github.com/SWivid/F5-TTS`, `huggingface.co/SWivid/F5-TTS`; MLX port `github.com/lucasnewman/f5-tts-mlx`. Investigated 2026-05-26. Rejected on EA gap.
- **IndexTTS-2** — `huggingface.co/IndexTeam/IndexTTS-2`, paper `arXiv:2506.21619`. Investigated 2026-05-26. Rejected on EA gap.
- **emotion2vec** — `huggingface.co/iic/emotion2vec_plus_base`, paper `arXiv:2312.15185`. Active SER candidate.
- **EMO-SUPERB** — `arXiv:2402.13018`, `emosuperb.github.io`. Background reference for SER model selection.
