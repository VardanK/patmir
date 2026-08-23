# Stage: asr

Transcribes the source-language vocal stem into word-level timestamps grouped into sentences. The runner immediately follows ASR with `apply_speaker_tags`, `apply_emotion_tags` (Phase 2.x), and `_backfill_ref_text` — all of which mutate the returned `Transcript`.

## I/O

- **Input:** `AudioTrack` (the ingest WAV) — but the candidates actually consume `vocals.wav` from the separate stage (the runner sets `ctx.extra` to make this available).
- **Output:** [`Transcript`](../../data-contracts.md#transcript) — word-level timing + sentence grouping. Per-sentence `speaker_id` and `emotion` are attached by the runner after the candidate returns.

## Protocol

```python
class ASRStage(Protocol):
    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript: ...
```

## Candidates

- **[qwen3-asr-mlx](qwen3-asr-mlx.md)** — **active on the default (Apple Silicon) profile** since 2026-07-14. Much faster than Whisper (2026-08-23 re-baseline: 10.8% of batch wall vs 29%); a 2026-07-14 eval note called the transcripts nonsense to a native speaker, but the author's flip verdict was "both garble sometimes, pick the fastest" and the ear-test in `runs/20260715-reacher-asr-ab/` is still pending.
- **[qwen3-asr-transformers](qwen3-asr-transformers.md)** — **active on the `cuda` profile** since 2026-08-23. Same model family on torch via transformers' native `qwen3_asr` (5.13+); deliberately does NOT use the `qwen-asr` package, whose exact `transformers==4.57.6` pin is mutually exclusive with the Higgs TTS checkpoint. Never executed.
- **[whisper-mlx-large-v3](whisper-mlx-large-v3.md)** — registered fallback on both profiles (was active 2026-06-19 → 07-14; the only local engine clean on fast cartoon RU).
- **[nemotron-mlx](nemotron-mlx.md)** — registered fast alternate (garbles fast cartoon dialogue; fine for slower/clearer speech).
- **[voxtral-mini-realtime](voxtral-mini-realtime.md)** — evaluated 2026-07-14, not adopted (quality ≈ Whisper but RTF 0.64 — slower than Whisper in our batch-over-slices architecture).
- **[faster-whisper](faster-whisper.md)** — registered fallback (CTranslate2). `faster_whisper_large_v3_fp16` was the `cuda` profile's ASR for part of 2026-08-23, chosen because CTranslate2 has no transformers dependency at all; superseded the same day and kept as the no-Qwen escape hatch.
- `elevenlabs_scribe_v2` — cloud fallback (needs ElevenLabs credits).

## Side-cars (live next to the candidates)

- `patmir/pipeline/stages/asr/_sentence.py` — `group_into_sentences(words, max_gap_s=0.6)`. Splits on `.!?` or word-gap. Used by every ASR candidate.
- `patmir/pipeline/stages/asr/speaker_tag.py` — `apply_speaker_tags(transcript, diarization)`. Tags each sentence with `speaker_id`, splits sentences spanning two speakers when both halves carry ≥1 word.
- `patmir/pipeline/stages/asr/emotion_tag.py` (Phase 2.x) — `apply_emotion_tags(transcript, predictions)`. Mirrors `speaker_tag` for emotion bins.
- `patmir/pipeline/stages/asr/gap_filler.py` — re-ASRs turns ≥1.5 s containing only non-lexical words. Added in Phase 2 to recover dialogue in laughter-dominated regions.

## See also

- [current.md](current.md)
