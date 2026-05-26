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

- **[whisper-mlx-large-v3](whisper-mlx-large-v3.md)** — active.
- **[faster-whisper](faster-whisper.md)** — registered as a deferred fallback (CTranslate2 int8).
- **[qwen3-asr-mlx](qwen3-asr-mlx.md)** — experimental; alt-architecture ASR.

## Side-cars (live next to the candidates)

- `patmir/pipeline/stages/asr/_sentence.py` — `group_into_sentences(words, max_gap_s=0.6)`. Splits on `.!?` or word-gap. Used by every ASR candidate.
- `patmir/pipeline/stages/asr/speaker_tag.py` — `apply_speaker_tags(transcript, diarization)`. Tags each sentence with `speaker_id`, splits sentences spanning two speakers when both halves carry ≥1 word.
- `patmir/pipeline/stages/asr/emotion_tag.py` (Phase 2.x) — `apply_emotion_tags(transcript, predictions)`. Mirrors `speaker_tag` for emotion bins.
- `patmir/pipeline/stages/asr/gap_filler.py` — re-ASRs turns ≥1.5 s containing only non-lexical words. Added in Phase 2 to recover dialogue in laughter-dominated regions.

## See also

- [current.md](current.md)
