# Stage: tts

Synthesizes Eastern Armenian speech per sentence. The single most quality-load-bearing stage in the pipeline; voice fidelity, emotion transfer, and "watchable for hours?" all depend on this.

## I/O

- **Input:** [`TranslatedTranscript`](../../data-contracts.md#translatedtranscript) + `ctx.extra["diarization"]` (for per-`(speaker, emotion)` reference clip selection)
- **Output:** [`SynthesizedAudio`](../../data-contracts.md#synthesizedaudio) — one WAV per sentence

## Protocol

```python
class TTSStage(Protocol):
    def synthesize(self, translated: TranslatedTranscript, ctx: JobContext) -> SynthesizedAudio: ...
```

## Candidates

### Active

- **[omnivoice-mlx](omnivoice-mlx.md)** — active. Zero-shot voice cloning via ref clip. EA support verified.

### Fallback

- **[piper-hye](piper-hye.md)** — used when no ref clip is available for a speaker.

### Rejected (kept on file so we don't re-investigate)

- **[f5-tts-mlx](f5-tts-mlx.md)** — MLX-native, but **no Eastern Armenian support** (EN+ZH base; no published EA fine-tune). Rejected 2026-05-26.
- **[indextts-2](indextts-2.md)** — explicit 8-dim emotion vector, but **no Eastern Armenian support** + no MLX port. Rejected 2026-05-26.
- **[xtts-v2](xtts-v2.md)** — 17 supported languages; Armenian not among them. Rejected (corrected from earlier tech-stack notes).
- **[bark](bark.md)** — Armenian dialect handling unverified; deferred indefinitely.
- **[mms-tts-hye](mms-tts-hye.md)** — does not exist on HF (only the Western `mms-tts-hyw`). Documented dead end.

### Cloud (opt-in only)

- **[elevenlabs-multilingual-v2](elevenlabs-multilingual-v2.md)** — disabled; primary cloud option if local fails.
- **[google-tts-cloud](google-tts-cloud.md)** — disabled; roughly Piper-tier, not a meaningful upgrade.

## Per-sentence cloning

The adapter (`stages/tts/adapter.py`) calls `inner.synthesize(text, ref_audio=..., ref_text=..., guidance_scale=...)` per sentence. Reference lookup chain (Phase 2.x):

1. `refs[(speaker_id, emotion)]` — exact match
2. `refs[(speaker_id, "neutral")]` — same speaker, fallback emotion
3. Global default ref configured on the candidate

If no ref clip is available at all, the adapter falls back to the **Piper** candidate so the sentence still gets synthesized.

## See also

- [current.md](current.md) — active config
- ADR: [OmniVoice locked for EA](../../decisions/2026-05-26-omnivoice-locked-for-ea.md)
- ADR: [OmniVoice supersedes Piper](../../decisions/2026-05-25-omnivoice-supersedes-piper.md)
- ADR: [Emotion routed through ref-clip binning](../../decisions/2026-05-26-emotion-routed-through-ref-clip-binning.md)
