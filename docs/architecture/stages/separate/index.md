# Stage: separate

Splits the audio into vocals + accompaniment + speech-VAD-masked vocals + non-speech vocals + VAD interval list. This is what allows the dub to overlay clean Armenian voice on the source's music/SFX bed, and what gives diarize/asr/ref_picker clean inputs.

## I/O

- **Input:** `source_video: Path`
- **Output:** [`SeparatedAudio`](../../data-contracts.md#separatedaudio)

## Protocol

```python
class SeparateStage(Protocol):
    def separate(self, source_video: Path, ctx: JobContext) -> SeparatedAudio: ...
```

## Candidates

- **[htdemucs](htdemucs.md)** — active. Demucs `htdemucs_ft` + Silero VAD masking.

## See also

- [current.md](current.md) for the active config.
- `runs.md` describes the on-disk artifacts (`vocals.wav`, `accompaniment.wav`, `speech_vocals.wav`, `non_speech_vocals.wav`).
