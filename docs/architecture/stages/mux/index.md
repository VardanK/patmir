# Stage: mux

Concatenates the per-sentence synthesized WAVs into a single Armenian track, with sentence-level time alignment, pitch-preserving stretch, accompaniment mixing, and (Phase 2.x) per-sentence afade in/out to suppress voice leak.

## I/O

- **Input:** `SynthesizedAudio` + `AudioTrack` (source) + `SeparatedAudio`
- **Output:** [`MuxedAudio`](../../data-contracts.md#muxedaudio) at `<job_dir>/dub.wav`

## Protocol

```python
class MuxStage(Protocol):
    def mux(
        self,
        synth: SynthesizedAudio,
        source: AudioTrack,
        separated: SeparatedAudio,
        ctx: JobContext,
    ) -> MuxedAudio: ...
```

## Candidates

- **[ffmpeg-pad-stretch](ffmpeg-pad-stretch.md)** — active.

## Phase 2.x change (voice-leak fix)

Per-sentence 50 ms `afade=in` + 50 ms `afade=out` on the synthesized track at sentence boundaries — suppresses the original-voice bleed observed in Phase 2 output. See `quality-issues.md` issue #4.
