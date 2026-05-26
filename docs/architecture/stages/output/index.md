# Stage: output

Muxes the synthesized Armenian audio back into the source video container, retaining the original audio as a secondary track and emitting an Armenian SRT.

## I/O

- **Input:** `MuxedAudio` + `source_video: Path` + `TranslatedTranscript` (for SRT generation)
- **Output:** [`OutputArtifacts`](../../data-contracts.md#outputartifacts) — `out.mkv` + `out.hy.srt`

## Protocol

```python
class OutputStage(Protocol):
    def write(
        self,
        muxed: MuxedAudio,
        source_video: Path,
        translated: TranslatedTranscript,
        ctx: JobContext,
    ) -> OutputArtifacts: ...
```

## Candidates

- **[ffmpeg-mkv-default](ffmpeg-mux.md)** — active. MKV container with Armenian default audio, original retained, Armenian SRT default subs.

## Future

- MP4 output as a sibling candidate (same protocol, different ffmpeg flags) — not yet implemented.
