# Stage: ingest

Extracts audio from the source video into a 16 kHz mono PCM WAV.

## I/O

- **Input:** `video_path: Path` (any container ffmpeg can read)
- **Output:** [`AudioTrack`](../../data-contracts.md#audiotrack) — 16 kHz mono PCM at `<job_dir>/audio.wav` + duration from ffprobe

## Protocol

```python
class IngestStage(Protocol):
    def extract(self, video_path: Path, ctx: JobContext) -> AudioTrack: ...
```

## Candidates

- **[ffmpeg](ffmpeg.md)** — active. Subprocess call to ffmpeg + ffprobe.

## See also

- [current.md](current.md) for the active config.
- Phase 2 roadmap calls for `yt-dlp` ingest (YouTube URLs). Not implemented yet; would live alongside `ffmpeg.py`.
