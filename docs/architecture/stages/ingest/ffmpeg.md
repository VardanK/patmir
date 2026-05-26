# ffmpeg (ingest)

- **Repo / version:** system `ffmpeg` (4.x or 5.x)
- **Parameters / size / runtime memory:** n/a (binary)
- **Quantization available:** n/a
- **Apple Silicon / CUDA / CPU:** all (subprocess only)
- **Languages:** n/a (passthrough)
- **License / gating:** LGPL / GPL depending on build
- **Integration surface:** `FFmpegIngest.extract(video_path, ctx) -> AudioTrack` (`patmir/pipeline/stages/ingest/ffmpeg.py`)
- **Status in pipeline:** **active**
- **Last evaluated:** 2026-05-24 (Phase 1)
- **Our experience:** Reliable. ffprobe-derived duration is used for downstream progress reporting.
- **Notes:** Source video remains untouched; the audio extraction is a non-destructive side artifact.
