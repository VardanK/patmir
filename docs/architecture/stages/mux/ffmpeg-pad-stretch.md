# ffmpeg-pad-stretch (mux)

- **Repo / version:** in-tree — `patmir/pipeline/stages/mux/ffmpeg_pad_stretch.py`
- **Quantization available:** n/a (no model)
- **Apple Silicon / CUDA / CPU:** all (subprocess to ffmpeg + Python pyrubberband)
- **Status in pipeline:** **active**
- **Last evaluated:** 2026-05-26 (Phase 2)
- **Our experience:** Reliable. Sentence-level alignment + stretch cap is sufficient for the "lip sync is not a goal" target.
- **Notes:**
  - The `pyrubberband.time_stretch` call requires `rubberband-cli` to be installed system-wide.
  - **Phase 2.x voice-leak fix (in flight):** add `afade=t=in:st=<start>:d=0.05` + `afade=t=out:st=<end-0.05>:d=0.05` per synthesized line.
  - If the simple afade doesn't resolve the leak, the fix is bumped to its own spec — likely involves ducking the accompaniment under the synthesized line by 6-12 dB during the line and crossfading the duck at boundaries.

## See `quality-issues.md` issue #4 for current status.
