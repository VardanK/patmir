# ffmpeg-mux MKV-default (output)

- **Repo / version:** in-tree — `patmir/pipeline/stages/output/mkv.py`
- **Status in pipeline:** **active**
- **Last evaluated:** Phase 1
- **Notes:**
  - Designed so an `MP4Output` becomes a sibling module later (same protocol, different ffmpeg flags). Not yet built.
  - The Armenian SRT is generated from `TranslatedTranscript.sentences` (one cue per `TranslatedSentence`, using `target_text` and `start`/`end`).
  - Drift between SRT and dub audio is a bug, not a divergence — both share the same `translated.json`.
