# Active mux config

```toml
[pipeline]
mux = "ffmpeg_pad_stretch"

[pipeline.options]
stretch_cap       = 0.20    # max ±20% time-stretch ratio per sentence
music_bed_gain    = 0.6     # accompaniment level under the AR voice (1.0 = unchanged)
voice_gain_db     = 0.0     # extra dB applied to the AR voice
```

- Per-sentence stretch via `pyrubberband` (preserves pitch). Skipped when the required ratio is within `[0.98, 1.02]`.
- Stretched audio that's still shorter than the target window is padded with silence; still longer → overruns (lip sync is not a goal).
- Accompaniment (Demucs music + sound-effects bed) mixed under the AR voice at `music_bed_gain`. `non_speech_vocals` is added back into the bed so atmospheric vocalizations survive.
- Phase 2.x: 50 ms `afade` in/out per synthesized line.

## Requires

- `ffmpeg` on `PATH`
- `rubberband-cli` (`brew install rubberband`)
- Python: `pyrubberband>=0.3`
