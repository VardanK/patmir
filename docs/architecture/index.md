# docs/architecture — entry point

Human-facing navigation. The agent loads the always-loaded tier automatically via `CLAUDE.md`; humans land here when they want to read the docs.

## Always-loaded tier (the agent's defaults)

- **[pipeline.md](pipeline.md)** — stage order, ASCII diagram, `ctx.extra` plumbing
- **[current-models.md](current-models.md)** — table of active models + key knobs
- **[decisions.md](decisions.md)** — index of every locked decision
- **[quality-issues.md](quality-issues.md)** — live tracker of observed problems

## Setup

- **[../setup/windows-wsl2-cuda.md](../setup/windows-wsl2-cuda.md)** — running the pipeline on the NVIDIA workstation via WSL2 + CUDA

## On-demand tier (read when working in that area)

- **[data-contracts.md](data-contracts.md)** — every dataclass from `stage.py`, fields + meanings
- **[language-support.md](language-support.md)** — EA constraint; which models produce `hy_AM` vs `hyw` vs nothing
- **[runs.md](runs.md)** — what every file in `runs/<job>/` is; debugging guide
- **[glossary.md](glossary.md)** — terms (SER, VAD, DiT, ref clip, flow matching, etc.)
- **[performance.md](performance.md)** — per-stage RTF baselines (batch + watch-along) + perf roadmap; how `patmir bench` measures

### Decisions (ADRs)

The full text of every locked decision. The one-line index is in [`decisions.md`](decisions.md). Files are date-prefixed for easy chronological reading:

```
decisions/
├── 2026-05-24-eastern-armenian-only.md
├── 2026-05-24-local-first-strict.md
├── 2026-05-24-piper-hye-as-tts-floor.md
├── 2026-05-25-nllb-3-3b-over-600m.md
├── 2026-05-25-omnivoice-supersedes-piper.md
├── 2026-05-26-pyannote-3-1-for-diarization.md
├── 2026-05-26-omnivoice-locked-for-ea.md
├── 2026-05-26-emotion-routed-through-ref-clip-binning.md
└── 2026-05-26-auto-loaded-architecture-docs.md
```

### Stages

Per-stage details — what the stage does, its input/output contracts, every candidate considered, and the active one. The agent reads these when working in the relevant area:

```
stages/
├── ingest/      (ffmpeg)
├── separate/    (htdemucs)
├── diarize/     (pyannote 3.1)
├── emotion/     (emotion2vec)
├── asr/         (mlx-whisper large-v3 + alternates)
├── translate/   (nllb-3.3b + alternates + cloud)
├── tts/         (omnivoice-mlx-q8 + piper fallback + rejected candidates)
├── mux/         (ffmpeg pad-stretch)
└── output/      (mkv default)
```

Every `stages/<stage>/` contains:

- `index.md` — what the stage does + input/output contracts
- `current.md` — active candidate + key knob values
- One file per candidate considered (status: active | fallback | superseded | rejected | future-candidate)
