# Standards for CUDA pipeline performance re-baseline and optimization

`agent-os/standards/index.yml` contains no entries, so no project standards apply to this work.

The binding conventions are instead:

- **`CLAUDE.md` — Maintenance contract.** Every behaviour change updates the corresponding doc in the same change: `current-models.md` for candidate/knob changes, a new `decisions/<date>-<slug>.md` + one line in `decisions.md` for non-trivial decisions, `quality-issues.md` for any change awaiting an ear verdict, `data-contracts.md` if a stage dataclass changes, `runs.md` for new per-job artifacts.
- **`docs/architecture/performance.md` conventions:** RTF = wall / source seconds (> 1 is slower than realtime); aggregate = Σ stage walls / source; the **warm** run is the baseline; re-measure, do not re-derive.
- **Quality classes defined in `plan.md`** (A bitwise / B numeric drift + ear-test / C behavioural with acceptance test) govern what may ship and how it is verified.
- **Reproducibility invariant (ADR 2026-08-24-tts-sampling-reproducibility):** same sentence set → byte-identical TTS audio. Every TTS change must preserve it and is tested against it.
