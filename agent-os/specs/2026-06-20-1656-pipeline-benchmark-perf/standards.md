# Standards for Pipeline benchmark + perf

`agent-os/standards/index.yml` contains only a header — no standards are defined
in this repo at the time of this spec. None apply.

Conventions followed instead (from the existing codebase):

- CLI subcommands use `argparse` subparsers in `patmir/cli.py` (`cmd_<name>`).
- Timing uses `time.perf_counter()` for operation latency, `time.time()` for
  manifest wall-clock (matches `patmir/eval/*_eval.py` and `runner.py`).
- RTF convention: `wall_seconds / source_seconds` (>1.0 = slower than realtime),
  matching `spike/d3_throughput/tts_sweep.py`.
- Per-run artifacts are JSON / JSONL under the job dir (matches `manifest.json`,
  `progress.jsonl`, `synthesized.json`).
