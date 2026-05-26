# Test current Patmir pipeline on a 5-min clip of samples/masha.mkv

## Context

Phase 1 of the pipeline landed in commits `0df3b89` (MVP) and `b15f49f` (separate + lazy stage loading). The `[pipeline]` block in `patmir.toml` currently wires together a fairly aggressive local stack:

- `ingest = ffmpeg`
- `separate = htdemucs`
- `asr = qwen3_asr_1_7b_mlx_8bit`
- `translate = qwen3_6_27b_unsloth_ud_mlx_4bit` (~26 GB on disk, ~15 GB in unified memory)
- `tts = omnivoice` (voice-clone TTS, dialect = macrolanguage `hy`, ear-judged)
- `mux = ffmpeg_pad_stretch`
- `output = mkv_default`

None of this has been exercised end-to-end on a real clip yet — `runs/` is empty. Goal: run the pipeline against a trimmed 5-minute clip of `samples/masha.mkv`, watch the resulting `out.mkv`, and confirm the Eastern-Armenian dub is intelligible. Success = pipeline completes and the author reviews the output.

A 5-minute clip is the right unit of test: long enough to stress every stage (htdemucs separation, multi-sentence ASR + translate, dozens of TTS chunks, time-stretch mux) but short enough to iterate within an hour. The CLI has no `--duration` flag (see `patmir/cli.py:34`), so trimming happens offline with ffmpeg before invoking `patmir dub`.

## Task 1 — Save spec documentation

Create `agent-os/specs/2026-05-25-1745-test-pipeline-masha-clip/` with:

- `plan.md` — copy of this plan.
- `shape.md` — scope, decisions (5-min clip, accept current config as-is including `omnivoice` even though its eval candidate is marked `enabled=false` — that flag is eval-only and not consulted by `patmir/pipeline/config.py`), success criteria.
- `references.md` — pointers to `patmir/pipeline/runner.py`, `patmir/pipeline/stages/**`, `patmir.toml [pipeline]` block, and the Phase 1 spec at `agent-os/specs/2026-05-24-1704-phase-1-mvp-pipeline/`.
- `standards.md` — note that `agent-os/standards/index.yml` is empty; nothing to apply.
- `visuals/` — empty (no mockups for a CLI test run).

## Task 2 — Pre-flight checks

Confirm before any heavy work:

- `samples/masha.mkv` exists (1 007 910 259 bytes, Matroska — already verified).
- `ffmpeg`, `ffprobe`, `rubberband` on `PATH` (already verified at `/opt/homebrew/bin/...`).
- `uv run python -c "import mlx_lm, mlx_qwen3_asr, omnivoice"` succeeds (already verified — the deps live in the uv venv, not system Python; all `patmir` invocations must go through `uv run`).
- OmniVoice reference audio: `eval_runs/20260524-160019/tts/piper_hye/hy-cartoon-01.wav` exists (verified). This is the cloned voice; if it's missing, OmniVoice will fail with no reference.
- Disk headroom: model auto-downloads on first use will pull ~26 GB (Qwen3.6-27B-UD), ~1.7 GB (Qwen3-ASR-1.7B-8bit), ~1.3 GB (Qwen3-ForcedAligner), ~600 MB (OmniVoice), ~300 MB (Demucs). Budget ~30 GB free under `~/.cache/huggingface`.

No code changes — read-only verification.

## Task 3 — Trim a 5-minute clip

Pick a representative window with steady speech (skipping over silent intros / music-only intervals will give the dub a fairer hearing). Default to **00:01:00 → 00:06:00** unless the author specifies otherwise.

```bash
ffmpeg -ss 00:01:00 -i samples/masha.mkv -t 300 -c copy samples/masha-5min.mkv
```

`-c copy` keeps it cheap; if the clip won't keyframe-align, fall back to a re-encode:

```bash
ffmpeg -ss 00:01:00 -i samples/masha.mkv -t 300 \
       -c:v libx264 -preset veryfast -crf 18 -c:a aac samples/masha-5min.mkv
```

Verify with `ffprobe -v error -show_entries format=duration samples/masha-5min.mkv` that duration ≈ 300 s.

## Task 4 — Run the pipeline

```bash
uv run patmir dub samples/masha-5min.mkv
```

The runner (`patmir/pipeline/runner.py:65`) creates `runs/<YYYYMMDD-HHMMSS>-masha-5min/` and writes `progress.jsonl` + `manifest.json` after every stage transition. The CLI tails `progress.jsonl` and renders one line per stage.

First run will download all model weights — expect the ASR + translate + TTS stages to each block briefly on download. The 27B translate model is the longest single download (~26 GB). Run this in a session that won't be interrupted.

If a single stage fails, resume in place without rerunning earlier stages:

```bash
uv run patmir dub samples/masha-5min.mkv \
       --resume-job runs/<the-job-dir> \
       --resume-from <failed-stage>
```

## Task 5 — Inspect artifacts

After the run, the job dir contains:

- `audio.wav` — 16 kHz mono ingest.
- `accompaniment.wav`, `vocals.wav`, `speech_vocals.wav`, `non_speech_vocals.wav` — htdemucs + Silero VAD outputs.
- `transcript.json` — ASR words + sentences with timestamps.
- `translated.json` — sentence-aligned Armenian translation.
- `tts/<idx>.wav` — per-sentence Armenian audio.
- `dub.wav` — mixed Armenian voice + original music/SFX.
- `out.mkv` — final container.
- `out.hy.srt` — sidecar Armenian subtitles.

Eyeball checks before playback:

```bash
jq '.sentences | length' runs/<job>/transcript.json     # sentence count
jq '.sentences[0:3]' runs/<job>/translated.json         # AR text sanity
ls runs/<job>/tts | wc -l                               # one wav per sentence
```

## Task 6 — Verification

Open `out.mkv` in a player (QuickTime / IINA / VLC) and judge by ear:

- Audio is Eastern Armenian, not Western (the OmniVoice ref clip is `hy_AM` Piper output, so this should hold).
- Dub timing roughly tracks the original speech turns (mux uses pad+stretch — no lip-sync expected).
- Music / SFX from `accompaniment.wav` survives in the mix.
- No long stretches of silence where speech existed in the original.

Failures worth flagging back to the author:

- Western-Armenian-sounding pronunciation → swap `tts = "piper_hye"` and re-run from the TTS stage.
- Garbled / hallucinated AR translation → fall back to `translate = "qwen3_8b_mlx"` (smaller, already exercised in eval).
- Pipeline crash on the 27B model → low memory; try `qwen3_6_27b_mlx_4bit` (plain MLX 4-bit, ~15 GB) instead of the Unsloth UD variant.

No code edits are part of this plan — it's purely a configured run plus review. Any model swap requires a follow-up plan.
