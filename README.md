# Patmir (պատմիր — "tell the story")

Translates foreign-language video into an Eastern-Armenian (`hy_AM`) dub, locally.
Built for the author's hearing-aid-wearing son. A workstation tool, not a SaaS —
a polished UI is explicitly not a goal.

Nine swappable stages: `ingest → separate → diarize → emotion → asr → translate →
tts → mux → output`. `patmir.toml [pipeline]` is the single source of truth for
which model runs each stage.

## Quick start

```bash
uv pip install -e ".[local,ui,dev]"          # Apple Silicon
uv pip install -e ".[local,ui,dev,cuda]"     # Linux + NVIDIA

patmir dub samples/sample_1.mkv --source-lang ru     # batch (quality) mode
patmir ui --port 8000                                # local server + watch-along
patmir bench samples/sample_1.mkv --mode batch --repeat 3
```

Outputs land in `runs/<job_id>/`: `out.mkv` (Armenian default audio, original
retained, Armenian SRT), `dub.wav`, `out.hy.srt`.

## Machine profiles

One committed `patmir.toml` serves machines with different accelerators. The
default `[pipeline]` table is the Apple-Silicon laptop (MLX); a profile overlays
the stages that must change.

```bash
export PATMIR_PROFILE=cuda        # or: patmir bench <clip> --profile cuda
```

An unknown profile name is a hard error, never a silent fall back to the
defaults. Measured baselines on `samples/sample_1.mkv` (301 s):

| | batch (warm) | watch-along |
| --- | --- | --- |
| RTX 3090 / WSL2 | **0.461** RTF | **1.42** agg, **0.831** steady-state |
| M1 Pro 32 GB | 1.17 RTF | 1.611 |

## Troubleshooting

### CUDA "out of memory" on WSL2 while gigabytes of VRAM are free

**The pagefile, not the GPU, sets the ceiling.** This one cost a full afternoon,
so read it before believing any OOM on a WSL2 box.

A CUDA allocation under WSL2 consumes Windows **commit charge** at ~1.06 GB per
GB of VRAM: the Windows display driver model requires a system-memory backing
store for every GPU allocation and commits it up front, even though the data
lives on the card and the backing is never touched. Commit is a *promise*, and:

```
commit limit = physical RAM + pagefile size
```

So a small pagefile caps GPU allocation regardless of how much VRAM the card has.
On the dev box: 31.84 GB RAM + a manually-pinned **4 GB** pagefile = 35.84 GB
limit, minus a ~19.7 GB idle baseline = **~16 GB usable on a 24 GB card** — just
under what the pipeline wanted, so it failed intermittently.

**Recognise it by the shape of the error, which is thoroughly misleading:**

```
CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of
24.00 GiB of which 14.46 GiB is free. Including non-PyTorch memory, this process
has 17179869184.00 GiB memory in use.
```

- **A 20 MiB request failed with 14.46 GiB free** — never a real capacity problem.
- **`17179869184.00 GiB` is 2^34 GiB**, i.e. garbage. PyTorch infers "non-PyTorch
  memory" by subtraction and the arithmetic goes negative, because the refusal
  did not come from VRAM accounting at all.
- The traceback lands on whatever tiny allocation came next, **blaming innocent
  code**.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which the message suggests,
  **does not help** — there is no fragmented pool to compact.

**Fix:** enlarge the Windows pagefile. It raises the limit without causing paging
— nothing is written while physical RAM is free (measured: a 4 GB pagefile held
0.24 GB with 21.85 GB of RAM free). Set `initial` equal to `maximum` so the
headroom exists immediately rather than growing on demand, and host it on a data
drive if the system drive is tight.

| goal | VRAM | commit limit | pagefile |
| --- | --: | --: | --: |
| warm model cache + Higgs TTS | ~15 GB | ~40 GB | ~10 GB |
| the whole 24 GB card | 24 GB | ~48-50 GB | ~16-20 GB |

**Two corollaries:**

- **Keep the Windows desktop quiet during a benchmark.** Chrome playing video
  holds commit and directly shrinks CUDA's ceiling — two runs failed with a video
  playing and the identical run succeeded with Chrome closed. This is the WSL2
  analogue of the `caffeinate` warning in `docs/architecture/performance.md`.
- **A reboot buys headroom for free** by dropping the commit baseline (~19.7 GB →
  ~11 GB on this box). Legitimate first move when a run fails near the limit.

**What it is *not*** — both were checked, so nobody needs to check again:

- *Not faulty VRAM.* 19 GiB tested with four patterns (zeros, all-ones, 0xAA55
  coupling, address-in-address) = 76 GiB written and verified, **0 errors**. Note
  a consumer card has no ECC, so `nvidia-smi -q` reports ECC counters as `N/A`
  and cannot clear the memory by itself — a pattern test is the instrument.
- *Not the Windows "shared GPU memory" pool.* Task Manager may show ~40 GB of GPU
  memory (dedicated + shared); CUDA under WSL2 does not use the shared half —
  per-process shared usage stayed at 0.21 GB flat through a model load. Its
  presence costs nothing and shrinking it would free nothing.
- *Not `.wslconfig memory=`.* That is a cap, not a reservation — `vmmemWSL` held
  3.98 GB against a 24 GB cap with 21.85 GB of RAM free.

Full write-up, including the per-GB measurement table:
`docs/setup/windows-wsl2-cuda.md`.

### Other known environment issues

- **`torchaudio.load` raises on `libtorchcodec_*.so`** (ffmpeg 8.x vs torchcodec
  0.16). Nothing on the CUDA path hits it — audio is decoded through
  `patmir/eval/candidates/tts/_audio_io.py` (ffmpeg) or soundfile — but new code
  reaching for `torchaudio.load` will. Use those helpers instead.
- **`transformers>=5.13` is required**, not optional: below it the ASR stage dies
  with `KeyError: 'qwen3_asr'`. Never install the `qwen-asr` package — it pins
  `transformers==4.57.6` exactly and breaks the TTS stage.

## Documentation

`docs/architecture/` is a two-tier knowledge base. Start with:

| file | what it answers |
| --- | --- |
| `pipeline.md` | how the nine stages fit together, both execution modes |
| `current-models.md` | which model runs each stage, and why |
| `decisions.md` | index of every ADR — **read before re-investigating anything** |
| `quality-issues.md` | live tracker of known output problems |
| `performance.md` | measured baselines per machine; re-measure, don't re-derive |
| `eval-clips.md` | which test clips exist and what they are |
| `runs.md` | what a job directory contains, and how to debug from it |
| `data-contracts.md` | every dataclass field |

`CLAUDE.md` carries the maintenance contract: which docs must be updated
alongside which kind of change.
