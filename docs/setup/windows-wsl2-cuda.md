# Running Patmir on Windows (WSL2 + CUDA)

Setup for the NVIDIA workstation. Verified 2026-08-22 against an RTX 3090 / i5-14400 / B760 /
32 GB DDR4 / 1000 W box.

## Why WSL2, not native Windows or Docker

Native Windows fails on dependencies the pipeline actually needs:

- `pyrubberband` (mux stage) shells out to the `rubberband` CLI binary — one `apt install` on
  Linux, a build hunt on Windows
- NeMo (if `diarize` uses Sortformer) is effectively Linux-only
- flash-attention and Triton are painful or unavailable
- `funasr` (emotion2vec) is Linux-first
- Python has no `fork` on Windows, only `spawn`

Docker Desktop on Windows runs on the WSL2 backend anyway, so it buys isolation and
reproducibility — not compatibility — while adding a rebuild/remount loop during active
development, another port hop between the Chrome extension and the FastAPI server, and volume
management for a ~20 GB HuggingFace cache. Revisit Docker once the CUDA port is stable.

WSL2 also keeps Windows available alongside for GPU monitoring (HWiNFO64, GPU-Z, MSI Afterburner).

## Version choices

| Choice | Value | Why |
| --- | --- | --- |
| Distro | Ubuntu 26.04 LTS | released 2026-04-23, supported to 2031 |
| CUDA repo | `wsl-ubuntu` | version-agnostic: one repo containing only `x86_64/`, so the Ubuntu version does not gate CUDA availability |
| CUDA toolkit | 12.8 | 13.3 exists, but third-party ML wheels (flash-attention, xformers, bitsandbytes) overwhelmingly target 12.x |
| PyTorch wheels | cu128 | indexes cu126–cu132 are published |
| Python | 3.12, pinned via `uv` | 26.04 defaults to 3.14, and on cp314 PyTorch installs a **CPU-only** build with no error (pytorch#169929) |

## Phase 0 — Windows checks

```powershell
nvidia-smi        # must report the GPU
wsl --version     # run `wsl --update` if missing or old
```

## Phase 1 — Install Ubuntu

```powershell
wsl --list --online
wsl --install -d Ubuntu-26.04
```

Prompts for a UNIX username and password, unrelated to the Windows account. Keep it short and
lowercase — it becomes `/home/<name>` and appears in every path. Avoid dots and reserved names
(`root`, `ubuntu`, `admin`).

## Phase 2 — Resource limits

`C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB
processors=12
swap=8GB
networkingMode=mirrored
```

Sized for a 32 GB / 16-thread host, leaving 8 GB and 4 threads for Windows. `networkingMode=mirrored`
gives bidirectional localhost, which the Chrome extension needs; omit it on Windows 10.

```powershell
wsl --shutdown
```

## Phase 3 — Confirm GPU passthrough

```bash
nvidia-smi
```

**No driver is installed inside WSL** — it passes through from the Windows driver. A failure here
means the Windows driver or the WSL kernel, not Ubuntu.

## Phase 4 — CUDA toolkit

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

> Install `cuda-toolkit-12-8` and nothing else. Never `cuda`, `cuda-drivers`, or `nvidia-driver-*`
> — those pull a Linux display driver that breaks WSL GPU passthrough. This is the most common way
> a working WSL CUDA setup gets broken.

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

## Phase 5 — System packages

```bash
sudo apt install -y build-essential cmake pkg-config git curl \
  ffmpeg rubberband-cli libsndfile1
```

> Do **not** add `libsndfile1-dev`. It is absent on Ubuntu 26.04 (Debian renamed it to
> `libsndfile-dev` and the transitional package appears dropped), and it is not needed —
> `soundfile` has shipped manylinux wheels with libsndfile bundled since 0.12. Only revisit if
> something later fails on a missing `sndfile.h`.

Verify the two binaries the pipeline shells out to:

```bash
ffmpeg -version | head -1
rubberband --version
```

## Phase 6 — Python, pinned

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv python install 3.12
```

## Phase 7 — Repo and PyTorch

```bash
cd ~
git clone <repo-url> patmir
cd patmir
uv venv --python 3.12
source .venv/bin/activate
python --version          # MUST be 3.12.x — stop here if it says 3.14
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Requires `True` and the GPU name before going further.

```bash
uv pip install -e ".[local,ui,dev]"
pytest patmir/tests/ -q
```

NeMo, for the `sortformer_nemo` diarize candidate, installs separately — it wants Cython present
first and is not a published wheel:

```bash
uv pip install Cython
uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
python -c "from nemo.collections.asr.models import SortformerEncLabelModel; print('nemo ok')"
```

This is the heaviest dependency in the stack and the most likely to fight the Python version — it is
why 3.12 is pinned. If it refuses to build, the pipeline still runs: flip the profile's `diarize` to
`pyannote_3_1` and revisit.

## Phase 8 — Model cache and filesystem placement

```bash
echo 'export HF_HOME=$HOME/.cache/huggingface' >> ~/.bashrc
```

The test suite locates the model cache through `HF_HUB_CACHE` / `HF_HOME` too, so a cache on another
drive is fine — but if those are unset in the shell running `pytest`, the eight batched-decode
equivalence tests skip rather than fail, and the summary still looks green. Check for `s` characters
in pytest's progress output, or run with `-rs` to see skip reasons.

Keep both the repo and `HF_HOME` **inside** the WSL filesystem (`~/`), never under `/mnt/c/`.
Cross-OS access goes through a slow translation layer, and `runs/` is a live working directory with
an appended-and-seek-tailed progress log, non-atomic per-stage manifest rewrites, and in-place
ffmpeg/soundfile writes. Around 20 GB of models land in the cache; WSL's virtual disk grows on
demand, so the space comes off the Windows drive.

## Phase 9 — Select the CUDA profile, then run end to end

Do **not** edit `[pipeline]` in `patmir.toml`. That table is the Apple-Silicon dev laptop, and
editing it on this box makes the two working copies diverge on the one file that decides which
model runs. Instead select the machine profile:

```bash
echo 'export PATMIR_PROFILE=cuda' >> ~/.bashrc
source ~/.bashrc
```

`[pipeline.profiles.cuda]` in `patmir.toml` overlays the stages that have to change:

| Stage | Profile selects | Note |
| --- | --- | --- |
| `ingest` / `mux` / `output` | unchanged | ffmpeg + rubberband, no accelerator |
| `emotion` | unchanged | `emotion2vec.py` resolves device `auto` to cuda |
| `translate` | unchanged | cloud (OpenRouter), platform-independent |
| `separate` | `htdemucs_vocals` | torch twin of `htdemucs_mlx_vocals`; same vocals-specialist sub-model (`04573f0d`), `accompaniment = mix - vocals` |
| `asr` | `qwen3_asr_1_7b` | torch sibling of the MLX candidate; bf16 rather than 8-bit |
| `diarize` | `sortformer_nemo` | Same Sortformer model as the Mac, on NVIDIA's own NeMo. Needs the extra install below |
| `tts` | `higgs_v3_torch_bf16` | same Higgs v3 model, decoded **batched across sentences** (`max_batch`, default 8) |

Watch-along additionally picks up `[pipeline.profiles.cuda.streaming]`, which replaces the MLX-only
`separate` override that `streaming/runner.py` hardcodes.

Anything unexpected here should fail loudly: a typo'd profile name is a hard error rather than a
silent fall back to the MLX candidates. Confirm the resolved set before running anything long:

```bash
python -c "
from pathlib import Path
from patmir.pipeline.config import load_pipeline_config
cfg = load_pipeline_config(Path('patmir.toml'))
for stage, entry in cfg.entries.items():
    print(f'{stage:10} {entry[\"id\"]}')
"
```

Every id should be the profile's, and none should contain `mlx`.

### First run

```bash
patmir bench samples/Reacher-S02E01-5min.mkv --mode batch \
  --report runs/bench/reacher-cuda-baseline.json --label cuda-first-run
```

Baselines to beat, measured warm on the M1 Pro against the same clip (301.1 s of source):

| | batch | watch-along |
| --- | --- | --- |
| agg RTF | 1.17 | 1.611 |
| TTS share of wall | 244.4 s (69.4%) | 242.5 s (54.9%) |
| ASR share of wall | 38.0 s (10.8%) | 69.4 s (15.7%) |

The projection for this box is ~0.35 batch / ~0.51 watch-along. It is a projection from memory
bandwidth — **measure it, do not assume it.**

Two things to watch on the first run, both consequences of the profile rather than the hardware:

- **Diarization.** `sortformer_nemo` was written against the model card, and two things about
  NeMo's runtime are documented thinly: `diarize()`'s return type is described but never shown
  literally, and the model wants mono 16 kHz input. Both are handled defensively — the parser
  accepts string/tuple/object segments and **raises** on anything else rather than guessing, and the
  stem is converted with ffmpeg into `diarize/vocals_16k_mono.wav` before NeMo sees it. If the parser
  raises, its message carries the offending value, which should make it a one-line fix. Check
  `diarize/turns.json` for a plausible speaker count either way; if NeMo won't install at all, flip
  the profile's `diarize` to `pyannote_3_1` and accept the short-turn merging (quality issue #13).
- **TTS output.** `higgs_v3_torch_bf16` had never been executed on a GPU when it was written. Its
  batching is verified bitwise against single-sentence decode on a tiny CPU model, but throughput is
  not. Ear-test one line before benching a full clip, then check
  `tts/voice_cloning_summary.json` for `overflows` and `batches`.

  `max_batch = 8` in the candidate entry is a conservative guess from memory arithmetic, not a
  measurement — weights are ~8 GB bf16 and the KV cache only ~144 KiB/token (GQA, 8 KV heads), so
  B=8 over a 1500-token context adds just ~1.7 GiB. The ceiling should be compute, not memory, so
  sweep it upward once a baseline exists:

  ```bash
  for b in 4 8 16; do
    # edit max_batch in patmir.toml, then:
    patmir bench samples/Reacher-S02E01-5min.mkv --mode batch \
      --report runs/bench/reacher-cuda-b$b.json --label "max_batch=$b"
  done
  ```

  Under batching, `tts/timings.jsonl` rows are per **chunk**, not per sentence (`"kind": "chunk"`),
  because a per-sentence wall no longer exists — the chunk's cost is shared. Bench's `total_s` and
  `tts_rtf` stay correct; its mean/median/p90/max describe chunks.

Then, in order: fix whatever the first run surfaces, capture the baseline, and only then look at
optimizations (batched TTS decode across sentences, section pipelining — which pays here in a way
it cannot on MLX, where everything serializes on one GPU stream).

## Phase 10 — Server reachability

```bash
patmir ui --port 8000
```

Open `http://localhost:8000` from Windows Chrome. If unreachable, add `networkingMode=mirrored`
to `.wslconfig` and `wsl --shutdown`.

## Pitfalls, by likelihood

| Symptom | Cause |
| --- | --- |
| `torch.cuda.is_available()` is False | Python 3.14 instead of 3.12 — check `python --version` first, always |
| `nvidia-smi` works but torch sees no GPU | a Linux driver package was installed inside WSL |
| Everything inexplicably slow | repo or `HF_HOME` sitting on `/mnt/c/` |
| Run dies after the first stage with `libmlx.so: cannot open shared object file` | Two compounding causes, both fixed in-tree — `git pull` if you hit it. `[local]` used to install the Apple-only `mlx-*` wheels here (now gated on `sys_platform == 'darwin'`), and `_free_accelerator_cache()`, which runs after **every** stage, probed MLX under `except ImportError` while dlopen actually raises `OSError`. It fired regardless of the active profile, so a correct `PATMIR_PROFILE=cuda` did not save you. Belt and braces: `uv pip uninstall mlx mlx-audio mlx-whisper mlx-lm` |
| Extension cannot reach the server | missing `networkingMode=mirrored` |
| Mux stage fails | `rubberband-cli` not installed |
| Higgs load report says `audio_head.weight \| MISSING \| newly initialized` | **Expected, not a problem.** The checkpoint omits that key because the model ties it to `audio_embedding.weight`; the tie is applied after the report is printed. Confirmed tied on the 3090 (shared storage, identical norms). The candidate asserts the tie at build time, so a genuinely broken tie is a hard error rather than silent noise |
| flash-attention will not build | use a prebuilt cu128 wheel instead of compiling |
| WSL2 gets OOM-killed | `memory=` too low in `.wslconfig` |

## GPU power limit

**Set it on the Windows side, not in WSL.** The GPU is owned by the Windows driver and WSL only gets
a stub (`libcuda.so` is stubbed in, and `nvidia-smi` lives at `/usr/lib/wsl/lib/nvidia-smi`).
NVIDIA's CUDA-on-WSL guide states that "modifiable state features (ECC, Compute mode, Persistence
mode) will not be supported" — it does not name the power limit specifically, but that is the
category it falls in, so expect `nvidia-smi -pl` inside WSL to be refused.

Whatever limit is applied on the Windows side applies to the card globally, so WSL workloads inherit
it. There is nothing to configure inside WSL.

In order of reliability on a consumer GeForce card:

1. **MSI Afterburner** — the Power Limit slider (set to 80%, then apply). Reliable on GeForce because
   it goes through the vendor overclocking API rather than NVML. Tick the startup option to make it
   persist across reboots.
2. **Windows `nvidia-smi -pl 280`** in an Administrator prompt (280 W = 80% of a 350 W 3090). Often
   refused on GeForce cards under WDDM, where power management through NVML is restricted; worth one
   attempt, but Afterburner is the fallback.
3. **WSL `nvidia-smi -pl`** — expect "not supported". Listed only so nobody spends time on it.

Worth doing for long runs: a memory-bandwidth-bound workload gives up very little to a reduced core
power limit while shedding real heat, which matters on a 3090 whose GDDR6X sits on both sides of the
PCB. It is a mitigation, though, not a fix — if the memory junction runs hot, the cause is usually
the thermal pads rather than the power target.
