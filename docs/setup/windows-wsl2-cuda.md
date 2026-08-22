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

## Phase 8 — Model cache and filesystem placement

```bash
echo 'export HF_HOME=$HOME/.cache/huggingface' >> ~/.bashrc
```

Keep both the repo and `HF_HOME` **inside** the WSL filesystem (`~/`), never under `/mnt/c/`.
Cross-OS access goes through a slow translation layer, and `runs/` is a live working directory with
an appended-and-seek-tailed progress log, non-atomic per-stage manifest rewrites, and in-place
ffmpeg/soundfile writes. Around 20 GB of models land in the cache; WSL's virtual disk grows on
demand, so the space comes off the Windows drive.

## Phase 9 — End-to-end CUDA run before porting anything

Four of the five model stages move to CUDA with **config flips alone** in `patmir.toml [pipeline]`.
Do this first, to prove the environment before touching the Higgs port.

| Stage | Set to | Note |
| --- | --- | --- |
| `emotion` | unchanged | `emotion2vec.py` resolves `auto` to cuda |
| `asr` | `qwen3_asr_1_7b` | torch sibling of the MLX candidate |
| `diarize` | `pyannote_3_1` | already CUDA-capable |
| `separate` | `htdemucs` | torch fallback; loses the vocals-only optimization for now |
| `tts` | a torch OmniVoice candidate | takes a device string |

```bash
patmir bench samples/Reacher-S02E01-5min.mkv --mode batch \
  --report runs/bench/reacher-cuda-baseline.json --label cuda-first-run
```

That baseline is what later optimization is measured against. Then port Higgs TTS (upstream is
PyTorch, so this replaces the hand-written MLX port and gains bf16 + flash-attention + batched
decode), then decide whether Sortformer is worth NeMo or whether pyannote is good enough.

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
| Extension cannot reach the server | missing `networkingMode=mirrored` |
| Mux stage fails | `rubberband-cli` not installed |
| flash-attention will not build | use a prebuilt cu128 wheel instead of compiling |
| WSL2 gets OOM-killed | `memory=` too low in `.wslconfig` |

## GPU power limit

MSI Afterburner does not exist here. A bandwidth-bound workload loses very little to a reduced
core power limit while shedding meaningful heat, which matters on a 3090 whose GDDR6X sits on both
sides of the PCB:

```bash
sudo nvidia-smi -pl 280      # 80% of a 350 W card
```

Does not survive a reboot — put it in a systemd unit if it should persist.
