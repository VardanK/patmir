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

**It has to be in the WINDOWS user profile.** `C:\Users\<you>\.wslconfig`, which is
`/mnt/c/Users/<you>/.wslconfig` seen from inside WSL — *not* `~/.wslconfig` in the Linux
home, where it is silently ignored. Found on this box 2026-09-03: the file existed at
`/home/vardan/.wslconfig` with all four settings, and **none of them were in effect** —
15 GB of memory instead of 24 (the default is 50 % of the host), all 16 threads instead
of 12, 4 GB of swap instead of 8, and NAT instead of mirrored networking, which is what
made a LAN-bound server unreachable. Nothing warns you; every value just stays at its
default.

Verify after `wsl --shutdown`, from inside WSL — each line should match the file:

```bash
free -g | awk '/Mem:/ {print "memory " $2 " GB"}'   # 24, not 15
nproc                                               # 12, not 16
ip route get 1.1.1.1                                # mirrored: your LAN gateway,
                                                    # NAT: a 172.x.x.1 address
```

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
uv pip install -e ".[local,ui,dev,cuda]"
pytest patmir/tests/ -q
```

The `cuda` extra carries the deps only this machine needs — `transformers>=5.13` for the
torch Qwen3-ASR candidate and `nemo_toolkit[asr]` for Sortformer. It is deliberately
excluded from `all`, since NeMo has no business on the Apple-Silicon laptop where those
candidates cannot be selected anyway.

**Do not install `qwen-asr`.** It is the official Qwen package and the obvious way to run
Qwen3-ASR on torch, and it pins `transformers==4.57.6` *exactly* — mutually exclusive
with the Higgs TTS checkpoint's `>=5.5`. It is also unnecessary: transformers 5.13 carries
`qwen3_asr` natively, which is what the `qwen3_asr_1_7b` candidate drives. The floor in
the `cuda` extra makes a downgrade fail at install time rather than at TTS build time.

NeMo is the heaviest dependency in the stack and the most likely to fight the Python version — it is
why 3.12 is pinned. The `cuda` extra installs it from PyPI; verify it landed:

```bash
python -c "from nemo.collections.asr.models import SortformerEncLabelModel; print('nemo ok')"
```

If the PyPI build fails, try the git tip, which the model card recommends:

```bash
uv pip install Cython
uv pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"
```

Then re-check torch — NeMo pins a version and can quietly replace the cu128 build with a generic
wheel, leaving you on CPU with no error anywhere:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If it refuses to build at all, the pipeline still runs: flip the profile's `diarize` to
`pyannote_3_1` and accept the short-turn merging (quality issue #13).

## Phase 8 — Model cache and filesystem placement

```bash
echo 'export HF_HOME=$HOME/.cache/huggingface' >> ~/.bashrc
```

**This is done, and it mattered more than expected.** With `HF_HOME` on the 9p
Windows mount, cold model builds cost **154 s** and `performance.md` attributed
53 % of a cold run to them. On the ext4 virtual disk the same builds cost
**22-26 s**. Any cold-start figure in the docs older than 2026-09-01 was
measured across the slow mount.

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
| `asr` | `qwen3_asr_1_7b` | Qwen3-ASR 1.7B bf16 — the same model family as the Mac, via transformers' native `qwen3_asr` (needs `transformers>=5.13`). Takes the **`-hf`** checkpoints. Chunks on the separate stage's VAD silences and decodes them batched. `faster_whisper_large_v3_fp16` stays registered as the no-Qwen fallback |
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
patmir bench samples/sample_1.mkv --mode batch --repeat 3 \
  --report runs/bench/sample_1-cuda-batch.json --label cuda-baseline
```

`samples/sample_1.mkv` is the Reacher 5-minute clip (301.1 s), renamed from
`Reacher-S02E01-5min.mkv` — see `docs/architecture/eval-clips.md`. `--repeat 3`
yields the cold number (repeat 0) and the warm baseline (repeats 1-2) in one
command; the warm one is what the M1 figures below are comparable to.

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
    patmir bench samples/sample_1.mkv --mode batch --repeat 2 \
      --report runs/bench/sample_1-cuda-b$b.json --label "max_batch=$b"
  done
  ```

  Under batching, `tts/timings.jsonl` rows are per **chunk**, not per sentence (`"kind": "chunk"`),
  because a per-sentence wall no longer exists — the chunk's cost is shared. Bench's `total_s` and
  `tts_rtf` stay correct; its mean/median/p90/max describe chunks.

Then, in order: fix whatever the first run surfaces, capture the baseline, and only then look at
optimizations (batched TTS decode across sentences, section pipelining — which pays here in a way
it cannot on MLX, where everything serializes on one GPU stream).

### The real VRAM ceiling here is the Windows COMMIT LIMIT, not the card

**Measured 2026-08-24. Read this before diagnosing any CUDA OOM on this box.**

A WSL2 CUDA process cannot allocate the 3090's 24 GB. It gets whatever Windows'
*commit charge* headroom allows, because **every GB of VRAM allocated consumes ~1 GB
of Windows commit**. Measured directly:

| GPU allocated | Windows commit used | delta | VRAM still free |
| --: | --: | --: | --: |
| 1 GB | 21.02 GB | +1.28 GB | 21.76 GB |
| 3 GB | 22.94 GB | +3.21 GB | 19.76 GB |
| 6 GB | 26.09 GB | +6.36 GB | 16.76 GB |

**1.06 GB of commit per GB of VRAM.** Under the Windows display driver model a GPU
allocation must have a system-memory *backing store* (so the driver can evict it off
the card if another app needs the GPU), and Windows commits that backing up front —
even though the data lives on the card and the backing is typically never touched.

Commit is a *promise*, not usage: `commit limit = physical RAM + pagefile size`. So
the pagefile's size caps GPU allocation whether or not anything is ever paged.

On this box as first configured:

```
physical RAM               31.84 GB
pagefile (fixed, manual)    4.00 GB   -> commit limit 35.84 GB
idle commit baseline       ~19.70 GB
=> headroom for CUDA       ~16.00 GB   on a 24 GB card
```

That ~16 GB is *just under* the ~15 GB that the warm stage cache plus Higgs wants,
which is why the failure looked intermittent rather than systematic.

**How the failure presents, and why it misleads:**

```
CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of
24.00 GiB of which 14.46 GiB is free. Including non-PyTorch memory, this process
has 17179869184.00 GiB memory in use. Of the allocated memory 8.23 GiB is
allocated by PyTorch
```

Every part of that is a red herring:

- **A tiny request fails** (20-48 MiB) — so it is never a capacity problem in the
  usual sense. The refusal comes from commit, and it lands on whatever allocation
  happens to be next, pointing the traceback at innocent code (for us, the audio
  codec moving to device inside `_encode_ref`).
- **"14.46 GiB is free" is true** and irrelevant — that is VRAM, and VRAM was not the
  constraint.
- **`17179869184.00 GiB` is 2^34 GiB**, i.e. garbage. PyTorch infers "non-PyTorch
  memory" by subtraction and the arithmetic goes negative, because the refusal did
  not come from VRAM accounting at all.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which the message suggests,
  **does not help** — verified. Reserved-but-unallocated was 15 MiB, so there was no
  fragmented pool to compact.

**Observed ceilings, all consistent with commit headroom at the time:**

| scenario | allocated at failure | commit baseline |
| --- | --: | --- |
| bare torch process, quiet machine | 20.75 GB | lowest |
| four non-TTS stage models resident | 15.23 GB | ~19.7 GB |
| `patmir bench` warm cache, loading Higgs | 13.06 GB | higher |
| `patmir dub`, Chrome playing video | 8.23 GB | Chrome holding several GB |

**The fix is a Windows setting, not code.** Raise the pagefile — it lifts the commit
limit without causing paging, because nothing is written to it while physical RAM is
free (measured: 4 GB pagefile holding **0.24 GB**, lifetime peak 2.2 GB, with 21.85 GB
of RAM free). Sizing:

| goal | VRAM | commit limit needed | pagefile |
| --- | --: | --: | --: |
| warm cache + Higgs, comfortably | ~15 GB | ~40 GB | ~10 GB |
| use the whole card (e.g. a `max_batch` sweep) | 24 GB | ~48-50 GB | ~16-20 GB |

Put it on **E:** rather than C: — C: had 51 GB free, E: had 152 GB. The only cost is
disk space.

**Confirmed on this box 2026-08-24.** Pagefile moved to `E:` and grown to 20 GB:

```
commit limit   35.84 GB -> 51.84 GB     (exactly the +16 GB from 4 GB -> 20 GB)
```

The warm-cache batch bench then ran with **VRAM peaking at ~17.1 GB** and commit topping
out around 38 GB of the 51.84 available — comfortably past the 13.06 GB where every
previous attempt died, and past the 15.23 GB ceiling measured under the old limit.

Two practical notes from doing it:

- **Set `initial` equal to `maximum`.** Configured as initial 4 GB / max 20 GB, Windows
  grows the file on demand and the commit limit only rises as it grows — so a single
  large burst (Higgs moves ~8.5 GB in one `.to(device)`) can outrun the expansion. It
  did grow in time here, but initial = maximum = 20480 MB removes the race.
- **A reboot helps independently of the pagefile**, by dropping the commit baseline.
  Before: baseline ~19.7 GB. After: ~11 GB, i.e. ~8 GB more headroom for free. If a run
  fails near the limit, a restart is a legitimate first move.

**Corollary, and it still stands on its own:** keep the Windows desktop quiet during a
benchmark. Chrome playing video holds *commit*, which directly shrinks CUDA's ceiling.
Two runs failed at 14:02 and 14:07 with a video playing and the identical run
succeeded at 14:15 with Chrome closed. This is the WSL2 analogue of the `caffeinate`
warning in `performance.md`: a number that looks like a measurement but reflects what
else the machine was doing.

**What this is NOT:**

- **Not faulty VRAM.** Tested 19 GiB with four patterns (zeros, all-ones, 0xAA55
  coupling, address-in-address) = 76 GiB written and verified: **0 pattern errors, 0
  disturb errors on re-read**. No Xid entries; `Channel Repair Pending: No`,
  `TPC Repair Pending: No`. Note the 3090 is a consumer card with no ECC, so the
  ECC counters read `N/A` and cannot themselves clear it — hence the pattern test.
- **Not the shared GPU memory pool.** Windows reports 39.9 GB of "GPU memory"
  (24 dedicated + 15.9 shared). CUDA under WSL2 does not use the shared pool:
  per-process shared usage stayed at **0.21 GB, flat**, throughout a model load. That
  pool is for graphics APIs. Its presence is also not costing anything — like the
  commit limit, it is a ceiling, not a reservation.
- **Not `.wslconfig memory=`.** That is a cap, not a reservation; `vmmemWSL` was
  holding 3.98 GB against a 24 GB cap, with 21.85 GB of physical RAM free. Lowering it
  frees nothing and does not change the commit arithmetic.

### `.wslconfig`: a cap, not a reservation

Worth understanding so it is not mistaken for the problem above.

```bash
cat /mnt/c/Users/$USER/.wslconfig
free -m                       # what the VM actually sees
```

This box is configured `memory=24GB` on a 31.84 GB host. That is a **ceiling on what
the VM may take**, not memory removed from Windows: measured with the VM idle,
`vmmemWSL` held 3.98 GB and Windows had 21.85 GB of physical RAM free. So it is not
what caused the CUDA OOMs (the config was identical across the failures *and* the
successes), and lowering it would not have freed anything.

Two things it does affect, neither of them the ceiling above:

- WSL2 does not return freed memory — page cache included — to Windows by default, so
  the VM's footprint ratchets upward within its cap over a long session. Reading 27 GB
  of models off a 9p mount parks several GB of page cache in the VM.
  `[experimental] autoMemoryReclaim=gradual` hands it back.
- The VM's own memory is charged against the same host commit limit as everything
  else, so a VM that has ratcheted up does reduce CUDA headroom indirectly.

### torchcodec is broken on this box (does not block the pipeline)

`torchaudio.load` raises `OSError: Could not load this library: .../libtorchcodec_image.so` —
ffmpeg 8.0.1 against torchcodec 0.16, which has no build for that ffmpeg major. Nothing on the CUDA
path hits it: the Higgs candidates read reference clips through
`patmir/eval/candidates/tts/_audio_io.py` (an ffmpeg subprocess) and `htdemucs.py` reads through
soundfile. The one remaining `torchaudio` call, `functional.resample` in `qwen3_asr.py`, is pure
torch DSP with no codec involved. But **any new code reaching for `torchaudio.load` will hit it**,
so decode audio through `_audio_io` or soundfile.

## Phase 10 — Server reachability

```bash
patmir ui --profile cuda --port 8000
```

Open `http://localhost:8000` from Windows Chrome. If unreachable, add `networkingMode=mirrored`
to `.wslconfig` and `wsl --shutdown`.

`--profile cuda` matters: it exports `PATMIR_PROFILE`, so the watch-along path
picks the profile up too. Without it the server boots the MLX candidate set and
fails on the first job.

### From another machine on the LAN (2026-09-03)

The server binds loopback by default. To reach it from a second machine:

```bash
patmir ui --profile cuda --port 8000 --host 0.0.0.0
```

Then point that machine's extension at `http://<windows-host-ip>:8000` in its
options page, with the same bearer token the server prints at startup.

Three things have to line up, and only the first is ours:

1. **`--host 0.0.0.0`.** Opt-in rather than the default because five
   `/run/<job_id>` progress routes predate the bearer token and a job id is a
   guessable timestamp — so on a shared network, anyone who guesses one can
   read that job's audio and transcript. `/jobs`, `/stream` and `/cache` all
   require the token. The CLI prints a one-line note when the bind is not
   loopback.
2. **`networkingMode=mirrored`** in `.wslconfig` (already set for this box —
   see Phase 1). WSL2's default NAT gives the VM its own address, so binding
   `0.0.0.0` inside it is reachable from Windows but *not* from the LAN.
   Mirrored mode shares the host's interfaces, which is what makes the LAN
   case work at all. Without it you would need a
   `netsh interface portproxy` rule plus a firewall rule on Windows.
3. **Windows Firewall** must allow inbound on the port. A private-network
   rule is enough; the prompt on first bind usually offers it.

No auth change was needed for this: `_is_loopback_origin` already allows
`chrome-extension://`, which is the origin the extension sends, so the
token-protected routes accept it from any machine. A *page* served from the
LAN IP is still refused by that same check.

## Pitfalls, by likelihood

| Symptom | Cause |
| --- | --- |
| `torch.cuda.is_available()` is False | Python 3.14 instead of 3.12 — check `python --version` first, always |
| `nvidia-smi` works but torch sees no GPU | a Linux driver package was installed inside WSL |
| Everything inexplicably slow | repo or `HF_HOME` sitting on `/mnt/c/` |
| Run dies after the first stage with `libmlx.so: cannot open shared object file` | Two compounding causes, both fixed in-tree — `git pull` if you hit it. `[local]` used to install the Apple-only `mlx-*` wheels here (now gated on `sys_platform == 'darwin'`), and `_free_accelerator_cache()`, which runs after **every** stage, probed MLX under `except ImportError` while dlopen actually raises `OSError`. It fired regardless of the active profile, so a correct `PATMIR_PROFILE=cuda` did not save you. Belt and braces: `uv pip uninstall mlx mlx-audio mlx-whisper mlx-lm` |
| Extension stops working for no visible reason | a **second** `patmir ui` was started. Each prewarms the whole model set at boot (~16 GB on CUDA) and the card holds one, so the second starves the first — and nothing in the symptom mentions memory. A single-instance `flock` now refuses the second server by name; if you see that refusal, stop the other server rather than picking a different port |
| Server looks dead for the first minute or two | the startup hook prewarms every model *before* uvicorn accepts connections — ~45 s warm on this box, longer cold. It prints `Application startup complete` when the port goes live |
| Extension cannot reach a server on another host | the manifest grants `http://127.0.0.1:8765/*` only. Since 2026-09-03 the options page **requests** the host permission when you save a non-loopback address, and refuses to save if you decline — so re-save the address and accept the Chrome prompt. Reload the extension after pulling, since `optional_host_permissions` is a manifest change |
| Extension cannot reach the server | missing `networkingMode=mirrored` — and check *where* `.wslconfig` is: in the Linux home it is silently ignored (see Phase 2) |
| Extension on a second machine cannot reach the server | server bound loopback — add `--host 0.0.0.0` (see Phase 10), then check Windows Firewall allows the port |
| Mux stage fails | `rubberband-cli` not installed |
| `uv pip install` reports transformers 4.57.6 vs >=5.5 unsatisfiable | Expected if `qwen-asr` is installed: it pins transformers exactly, and the Higgs TTS checkpoint needs >=5.5. `uv pip uninstall qwen-asr` and never reinstall it — the `qwen3_asr_1_7b` candidate reaches the same models through transformers' native `qwen3_asr` (5.13+), so the package is not needed for Qwen3-ASR on this box |
| ASR dies with `KeyError: 'qwen3_asr'` | transformers older than 5.13, which is where native `qwen3_asr` landed. The candidate gates on this at build time and reports the fix, so you should see a `RuntimeError` naming the version instead; a raw `KeyError` means something bypassed the gate. `uv pip install -U "transformers>=5.13"` |
| ASR fails loading `Qwen/Qwen3-ASR-1.7B` | Missing the `-hf` suffix. The transformers-native conversions are `Qwen/Qwen3-ASR-1.7B-hf` and `Qwen/Qwen3-ForcedAligner-0.6B-hf`; the unsuffixed repos are the layout the `qwen-asr` package reads. The Mac's MLX candidate legitimately keeps the unsuffixed aligner — that one is read by `mlx-qwen3-asr` |
| TTS stage dies on `'list' object has no attribute 'keys'` | `transformers` got resolved down to 4.x — check with `uv pip list \| grep -i transformers`. The Higgs checkpoint needs >=5.5 (its tokenizer_config gives `extra_special_tokens` as a list, which 4.x mishandles). Installing `nemo_toolkit` was what pulled it down. `uv pip install -U "transformers>=5.5"`, then re-verify torch is still the cu128 build |
| Higgs load report says `audio_head.weight \| MISSING \| newly initialized` | **Expected, not a problem.** The checkpoint omits that key because the model ties it to `audio_embedding.weight`; the tie is applied after the report is printed. Confirmed tied on the 3090 (shared storage, identical norms). The candidate asserts the tie at build time, so a genuinely broken tie is a hard error rather than silent noise |
| TTS dies on `Could not load this library: .../torchcodec/libtorchcodec_image.so` | Fixed in-tree — `git pull` if you hit it. On torch >= 2.11 `torchaudio.load()` decodes nothing itself, it delegates to `torchcodec`, whose native libs link against both torch and the system FFmpeg and fail to dlopen on this box. The Higgs candidates read their reference clip through **ffmpeg** now (`_audio_io.decode_mono`) — already a hard dependency, higher-quality resampling, and no torchcodec. The separate stage had hit the same wall from Silero's `read_audio` and was already worked around. If something else in the stack reaches for torchcodec, the underlying dlopen is still worth diagnosing — most likely an FFmpeg major version torchcodec has no build for |
| flash-attention will not build | use a prebuilt cu128 wheel instead of compiling |
| WSL2 gets OOM-killed | `memory=` too low in `.wslconfig` |

## GPU power limit

**Current state (2026-09-02): the card runs at its stock 350 W and is NOT
limited.** The cooling was fixed and the 80 % mitigation below was removed.
This section is kept because it describes how to re-apply one, and because the
history matters: every CUDA figure in `performance.md` before 2026-09-02 was
measured under that limit while nothing in the repo recorded it, and the bench
captured no power or thermal state, so a throttled run could not be told from a
clean one. `patmir bench` now samples power, clocks, temperature and throttle
reasons every second (`patmir/telemetry.py`) and distinguishes a card at its
own stock ceiling — normal under load — from one capped below it, which makes a
run incomparable.

What removing the limit was worth, measured: **warm batch agg RTF 0.393 ->
0.377, about 4 %**, inside the run-to-run spread and exactly what the last
paragraph of this section predicted. Two of three repeats still logged **one
second each** of thermal slowdown at 80-81 C core, which is a transient rather
than sustained throttling but is the first thermal number this repo has had.
Worth re-checking on a feature-length dub, which runs 10-20x longer at the same
duty cycle.

**If you do re-apply one: set it on the Windows side, not in WSL.** The GPU is owned by the Windows driver and WSL only gets
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
