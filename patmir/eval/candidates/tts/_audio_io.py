"""Reference-clip decoding for the Higgs TTS candidates.

Both Higgs candidates used `torchaudio.load()` to read a speaker's reference
WAV. On torch >= 2.11 torchaudio no longer decodes anything itself — it
delegates to `torchcodec`, whose native libraries link against both torch and
the system FFmpeg. On the WSL2 / RTX 3090 box that dlopen fails outright:

    Could not load this library:
    .../site-packages/torchcodec/libtorchcodec_image.so

which killed the TTS stage right after the codec finished loading. The separate
stage had already hit the same wall from the other direction — Silero's
`read_audio` helper — and worked around it with soundfile plus `np.interp`.

Linear interpolation is fine for a VAD frontend and NOT fine here: this clip
is what the codec clones the character's voice from, and `np.interp` aliases.
So we decode with **ffmpeg**, which is already a hard dependency of this
project (ingest, mux and output all shell out to it), does high-quality
resampling, and reads whatever container the ref clip happens to be in. No
torchaudio, no torchcodec, no new dependency, and one subprocess per speaker
because the callers cache the result.

Shared by `higgs_v3_torch.py` and `higgs_v3_mlx.py` so the two runtimes cannot
drift on how a reference clip is read.
"""
from __future__ import annotations

import subprocess

import numpy as np


def decode_mono(path: str, sample_rate: int) -> np.ndarray:
    """Decode `path` to a 1-D float32 array, mono, at `sample_rate`.

    Raises `RuntimeError` with ffmpeg's own stderr if decoding fails — a
    reference clip that won't decode is a real problem (the speaker would
    silently fall back to a generic voice), so it is not swallowed here.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg could not decode reference clip {path!r}: {stderr}")

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError(f"reference clip {path!r} decoded to zero samples")
    # `frombuffer` returns a read-only view onto the subprocess bytes; torch
    # warns (and may alias) on non-writable buffers, so hand back a copy.
    return np.array(samples, dtype=np.float32)
