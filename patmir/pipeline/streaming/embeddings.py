"""Speaker-embedding extraction for cross-section identity (Slice 2 #8/#9).

Wraps speechbrain's ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) to turn a
speaker's reference clip into a 192-dim voice embedding. The streaming runner
embeds each section's speakers and cosine-matches them against the registry's
stored per-character embeddings (see `SpeakerRegistry.resolve`) so the same
character keeps one voice across sections.

ECAPA is tiny and embeddings are computed only a handful of times (once per
speaker in the prelude, per new speaker per section), so it runs on CPU — that
also sidesteps a speechbrain 1.1.0 issue where passing `run_opts={"device":...}`
raises `'EncoderClassifier' object has no attribute 'device_type'`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
_SAVEDIR = Path.home() / ".patmir" / "models" / "ecapa"


@lru_cache(maxsize=1)
def _encoder():
    """Load the ECAPA encoder once (CPU). Returns the classifier or None."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        # No run_opts → CPU; see module docstring for why.
        return EncoderClassifier.from_hparams(source=_MODEL, savedir=str(_SAVEDIR))
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"embeddings: ECAPA unavailable ({type(exc).__name__}: {exc}); "
              f"cross-section identity matching disabled.")
        return None


def _load_mono_16k(ref_audio: Path | str):
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(ref_audio), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        n = int(len(y) * ratio)
        y = np.interp(
            np.linspace(0, 1, n, endpoint=False),
            np.linspace(0, 1, len(y), endpoint=False),
            y,
        ).astype("float32")
    return y


def embed(ref_audio: Path | str | None):
    """Return a 192-dim ECAPA embedding (np.ndarray) for `ref_audio`, or None
    if the clip is missing/too short or the encoder is unavailable."""
    if ref_audio is None or not Path(str(ref_audio)).exists():
        return None
    clf = _encoder()
    if clf is None:
        return None
    import numpy as np
    import torch

    y = _load_mono_16k(ref_audio)
    if y.size < 16000 * 0.3:  # <0.3s is too little to embed reliably
        return None
    wav = torch.from_numpy(y).unsqueeze(0)
    try:
        emb = clf.encode_batch(wav).squeeze().detach().cpu().numpy()
    except Exception as exc:  # pragma: no cover
        print(f"embeddings: encode failed ({type(exc).__name__}: {exc}).")
        return None
    return np.asarray(emb, dtype="float32").reshape(-1)


def cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype="float32").reshape(-1)
    b = np.asarray(b, dtype="float32").reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
