"""Speaker-embedding extraction for cross-section identity (Slice 2 #8/#9).

Turns a speaker's reference clip into a voice embedding; the streaming runner
embeds each section's speakers and cosine-matches them against the registry's
stored per-character embeddings (see `SpeakerRegistry.resolve`) so the same
character keeps one voice across sections.

Backends (`PATMIR_EMBED_BACKEND` env var):

- `ecapa` (default) — speechbrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`,
  192-dim). Calibrated match threshold 0.12 (2026-06-13, thin margin: 0.157
  same-speaker-angry vs 0.06 different-speaker).
- `redimnet` — IDRnD ReDimNet-b2 via torch.hub (~5M params, ≈3× better EER
  than ECAPA, best published robustness on short/noisy/demographic-shifted
  speech — exactly our cartoon/child-voice failure mode). Use
  `match_threshold()` for the backend's calibrated threshold; never mix
  embeddings from different backends in one registry.

Both are tiny and embeddings are computed only a handful of times (once per
speaker in the prelude, per new speaker per section), so they run on CPU —
for ECAPA that also sidesteps a speechbrain 1.1.0 issue where passing
`run_opts={"device":...}` raises `'EncoderClassifier' has no attribute
'device_type'`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
_SAVEDIR = Path.home() / ".patmir" / "models" / "ecapa"

# Per-backend cosine thresholds for SpeakerRegistry.resolve. ECAPA's 0.12 was
# calibrated 2026-06-13 on Demucs-separated movie speech. ReDimNet's was
# calibrated 2026-07-14 the same way (see the calibration note in
# decisions/2026-07-14-watchalong-voice-consistency-and-warm-loop.md).
_THRESHOLDS = {"ecapa": 0.12, "redimnet": 0.30}


def backend() -> str:
    b = os.environ.get("PATMIR_EMBED_BACKEND", "ecapa").strip().lower()
    return b if b in ("ecapa", "redimnet") else "ecapa"


def match_threshold() -> float:
    return _THRESHOLDS[backend()]


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


@lru_cache(maxsize=1)
def _redimnet():
    """Load ReDimNet-b2 once via torch.hub (CPU). Returns the model or None."""
    try:
        import torch

        model = torch.hub.load(
            "IDRnD/ReDimNet", "ReDimNet",
            model_name="b2", train_type="ptn", dataset="vox2",
            trust_repo=True,
        )
        model.eval()
        return model
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"embeddings: ReDimNet unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to ECAPA.")
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
    """Return a voice embedding (np.ndarray) for `ref_audio` from the active
    backend, or None if the clip is missing/too short or no encoder loads."""
    if ref_audio is None or not Path(str(ref_audio)).exists():
        return None
    import numpy as np

    y = _load_mono_16k(ref_audio)
    if y.size < 16000 * 0.3:  # <0.3s is too little to embed reliably
        return None

    import torch

    wav = torch.from_numpy(y).unsqueeze(0)
    try:
        if backend() == "redimnet":
            model = _redimnet()
            if model is not None:
                with torch.no_grad():
                    emb = model(wav).squeeze().cpu().numpy()
                return np.asarray(emb, dtype="float32").reshape(-1)
            # unavailable → fall through to ECAPA
        clf = _encoder()
        if clf is None:
            return None
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
