"""Speaker profiling — gender + age bucket from a reference clip (Slice 2 #7).

Used by the speaker registry to assign a gender/age-matched pool voice
(ElevenLabs no-clone mode). Primary path is the audeering wav2vec2 age-gender
model (local, MPS); a librosa-pitch heuristic is the fallback if the model
can't be loaded.

Calibration note (measured 2026-06-13): the model's GENDER + child detection
are reliable, but the AGE regression skews high (adult voices read ~60–66y),
so thresholds below are deliberately conservative (elder only ≥70y) and the
voice pool tags each voice with a *range* of ages so a one-bucket age error
still lands on a sensible voice. Treat age as approximate.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from patmir.pipeline.streaming.registry import SpeakerProfile

_MODEL = "audeering/wav2vec2-large-robust-24-ft-age-gender"
_GENDERS = ["female", "male", "child"]
# Conservative buckets — the model over-estimates age, so bias toward "adult".
_YOUNG_MAX, _ADULT_MAX = 34.0, 69.0          # <13 child, <35 young, <70 adult, else elder
_PITCH_MALE_MAX_HZ, _PITCH_CHILD_MIN_HZ = 165.0, 255.0


def _bucket_age(years: float, is_child: bool) -> str:
    if is_child or years < 13:
        return "child"
    if years < _YOUNG_MAX:
        return "young"
    if years < _ADULT_MAX:
        return "adult"
    return "elder"


@lru_cache(maxsize=1)
def _load_model():
    """Load the audeering age-gender model once. Returns (processor, model, device)
    or None if unavailable (caller falls back to pitch)."""
    try:
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class _Head(nn.Module):
            def __init__(self, config, n):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, n)

            def forward(self, x):
                x = self.dropout(x)
                x = torch.tanh(self.dense(x))
                return self.out_proj(self.dropout(x))

        class _AgeGender(Wav2Vec2PreTrainedModel):
            # Compatibility shims for newer transformers (custom class predates them).
            _tied_weights_keys = []
            all_tied_weights_keys = {}

            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = _Head(config, 1)
                self.gender = _Head(config, 3)
                self.init_weights()

            def forward(self, input_values):
                h = torch.mean(self.wav2vec2(input_values).last_hidden_state, dim=1)
                return self.age(h), self.gender(h)

        # CPU by default, for the same reason the ECAPA embedder is CPU-only
        # (see `embeddings.py`): the workload is a handful of short ref clips —
        # once per speaker in the prelude, once per NEW speaker per section — so
        # the accelerator buys almost nothing, while this wav2vec2-large sits on
        # the card for the whole server lifetime once loaded (`lru_cache`).
        #
        # It measured ~1.2 GB of VRAM, and that mattered: on the 24 GB 3090 under
        # WSL2, prewarming it at boot (2026-08-25) put it resident BEFORE the
        # prelude's emotion stage instead of after ASR, and the emotion stage
        # then failed a 250 MiB allocation with 7.1 GB still free on the card —
        # the commit-limit false-OOM documented in
        # `docs/setup/windows-wsl2-cuda.md`. Set PATMIR_PROFILE_DEVICE to
        # override (e.g. "cuda") on a box with headroom to spare.
        forced = os.environ.get("PATMIR_PROFILE_DEVICE", "").strip().lower()
        if forced:
            device = forced
        else:
            device = "cpu"
        processor = Wav2Vec2Processor.from_pretrained(_MODEL)
        model = _AgeGender.from_pretrained(_MODEL).to(device).eval()
        return processor, model, device
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"speaker_profile: age-gender model unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to pitch heuristic.")
        return None


def _profile_with_model(ref_audio: Path | str) -> SpeakerProfile | None:
    bundle = _load_model()
    if bundle is None:
        return None
    import numpy as np
    import soundfile as sf
    import torch

    processor, model, device = bundle
    y, sr = sf.read(str(ref_audio), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    y = y[: sr * 8]  # cap analysis length for speed
    if y.size < sr * 0.2:
        return None
    inp = processor(y, sampling_rate=sr, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        age_logit, gender_logit = model(inp)
    years = float(torch.sigmoid(age_logit)[0, 0]) * 100
    probs = torch.softmax(gender_logit, dim=1)[0]
    is_child = int(torch.argmax(probs)) == 2 or float(probs[2]) > 0.5
    gender = "female" if float(probs[0]) >= float(probs[1]) else "male"
    return SpeakerProfile(gender=gender, age=_bucket_age(years, is_child))


def _profile_with_pitch(ref_audio: Path | str) -> SpeakerProfile:
    """Fallback: gender from median F0; age coarse (child if very high pitch,
    else adult — pitch can't reliably separate young/adult/elder)."""
    try:
        import librosa
        import numpy as np
        import soundfile as sf

        y, sr = sf.read(str(ref_audio), dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        y = y[: sr * 5]
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        vals = f0[~np.isnan(f0)]
        if vals.size < 5:
            return SpeakerProfile()
        med = float(np.median(vals))
        if med >= _PITCH_CHILD_MIN_HZ:
            return SpeakerProfile(gender="any", age="child")
        gender = "male" if med < _PITCH_MALE_MAX_HZ else "female"
        return SpeakerProfile(gender=gender, age="adult")
    except Exception:
        return SpeakerProfile()


def profile_speaker(ref_audio: Path | str | None) -> SpeakerProfile:
    """Return (gender, age bucket) for a speaker's reference clip. Model first,
    pitch fallback, then unknown ("any"/"any") if nothing works."""
    if ref_audio is None or not Path(str(ref_audio)).exists():
        return SpeakerProfile()
    return _profile_with_model(ref_audio) or _profile_with_pitch(ref_audio)
