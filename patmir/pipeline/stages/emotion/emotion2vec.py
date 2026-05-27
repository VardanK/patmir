"""Speech-emotion-recognition via emotion2vec_plus (funasr).

Runs `iic/emotion2vec_plus_base` on the Demucs vocal stem, one inference per
diarization turn. Returns `EmotionPrediction`s — one per turn window —
already collapsed from emotion2vec's 9-class output to the 5 ref-clip bins
declared in `patmir.toml [emotion_bins]`.

Predictions whose top-class confidence falls below `min_confidence` are
forced to `"neutral"` — emotion2vec is reasonably calibrated but the masha
clip is Russian and emotion2vec's training corpus is biased toward EN/ZH;
the floor avoids confidently-wrong tags on ambiguous segments.

Persists `emotion/segments.json` for the runs UI + resume path. The runner
later uses these predictions to (a) populate `SpeakerTurn.emotion` via
ref_picker's binning pass and (b) tag each ASR sentence via
`apply_emotion_tags`.

Lazy loads `funasr` because it's a heavy import (pulls modelscope).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    Diarization,
    EmotionPrediction,
    Emotion,
    SeparatedAudio,
)


# emotion2vec_plus class labels in order — see iic/emotion2vec_plus_base model card.
EMO2VEC_LABELS: tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

# Default 9→5 bin mapping used when the [emotion_bins] config is not provided.
DEFAULT_BIN_MAP: dict[str, Emotion] = {
    "angry":     "angry",
    "disgusted": "angry",
    "fearful":   "fearful",
    "happy":     "happy",
    "neutral":   "neutral",
    "other":     "neutral",
    "sad":       "sad",
    "surprised": "happy",
    "unknown":   "neutral",
}


def _resolve_device(requested: str) -> str:
    if requested in ("cpu", "cuda", "mps"):
        return requested
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _slice_mono(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Slice `audio` (1-D float32) to [start, end] seconds. Returns float32."""
    i0 = max(0, int(start * sr))
    i1 = min(len(audio), int(end * sr))
    if i1 <= i0:
        return np.zeros(0, dtype="float32")
    return audio[i0:i1].astype("float32", copy=False)


class Emotion2VecCandidate:
    id = "emotion2vec_plus_base"

    def __init__(
        self,
        model_id: str = "iic/emotion2vec_plus_base",
        device: str = "auto",
        min_confidence: float = 0.5,
        bin_map: dict[str, Emotion] | None = None,
        hub: str = "hf",
    ) -> None:
        from funasr import AutoModel  # lazy, heavy

        self.device = _resolve_device(device)
        # funasr defaults to ModelScope (mainland-China-hosted) which is slow
        # from outside Asia — emotion2vec is mirrored to HuggingFace under
        # `emotion2vec/emotion2vec_plus_base`; using `hub="hf"` pulls from
        # Cloudflare-fronted HF and is dramatically faster.
        hf_model_id = model_id
        if hub == "hf" and model_id.startswith("iic/"):
            # Map ModelScope id `iic/emotion2vec_plus_base` →
            # HuggingFace id `emotion2vec/emotion2vec_plus_base`.
            hf_model_id = "emotion2vec/" + model_id.split("/", 1)[1]
        self.model = AutoModel(model=hf_model_id, device=self.device, hub=hub)
        self.min_confidence = float(min_confidence)
        self.bin_map: dict[str, Emotion] = dict(DEFAULT_BIN_MAP)
        if bin_map:
            # Allow overrides; unknown keys fall back to the default map.
            for k, v in bin_map.items():
                k_norm = k.lower()
                if v in ("neutral", "happy", "sad", "angry", "fearful"):
                    self.bin_map[k_norm] = v  # type: ignore[assignment]

    def detect(
        self,
        separated: SeparatedAudio,
        diarization: Diarization,
        ctx: JobContext,
    ) -> list[EmotionPrediction]:
        ctx.progress("emotion", 0.05, f"loaded emotion2vec on {self.device}")

        # Read the full vocal stem once, mono-downmix in memory. emotion2vec
        # was trained on 16 kHz mono; resample on the fly.
        audio_full, src_sr = sf.read(str(separated.vocals), dtype="float32", always_2d=True)
        mono_full = audio_full.mean(axis=1)

        if src_sr != 16_000:
            n_target = int(len(mono_full) * 16_000 / src_sr)
            if n_target > 0:
                x_old = np.linspace(0, 1, len(mono_full), endpoint=False)
                x_new = np.linspace(0, 1, n_target, endpoint=False)
                mono_16k = np.interp(x_new, x_old, mono_full).astype("float32")
            else:
                mono_16k = np.zeros(0, dtype="float32")
            scale = 16_000 / src_sr
        else:
            mono_16k = mono_full.astype("float32", copy=False)
            scale = 1.0

        # Convert turn time windows to 16 kHz indices via the scale factor —
        # turn timestamps are in seconds, independent of the source SR.
        del scale  # unused, but kept for readability

        predictions: list[EmotionPrediction] = []
        n_turns = max(1, len(diarization.turns))
        for i, turn in enumerate(diarization.turns):
            clip = _slice_mono(mono_16k, 16_000, turn.start, turn.end)
            if len(clip) < int(0.2 * 16_000):
                # < 200 ms — too short to score reliably; default to neutral.
                predictions.append(EmotionPrediction(
                    start=turn.start, end=turn.end, emotion="neutral", confidence=0.0,
                ))
                continue

            raw = self.model.generate(clip, granularity="utterance", extract_embedding=False)
            label, conf = _top_label(raw)
            if conf < self.min_confidence:
                emo: Emotion = "neutral"
            else:
                emo = self.bin_map.get(label.lower(), "neutral")
            predictions.append(EmotionPrediction(
                start=turn.start, end=turn.end, emotion=emo, confidence=float(conf),
            ))

            if (i + 1) % 10 == 0 or i + 1 == n_turns:
                ctx.progress("emotion", 0.1 + 0.85 * (i + 1) / n_turns,
                             f"scored {i + 1}/{n_turns} turns")

        emotion_dir = ctx.job_dir / "emotion"
        emotion_dir.mkdir(exist_ok=True)
        (emotion_dir / "segments.json").write_text(json.dumps({
            "model": self.id,
            "device": self.device,
            "min_confidence": self.min_confidence,
            "predictions": [
                {
                    "start": p.start,
                    "end": p.end,
                    "emotion": p.emotion,
                    "confidence": p.confidence,
                }
                for p in predictions
            ],
        }, indent=2))

        # Count bin coverage for the progress message.
        from collections import Counter
        bin_counts = Counter(p.emotion for p in predictions)
        ctx.progress("emotion", 1.0,
                     f"{len(predictions)} turns: " + ", ".join(
                         f"{e}={n}" for e, n in bin_counts.most_common()))
        return predictions


def _top_label(raw: Any) -> tuple[str, float]:
    """Parse funasr's emotion2vec output to (label, confidence).

    funasr returns a list of dicts; for emotion2vec each dict has 'labels'
    (list[str]) and 'scores' (list[float]) in matching order. We take the
    argmax of scores.
    """
    if isinstance(raw, list) and raw:
        item = raw[0]
    else:
        item = raw
    labels = item.get("labels") if isinstance(item, dict) else None
    scores = item.get("scores") if isinstance(item, dict) else None
    if not labels or not scores or len(labels) != len(scores):
        return ("neutral", 0.0)
    best = max(range(len(scores)), key=lambda j: scores[j])
    label_full = labels[best]
    # emotion2vec labels are commonly formatted like "生气/angry" or "angry/愤怒"
    # Split on "/" and look for an English match.
    parts = [p.strip().lower() for p in str(label_full).split("/")]
    for p in parts:
        if p in EMO2VEC_LABELS:
            return (p, float(scores[best]))
    # Fall back to the raw string.
    return (str(label_full).lower(), float(scores[best]))


def build(config: dict[str, Any]) -> Emotion2VecCandidate:
    return Emotion2VecCandidate(
        model_id=config.get("model_id", "iic/emotion2vec_plus_base"),
        device=config.get("device", "auto"),
        min_confidence=float(config.get("min_confidence", 0.5)),
        bin_map=config.get("bin_map"),
        hub=config.get("hub", "hf"),
    )
