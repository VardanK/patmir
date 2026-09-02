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
        with ctx.probe.section("read_s"):
            audio_full, src_sr = sf.read(str(separated.vocals), dtype="float32",
                                         always_2d=True)
        mono_full = audio_full.mean(axis=1)

        # A linear np.interp over the whole stem (~13 M samples at 44.1 kHz).
        with ctx.probe.section("resample_s"):
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

        n_turns = max(1, len(diarization.turns))
        ctx.probe.bump("n_turns", len(diarization.turns))

        # Slice first, score in batches. This was one `generate` call per turn
        # — ~108 calls on a 5-minute clip — which paid funasr's per-call setup
        # every time. Handing it the list instead measured **2.96x** on 24 real
        # turns (0.65 s -> 0.22 s) with **24/24 labels and confidences
        # identical**, because funasr still runs each clip at batch_size 1
        # internally: there is no padding to leak into the pooled embedding,
        # which is what would otherwise have made this a quality change rather
        # than a scheduling one.
        clips: list[Any] = []
        scored_at: list[int] = []          # index into `diarization.turns`
        for i, turn in enumerate(diarization.turns):
            clip = _slice_mono(mono_16k, 16_000, turn.start, turn.end)
            if len(clip) < int(0.2 * 16_000):
                continue                   # < 200 ms — see the neutral default below
            clips.append(clip)
            scored_at.append(i)

        labels: dict[int, tuple[str, float]] = {}
        for offset in range(0, len(clips), EMOTION_BATCH):
            group = clips[offset:offset + EMOTION_BATCH]
            with ctx.probe.section("infer_s"):
                raw = self.model.generate(group, granularity="utterance",
                                          extract_embedding=False)
            ctx.probe.bump("n_scored", len(group))
            # One result per clip. A shape mismatch would silently misalign
            # every emotion with the wrong turn, so fall back to per-clip
            # calls rather than trusting the length.
            rows = raw if isinstance(raw, list) and len(raw) == len(group) else None
            for k, clip in enumerate(group):
                item = [rows[k]] if rows is not None else self.model.generate(
                    clip, granularity="utterance", extract_embedding=False)
                labels[scored_at[offset + k]] = _top_label(item)
            done = min(len(clips), offset + EMOTION_BATCH)
            ctx.progress("emotion", 0.1 + 0.85 * done / max(1, len(clips)),
                         f"scored {done}/{len(clips)} turns")

        predictions: list[EmotionPrediction] = []
        for i, turn in enumerate(diarization.turns):
            if i not in labels:
                # Too short to score reliably; default to neutral.
                predictions.append(EmotionPrediction(
                    start=turn.start, end=turn.end, emotion="neutral",
                    confidence=0.0,
                ))
                continue
            label, conf = labels[i]
            if conf < self.min_confidence:
                emo: Emotion = "neutral"
            else:
                emo = self.bin_map.get(label.lower(), "neutral")
            predictions.append(EmotionPrediction(
                start=turn.start, end=turn.end, emotion=emo,
                confidence=float(conf),
            ))

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


# Turns handed to funasr per call. It processes them at batch_size 1 internally,
# so this only amortizes per-call setup and a larger value buys nothing beyond
# a point; 32 keeps the progress line moving on a long clip.
EMOTION_BATCH = 32


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
