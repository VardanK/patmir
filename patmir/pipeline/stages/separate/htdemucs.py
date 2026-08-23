"""Source separation via Demucs (htdemucs_ft).

Extracts full-quality stereo audio from the source video, then runs Demucs to
split it into vocals, drums, bass, and other stems. Drums + bass + other are
summed into an "accompaniment" track that becomes the music + sound-effects
bed under the Armenian dub; vocals are kept as a sidecar for inspection.

`htdemucs_ft` is the fine-tuned bagged 4-model variant — state-of-the-art
quality, ~300 MB of weights. MIT-licensed. Auto-downloaded on first use.

Uses the lower-level `demucs.pretrained` + `demucs.apply` API since
`demucs.api.Separator` only exists on the unreleased tip.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import SeparatedAudio


def _resolve_device(requested: str) -> str:
    if requested in ("cpu", "cuda", "mps"):
        return requested
    try:
        import torch  # type: ignore[import-not-found]
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class HTDemucs:
    id = "htdemucs"

    def __init__(self, model_name: str = "htdemucs_ft", device: str = "auto",
                 vocals_only: bool = False) -> None:
        from demucs.pretrained import get_model  # lazy import (heavy)

        # Vocals-only mode: `model_name` is the bare signature of htdemucs_ft's
        # vocals-specialist sub-model (`04573f0d`) rather than the 4-model bag.
        # The bag's vocals stem comes 100% from that sub-model (its BagOfModels
        # weight matrix is the identity: vocals model -> [0,0,0,1]), so vocals
        # are mathematically identical while the drums/bass/other models are
        # skipped. accompaniment is then derived as `mix - vocals` — this
        # model's own non-vocals stems are not trusted. Mirrors the MLX
        # candidate; ADR 2026-06-19-vocals-only-separation.
        self.vocals_only = vocals_only
        self.device = _resolve_device(device)
        self.model_name = model_name
        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model.eval()
        # model.sources is a list like ['drums', 'bass', 'other', 'vocals'].
        self.stem_names: list[str] = list(self.model.sources)
        self.sample_rate: int = int(self.model.samplerate)
        self.channels: int = int(self.model.audio_channels)

    def separate(self, source_video: Path, ctx: JobContext) -> SeparatedAudio:
        import numpy as np  # lazy
        import soundfile as sf  # lazy
        import torch  # lazy
        from demucs.apply import apply_model  # lazy

        # Demucs needs full-quality stereo. ASR's 16 kHz mono is not good enough.
        # We extract straight to Demucs's native sample rate + channel count so
        # we don't need any resampling step in the Python path.
        full_audio = ctx.job_dir / "full_audio.wav"
        if not full_audio.exists():
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(source_video),
                "-vn",
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-c:a", "pcm_s16le",
                str(full_audio),
            ], check=True, capture_output=True)
        ctx.progress("separate", 0.05, "extracted full-quality audio")

        # soundfile returns (samples, channels); demucs wants (batch, channels, samples).
        audio_np, _sr = sf.read(str(full_audio), dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio_np.T.copy())
        if wav.shape[0] == 1 and self.channels == 2:
            wav = wav.repeat(2, 1)
        wav = wav.unsqueeze(0).to(self.device)
        ctx.progress("separate", 0.10, f"running demucs ({self.model_name}) on {self.device}")

        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav,
                device=self.device,
                split=True,
                progress=False,
            )
        # sources shape: (batch=1, n_sources, channels, samples).
        ctx.progress("separate", 0.85, "stems ready")

        stems = {name: sources[0, i] for i, name in enumerate(self.stem_names)}
        vocals = stems.get("vocals")
        if self.vocals_only and vocals is not None:
            # accompaniment = original mix - vocals (the residual). apply_model
            # can return a slightly different length than it was given, so trim
            # both to the shorter one before subtracting.
            mix = wav[0]
            n = min(mix.shape[-1], vocals.shape[-1])
            vocals = vocals[..., :n]
            accompaniment = mix[..., :n] - vocals
        else:
            accompaniment = torch.zeros_like(stems[self.stem_names[0]])
            for name in self.stem_names:
                if name == "vocals":
                    continue
                accompaniment = accompaniment + stems[name]

        accomp_path = ctx.job_dir / "accompaniment.wav"
        vocals_path = ctx.job_dir / "vocals.wav"
        sf.write(
            str(accomp_path),
            accompaniment.transpose(0, 1).cpu().numpy(),
            self.sample_rate,
            subtype="PCM_16",
        )
        if vocals is not None:
            sf.write(
                str(vocals_path),
                vocals.transpose(0, 1).cpu().numpy(),
                self.sample_rate,
                subtype="PCM_16",
            )

        if not ctx.options.keep_intermediate and full_audio.exists():
            full_audio.unlink()

        # Subdivide vocals.wav into speech vs non-speech vocals via Silero VAD.
        # Demucs's vocal stem captures any vocal-like sound — screams, laughs,
        # grunts — which (a) makes ASR hallucinate and (b) means those
        # atmospheric sounds get lost from the dub bed. The split keeps them.
        speech_path: Path | None = None
        non_speech_path: Path | None = None
        vad_intervals: list[tuple[float, float]] | None = None
        if vocals is not None:
            ctx.progress("separate", 0.9, "running Silero VAD on vocals")
            speech_path, non_speech_path, vad_intervals = _split_vocals_with_vad(
                vocals_path=vocals_path,
                accompaniment_shape=accompaniment.shape,
                sample_rate=self.sample_rate,
                job_dir=ctx.job_dir,
            )

        meta = {
            "accompaniment": str(accomp_path),
            "vocals": str(vocals_path) if vocals is not None else None,
            "speech_vocals": str(speech_path) if speech_path else None,
            "non_speech_vocals": str(non_speech_path) if non_speech_path else None,
            "vad_intervals": vad_intervals,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "model": self.model_name,
            "vocals_only": self.vocals_only,
            "device": self.device,
        }
        (ctx.job_dir / "separated.meta.json").write_text(json.dumps(meta, indent=2))
        ctx.progress("separate", 1.0, str(accomp_path))
        return SeparatedAudio(
            accompaniment=accomp_path,
            vocals=vocals_path,
            sample_rate=self.sample_rate,
            channels=self.channels,
            speech_vocals=speech_path,
            non_speech_vocals=non_speech_path,
            vad_intervals=vad_intervals,
        )


def _split_vocals_with_vad(
    *,
    vocals_path: Path,
    accompaniment_shape: tuple[int, int],
    sample_rate: int,
    job_dir: Path,
) -> tuple[Path, Path, list[tuple[float, float]]]:
    """Run Silero VAD on `vocals_path` and write speech-only / non-speech-only stems.

    Returns (speech_path, non_speech_path, intervals) where intervals is the
    list of (start_seconds, end_seconds) speech windows. ASR uses the intervals
    as a post-filter on its own word timestamps; the .wav stems are kept as
    inspection artifacts and `non_speech.wav` is mixed into the dub bed.

    Silero's `read_audio` helper calls into torchaudio which on >=2.11 requires
    `torchcodec`. We read the WAV with soundfile and resample to 16 kHz mono
    ourselves; the model only needs a 1-D float32 tensor at 16 kHz.
    """
    import numpy as np
    import soundfile as sf
    import torch  # type: ignore[import-not-found]
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=False)

    # 1) Load the vocals at the original SR via soundfile (no torchaudio).
    vocals_np, sr = sf.read(str(vocals_path), dtype="float32", always_2d=True)
    n_samples = vocals_np.shape[0]
    # 2) Build a 16k mono float32 view for Silero.
    mono = vocals_np.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        n_target = int(len(mono) * ratio)
        x_old = np.linspace(0, 1, len(mono), endpoint=False)
        x_new = np.linspace(0, 1, n_target, endpoint=False)
        mono16 = np.interp(x_new, x_old, mono).astype("float32")
    else:
        mono16 = mono.astype("float32")
    wav16 = torch.from_numpy(mono16)

    # Silero's default threshold=0.5 is for noisy real-world recordings. On
    # cartoon dialogue running over music, even after Demucs separation the
    # vocal stem retains some background energy and Silero misses most
    # utterances at 0.3. 0.2 is the floor below which background music
    # starts firing as speech.
    intervals = get_speech_timestamps(
        wav16,
        model,
        sampling_rate=16000,
        return_seconds=True,
        threshold=0.2,
        min_speech_duration_ms=100,
        min_silence_duration_ms=200,
        speech_pad_ms=500,
    )

    # 3) Build a per-sample float32 speech mask at the original (Demucs) sample rate.
    vocals = vocals_np
    mask = np.zeros(n_samples, dtype="float32")
    fade = int(0.020 * sr)  # 20 ms cosine cross-fade at boundaries (avoid clicks)
    fade_in = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade, dtype="float32")))
    fade_out = fade_in[::-1]
    for iv in intervals:
        s = max(0, int(iv["start"] * sr))
        e = min(n_samples, int(iv["end"] * sr))
        if e <= s:
            continue
        mask[s:e] = 1.0
        # Fade-in: e.g. ramp mask[s:s+fade] from 0→1 (but already set to 1.0;
        # overwrite the leading/trailing edges with the ramp shape).
        a = min(fade, e - s)
        mask[s:s + a] = fade_in[:a]
        mask[max(s, e - a):e] = fade_out[:a]

    speech = vocals * mask[:, None]
    non_speech = vocals * (1.0 - mask)[:, None]

    speech_path = job_dir / "speech_vocals.wav"
    non_speech_path = job_dir / "non_speech_vocals.wav"
    sf.write(str(speech_path), speech, sr, subtype="PCM_16")
    sf.write(str(non_speech_path), non_speech, sr, subtype="PCM_16")
    interval_list = [(float(iv["start"]), float(iv["end"])) for iv in intervals]
    return speech_path, non_speech_path, interval_list


def build(config: dict[str, Any]) -> HTDemucs:
    return HTDemucs(
        model_name=config.get("model", "htdemucs_ft"),
        device=config.get("device", "auto"),
        vocals_only=bool(config.get("vocals_only", False)),
    )
