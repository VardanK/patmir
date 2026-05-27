"""ElevenLabs Multilingual TTS — cloud-side TTS escalation.

The cloud-TTS opt-in per the local-first policy after OmniVoice MLX
emotion-routing still ear-tested flat on the masha clip. ElevenLabs
exposes:

- An Armenian-capable multilingual TTS (`eleven_multilingual_v2`).
- Per-emotion control via `voice_settings` (stability / similarity_boost /
  style) — lower stability + higher style = more emotional variation.
- Instant Voice Cloning (IVC) — upload a 3-10 s reference clip → get a
  voice_id → use that voice for cloned synthesis. We already have
  per-(speaker, emotion) ref clips sliced into
  `diarize/refs/<speaker>/<emotion>.wav` — we feed them to IVC so each
  character keeps their own voice (preserves Phase 2 multi-voice
  behavior; doesn't regress to single-voice).

The candidate caches the (ref_audio path → voice_id) mapping per job and
cleans up uploaded voices on shutdown (ElevenLabs has voice-slot caps
that fill up fast otherwise).

See ADR `docs/architecture/decisions/2026-05-26-elevenlabs-tts-cloud-opt-in.md`.

Reads API key from `ELEVENLABS_API_KEY`. Uses `httpx` (already in venv).
"""
from __future__ import annotations

import atexit
import io
import os
import subprocess
import wave
from pathlib import Path
from typing import Any

import httpx

_API_BASE = "https://api.elevenlabs.io/v1"

# Default per-emotion ElevenLabs voice_settings.
# stability: lower → more emotional variation. higher → more consistent.
# style:     higher → more expressive / dramatic.
# similarity_boost: how closely to match the cloned voice; keep stable.
_DEFAULT_VOICE_SETTINGS: dict[str, dict[str, float]] = {
    "neutral": {"stability": 0.55, "similarity_boost": 0.80, "style": 0.20},
    "happy":   {"stability": 0.35, "similarity_boost": 0.80, "style": 0.55},
    "sad":     {"stability": 0.55, "similarity_boost": 0.80, "style": 0.40},
    "angry":   {"stability": 0.25, "similarity_boost": 0.80, "style": 0.70},
    "fearful": {"stability": 0.30, "similarity_boost": 0.80, "style": 0.60},
}


class ElevenLabsTTS:
    id = "elevenlabs_multilingual_v2"

    def __init__(
        self,
        model_id: str = "eleven_multilingual_v2",
        api_key_env: str = "ELEVENLABS_API_KEY",
        default_voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # "Rachel" — multilingual baseline
        voice_settings: dict[str, dict[str, float]] | None = None,
        timeout_seconds: float = 90.0,
        cleanup_voices: bool = True,
        # mp3_44100_128 works on every tier (Free → Pro). pcm_44100 is Pro-only.
        # We accept the per-call MP3 decode (~50 ms via ffmpeg) for tier
        # portability. Users on Pro+ can override via config to skip decode.
        output_format: str = "mp3_44100_128",
        # IVC upload sample-rate normalization isn't needed — ElevenLabs
        # accepts arbitrary WAV/MP3. The ref clips we slice are 16 kHz mono
        # PCM, which is fine.
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"elevenlabs tts: env var {api_key_env!r} is not set. "
                f"Add ELEVENLABS_API_KEY=... to .env, or `export {api_key_env}=...`."
            )
        self.api_key = api_key
        self.model_id = model_id
        self.default_voice_id = default_voice_id
        self.voice_settings: dict[str, dict[str, float]] = {
            **_DEFAULT_VOICE_SETTINGS,
            **(voice_settings or {}),
        }
        self.timeout = timeout_seconds
        self.cleanup_voices = cleanup_voices
        self.output_format = output_format
        self._client = httpx.Client(timeout=timeout_seconds)
        self._headers = {"xi-api-key": self.api_key}
        # Speakers whose IVC upload has failed once — skip retry to avoid
        # spamming the API. The default voice is used instead and a single
        # warning is logged the first time.
        self._ivc_failed_speakers: set[str] = set()
        # ref_audio path str → voice_id. Multiple emotions for the same speaker
        # use the same voice_id; emotion is reflected via voice_settings, not
        # by uploading separate IVC voices.
        self._voice_cache: dict[str, str] = {}
        # speaker_id (extracted from ref path) → voice_id, so different
        # emotion clips for the same speaker reuse the same upload.
        self._speaker_voice: dict[str, str] = {}
        self._uploaded_voice_ids: list[str] = []
        if self.cleanup_voices:
            atexit.register(self._cleanup)

    # ─── IVC voice management ──────────────────────────────────────────────

    def _voice_id_for_ref(self, ref_audio: Path | str) -> str:
        path = str(ref_audio)
        if path in self._voice_cache:
            return self._voice_cache[path]

        # Speaker is the parent directory name in our ref layout —
        # `diarize/refs/SPEAKER_03/angry.wav` → "SPEAKER_03". All emotion
        # clips for the same speaker reuse one IVC upload (the calmest /
        # longest one, in practice "neutral" if present, else the first one
        # we encounter for that speaker).
        speaker_id = Path(path).parent.name
        if speaker_id in self._speaker_voice:
            voice_id = self._speaker_voice[speaker_id]
            self._voice_cache[path] = voice_id
            return voice_id

        # Upload as Instant Voice Cloning sample.
        with open(path, "rb") as f:
            files = {"files": (Path(path).name, f, "audio/wav")}
            data = {
                "name": f"patmir-{speaker_id}",
                "description": f"Patmir IVC clone — {speaker_id}",
            }
            resp = self._client.post(
                f"{_API_BASE}/voices/add",
                headers=self._headers,
                data=data,
                files=files,
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"elevenlabs ivc upload failed for {speaker_id} ({resp.status_code}): {resp.text[:500]}"
            )
        voice_id = resp.json().get("voice_id")
        if not voice_id:
            raise RuntimeError(f"elevenlabs ivc returned no voice_id: {resp.text[:500]}")

        self._speaker_voice[speaker_id] = voice_id
        self._voice_cache[path] = voice_id
        self._uploaded_voice_ids.append(voice_id)
        return voice_id

    def _cleanup(self) -> None:
        """Delete every IVC voice this job uploaded — free slots for next run."""
        for voice_id in list(self._uploaded_voice_ids):
            try:
                self._client.delete(
                    f"{_API_BASE}/voices/{voice_id}",
                    headers=self._headers,
                    timeout=10.0,
                )
            except Exception:
                # Best-effort cleanup; don't raise out of atexit.
                pass

    # ─── synth ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        *,
        ref_audio: Path | str | None = None,
        ref_text: str | None = None,
        guidance_scale: float | None = None,
        emotion: str | None = None,
    ) -> bytes:
        """Return WAV bytes. `ref_audio` selects an IVC-cloned voice;
        `emotion` selects the voice_settings preset.

        `ref_text` and `guidance_scale` accepted for adapter uniformity but
        unused — ElevenLabs has no ref-transcript input and `guidance_scale`
        is an OmniVoice concept.
        """
        del ref_text, guidance_scale

        voice_id = self.default_voice_id
        if ref_audio is not None:
            speaker_id = Path(str(ref_audio)).parent.name
            if speaker_id in self._ivc_failed_speakers:
                # Already known to fail (tier issue, etc.) — skip retry,
                # quietly use the default voice for this speaker's lines.
                pass
            else:
                try:
                    voice_id = self._voice_id_for_ref(ref_audio)
                except Exception as exc:
                    self._ivc_failed_speakers.add(speaker_id)
                    print(
                        f"elevenlabs: IVC upload failed for {speaker_id}; "
                        f"falling back to default voice {self.default_voice_id} for this speaker. "
                        f"({type(exc).__name__}: {exc})"
                    )

        settings = self.voice_settings.get(
            emotion or "neutral",
            self.voice_settings["neutral"],
        )

        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": settings["stability"],
                "similarity_boost": settings["similarity_boost"],
                "style": settings.get("style", 0.0),
                "use_speaker_boost": True,
            },
        }
        resp = self._client.post(
            f"{_API_BASE}/text-to-speech/{voice_id}",
            headers={**self._headers, "Content-Type": "application/json"},
            params={"output_format": self.output_format},
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"elevenlabs synth failed ({resp.status_code}, voice_id={voice_id}): "
                f"{resp.text[:500]}"
            )

        audio_bytes = resp.content
        if self.output_format.startswith("mp3"):
            # Decode MP3 → PCM-16 mono 44.1k via ffmpeg subprocess (already
            # a hard project dependency for ingest + mux).
            return _mp3_to_wav(audio_bytes)
        elif self.output_format.startswith("pcm"):
            # `pcm_<sr>` — raw PCM-16 mono at that sample rate.
            sr = int(self.output_format.split("_", 1)[1])
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_bytes)
            return buf.getvalue()
        else:
            # Assume WAV; pass through.
            return audio_bytes


def _mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Decode MP3 bytes → WAV bytes (PCM-16 mono 44.1k) via ffmpeg subprocess."""
    proc = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-f", "mp3", "-i", "pipe:0",
            "-ac", "1", "-ar", "44100",
            "-f", "wav", "-c:a", "pcm_s16le",
            "pipe:1",
        ],
        input=mp3_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"elevenlabs: ffmpeg MP3→WAV decode failed: {proc.stderr.decode('utf-8', errors='replace')[:500]}"
        )
    return proc.stdout


def build(config: dict[str, Any]) -> ElevenLabsTTS:
    return ElevenLabsTTS(
        model_id=config.get("model_id", "eleven_multilingual_v2"),
        api_key_env=config.get("api_key_env", "ELEVENLABS_API_KEY"),
        default_voice_id=config.get("default_voice_id", "21m00Tcm4TlvDq8ikWAM"),
        voice_settings=config.get("voice_settings"),
        timeout_seconds=float(config.get("timeout_seconds", 90.0)),
        cleanup_voices=bool(config.get("cleanup_voices", True)),
        output_format=config.get("output_format", "mp3_44100_128"),
    )
