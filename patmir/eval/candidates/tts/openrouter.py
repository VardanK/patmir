"""Cloud TTS via OpenRouter — frontier TTS models behind one API key.

OpenRouter (openrouter.ai) added speech models in late 2025 — exposed via
the OpenAI-compatible `/v1/audio/speech` endpoint. Same auth key as the
translate-side OpenRouter integration; no separate vendor account.

Default model: `google/gemini-3.1-flash-tts-preview`. Picked because
Google's Gemini TTS supports **inline audio tags** — short bracketed
directives like `[happy]`, `[sad]`, `[angry]`, `[whispers]`, `[shouting]`
that the model interprets as prosody / emotion hints for the following
text. We inject these per sentence from `s.emotion` (mapped from
emotion2vec's per-turn predictions), giving us explicit cloud-side
emotion control without needing voice cloning or per-emotion settings.

Other registered candidates (see `patmir.toml`):

- `openai/gpt-4o-mini-tts-2025-12-15` — cheapest ($0.60 / 1M chars), no
  audio tags but 6 named voices.
- `mistralai/voxtral-mini-tts-2603` — has zero-shot voice cloning.
- `hexgrad/kokoro-82m` — cheapest of all ($0.62 / 1M chars), Apache-2.0
  open weights routed through OpenRouter.

See ADR `docs/architecture/decisions/2026-05-26-openrouter-tts-cloud-opt-in.md`.
Reads API key from `OPENROUTER_API_KEY`. Uses `httpx` (already in venv).
"""
from __future__ import annotations

import io
import os
import subprocess
import wave
from pathlib import Path
from typing import Any

import httpx

_API_BASE = "https://openrouter.ai/api/v1"

# Per-emotion inline audio tag, prepended to the sentence text before
# synthesis. Gemini's TTS reads these as prosody directives. Empty string
# = no tag (neutral pass-through).
_DEFAULT_EMOTION_TAGS: dict[str, str] = {
    "neutral": "",
    "happy":   "[happy]",
    "sad":     "[sad]",
    "angry":   "[angry]",
    "fearful": "[whispers]",
}


class OpenRouterTTS:
    id = "openrouter_tts"

    def __init__(
        self,
        model: str = "google/gemini-3.1-flash-tts-preview",
        api_key_env: str = "OPENROUTER_API_KEY",
        voice: str = "Kore",  # default when no ref_audio (or when voices pool is empty)
        voices: list[str] | None = None,
        # Rotation pool — speakers are assigned voices in encounter order. If
        # `voices` is empty/None, every speaker uses the single `voice`
        # (regression to single-voice on the cloud side). Gemini's documented
        # voice names include: Kore, Puck, Aoede, Charon, Fenrir, Leda, Orus,
        # Zephyr, Erinome, Algenib, Algieba, Rasalgethi, etc.
        voice_map: dict[str, str] | None = None,
        # Explicit speaker_id → voice override. Wins over the rotation pool.
        response_format: str = "mp3",   # also valid: "pcm", "wav" on some models
        speed: float = 1.0,
        emotion_tags: dict[str, str] | None = None,
        timeout_seconds: float = 90.0,
        base_url: str = _API_BASE,
        referer: str = "https://github.com/patmir-dub",
        title: str = "Patmir Armenian Dubbing",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"openrouter tts: env var {api_key_env!r} is not set. "
                f"Sign up at openrouter.ai, create a key, and "
                f"`export {api_key_env}=sk-or-...` (or add it to .env)."
            )
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.voices: list[str] = list(voices or [])
        self.voice_map: dict[str, str] = dict(voice_map or {})
        self._speaker_voice_cache: dict[str, str] = {}
        self.response_format = response_format
        self.speed = float(speed)
        self.emotion_tags: dict[str, str] = {**_DEFAULT_EMOTION_TAGS, **(emotion_tags or {})}
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_seconds)
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": referer,
            "X-Title": title,
        }

    def synthesize(
        self,
        text: str,
        *,
        ref_audio: Path | str | None = None,
        ref_text: str | None = None,
        guidance_scale: float | None = None,
        emotion: str | None = None,
    ) -> bytes:
        """Return WAV bytes. `emotion` triggers an inline audio tag prefix
        (e.g. `[happy]`); other ref/guidance kwargs accepted for adapter
        uniformity and ignored.
        """
        del ref_text, guidance_scale

        tag = self.emotion_tags.get(emotion or "neutral", "")
        prompt = f"{tag} {text}".strip() if tag else text

        # Per-speaker voice rotation. Speaker id is extracted from the ref
        # clip path (`diarize/refs/SPEAKER_xx/<emo>.wav` → "SPEAKER_xx").
        # Once a speaker is bound to a voice, it stays bound for the rest
        # of this job — consistent character voice across emotions.
        chosen_voice = self.voice
        if ref_audio is not None:
            speaker_id = Path(str(ref_audio)).parent.name
            if speaker_id in self.voice_map:
                chosen_voice = self.voice_map[speaker_id]
            elif speaker_id in self._speaker_voice_cache:
                chosen_voice = self._speaker_voice_cache[speaker_id]
            elif self.voices:
                chosen_voice = self.voices[len(self._speaker_voice_cache) % len(self.voices)]
                self._speaker_voice_cache[speaker_id] = chosen_voice

        payload: dict[str, Any] = {
            "model": self.model,
            "voice": chosen_voice,
            "input": prompt,
            "response_format": self.response_format,
        }
        if self.speed != 1.0:
            payload["speed"] = self.speed

        resp = self._client.post(
            f"{self.base_url}/audio/speech",
            headers=self._headers,
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"openrouter tts: HTTP {resp.status_code} from {self.model}: "
                f"{resp.text[:500]}"
            )

        audio_bytes = resp.content
        fmt = self.response_format.lower()
        if fmt == "wav":
            return audio_bytes
        if fmt == "mp3":
            return _mp3_to_wav(audio_bytes)
        if fmt.startswith("pcm"):
            # `pcm_<sr>` (OpenAI-style) or raw "pcm" (24 kHz default for
            # gemini per OpenRouter docs). When the suffix is missing,
            # default to 24 kHz.
            sr = 24_000
            if "_" in fmt:
                try:
                    sr = int(fmt.split("_", 1)[1])
                except ValueError:
                    sr = 24_000
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_bytes)
            return buf.getvalue()
        # Unknown — try ffmpeg autodetect as a last resort.
        return _decode_via_ffmpeg(audio_bytes)


def _mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Decode MP3 → WAV (PCM-16 mono 44.1k) via ffmpeg subprocess."""
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
            f"openrouter tts: ffmpeg MP3→WAV decode failed: "
            f"{proc.stderr.decode('utf-8', errors='replace')[:500]}"
        )
    return proc.stdout


def _decode_via_ffmpeg(audio_bytes: bytes) -> bytes:
    """Autodetect-and-decode fallback for unexpected response formats."""
    proc = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-i", "pipe:0",
            "-ac", "1", "-ar", "44100",
            "-f", "wav", "-c:a", "pcm_s16le",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"openrouter tts: ffmpeg autodetect decode failed: "
            f"{proc.stderr.decode('utf-8', errors='replace')[:500]}"
        )
    return proc.stdout


def build(config: dict[str, Any]) -> OpenRouterTTS:
    return OpenRouterTTS(
        model=config.get("model", "google/gemini-3.1-flash-tts-preview"),
        api_key_env=config.get("api_key_env", "OPENROUTER_API_KEY"),
        voice=config.get("voice", "Kore"),
        voices=config.get("voices"),
        voice_map=config.get("voice_map"),
        response_format=config.get("response_format", "mp3"),
        speed=float(config.get("speed", 1.0)),
        emotion_tags=config.get("emotion_tags"),
        timeout_seconds=float(config.get("timeout_seconds", 90.0)),
        base_url=config.get("base_url", _API_BASE),
        referer=config.get("referer", "https://github.com/patmir-dub"),
        title=config.get("title", "Patmir Armenian Dubbing"),
    )
