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

# Default preset voice pool for no-clone mode (`clone = false`). On `eleven_v3`
# any library voice speaks Armenian (language is model-driven, not voice-driven),
# so these are chosen purely for cartoon-character timbre variety — distinct
# ages plus the two `characters_animation` voices (Callum, Harry). Each diarized
# speaker is assigned a gender-matched voice (gender inferred from the speaker's
# ref-clip pitch) round-robin, and keeps it for the whole video. Override via
# `[[...]].voice_pool` in patmir.toml (entries may be plain ids or {id, gender}).
_DEFAULT_VOICE_POOL: list[dict[str, str]] = [
    {"id": "JBFqnCBsd6RMkjVDRZzb", "gender": "male"},    # George  — middle, warm storyteller
    {"id": "EXAVITQu4vr4xnSDxMaL", "gender": "female"},  # Sarah   — young
    {"id": "N2lVS1w4EtoT3dr4eOWO", "gender": "male"},    # Callum  — animation (husky trickster)
    {"id": "cgSgspJ2msm6clMCkdW9", "gender": "female"},  # Jessica — young, cute
    {"id": "IKne3meq5aSn9XLyUdCD", "gender": "male"},    # Charlie — young, energetic
    {"id": "FGY2WhTYpPnrIDTdsKH5", "gender": "female"},  # Laura   — young, quirky
    {"id": "SOYHLrjzK2X1ezoPC6cr", "gender": "male"},    # Harry   — animation (fierce/rough)
    {"id": "pqHfZKP75CvOlQylNhV4", "gender": "male"},    # Bill    — old, wise
]

# Median-F0 boundary (Hz) between male and female timbre. Adult males ~85–155,
# adult females ~165–255, children skew high → routed to the female sub-pool
# (a young/cute voice fits a kid acceptably).
_GENDER_F0_THRESHOLD_HZ = 165.0


class ElevenLabsTTS:
    id = "elevenlabs_multilingual_v2"
    # IVC clones from the audio file alone; the reference transcript is
    # ignored (`del ref_text` in `synthesize` below). The adapter reads this
    # to decide whether refs with empty `ref_text` are still usable.
    ref_text_optional = True

    def __init__(
        self,
        model_id: str = "eleven_v3",
        api_key_env: str = "ELEVENLABS_API_KEY",
        default_voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # "Rachel" — multilingual baseline
        voice_settings: dict[str, dict[str, float]] | None = None,
        timeout_seconds: float = 90.0,
        cleanup_voices: bool = True,
        # mp3_44100_128 works on every tier (Free → Pro). pcm_44100 is Pro-only.
        # We accept the per-call MP3 decode (~50 ms via ffmpeg) for tier
        # portability. Users on Pro+ can override via config to skip decode.
        output_format: str = "mp3_44100_128",
        # ISO language hint passed as `language_code` to the API. Only
        # `eleven_v3` accepts Armenian (`hy`); v2 / turbo_v2_5 / flash_v2_5
        # reject it. Earlier multilingual models silently fall back to
        # English-phoneme reading of Cyrillic Armenian, which sounds wrong.
        language_code: str | None = "hy",
        # IVC upload sample-rate normalization isn't needed — ElevenLabs
        # accepts arbitrary WAV/MP3. The ref clips we slice are 16 kHz mono
        # PCM, which is fine.
        #
        # clone=True (default) → Instant Voice Cloning from the source ref clip.
        # clone=False → no-clone "preset pool" mode: each speaker is assigned a
        # stable library voice from `voice_pool`. Avoids the intermittent
        # celebrity-clone block (HTTP 403 voice_access_denied) entirely and
        # removes the per-speaker IVC upload latency. The recommended mode for
        # near-realtime watch-along (ADR 2026-06-13-near-realtime-d3-spike).
        clone: bool = True,
        voice_pool: list[str] | list[dict[str, str]] | None = None,
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
        self.language_code = language_code
        self.clone = clone
        # Normalize pool entries to {id, gender, age} dicts. Plain-string
        # entries (legacy / untagged) become gender/age "any". The gender+age
        # tags are consumed by the speaker registry (Slice 2 #7), which profiles
        # each speaker and passes a matched `voice_id` explicitly; the candidate
        # itself only needs the id list for its standalone round-robin fallback.
        raw_pool = voice_pool if voice_pool else _DEFAULT_VOICE_POOL
        self.voice_pool: list[dict[str, str]] = [
            {"id": e, "gender": "any", "age": "any"} if isinstance(e, str)
            else {"id": e["id"], "gender": e.get("gender", "any"), "age": e.get("age", "any")}
            for e in raw_pool
        ]
        self._pool_ids: list[str] = [e["id"] for e in self.voice_pool]
        # Round-robin cursor for assigning pool voices to new speakers.
        self._pool_cursor = 0
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
        # Per-speaker call counters and the first ref WAV path seen for each
        # speaker — surfaced through `summarize()` so the adapter can write a
        # per-job voice_cloning_summary.json.
        self._speaker_lines: dict[str, int] = {}
        self._speaker_first_ref: dict[str, str] = {}
        self._ivc_error_by_speaker: dict[str, str] = {}
        # Count of sentences that errored out during synthesis (timeout, 5xx,
        # decode failure) and were replaced with a silent placeholder.
        self._silent_lines: int = 0
        if self.cleanup_voices:
            atexit.register(self._cleanup)

    # ─── preset-pool voice assignment (no-clone mode) ──────────────────────

    def _pool_voice_for_speaker(self, speaker_id: str) -> str:
        """Map a speaker to a stable pool voice (round-robin, cached).

        The same speaker_id always returns the same voice for the life of this
        candidate, so a character keeps one voice across the whole video. New
        speakers take the next pool slot, wrapping around if there are more
        speakers than pool voices.
        """
        existing = self._speaker_voice.get(speaker_id)
        if existing is not None:
            return existing
        voice_id = self._pool_ids[self._pool_cursor % len(self._pool_ids)]
        self._pool_cursor += 1
        self._speaker_voice[speaker_id] = voice_id
        return voice_id

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

    # ─── observability ─────────────────────────────────────────────────────

    def summarize(self) -> dict:
        """Return a per-job summary of voice-cloning decisions.

        The adapter calls this at the end of synthesis and writes the result
        to `<job_dir>/tts/voice_cloning_summary.json`. Speakers fall into:

        - `cloned`            — IVC succeeded; uses `voice_id`.
        - `fallback_default`  — IVC upload failed; uses `default_voice_id`.
        - `no_ref_provided`   — adapter found no ref for this speaker; default voice.
        - `no_speaker`        — `_no_ref` bucket: sentences with no speaker_id.
        """
        speakers: dict[str, dict] = {}
        for speaker_id, lines in self._speaker_lines.items():
            if not self.clone:
                # Preset-pool mode: every speaker has a pool voice assignment
                # (or the default if it was a no-ref/no-speaker line).
                speakers[speaker_id] = {
                    "status": "pool",
                    "voice_id": self._speaker_voice.get(speaker_id, self.default_voice_id),
                    "lines": lines,
                }
                continue
            if speaker_id == "_no_ref":
                speakers["_no_speaker"] = {
                    "status": "no_speaker",
                    "voice_id": self.default_voice_id,
                    "lines": lines,
                }
                continue
            if speaker_id in self._speaker_voice:
                speakers[speaker_id] = {
                    "status": "cloned",
                    "voice_id": self._speaker_voice[speaker_id],
                    "ref_audio": self._speaker_first_ref.get(speaker_id),
                    "lines": lines,
                }
            elif speaker_id in self._ivc_failed_speakers:
                speakers[speaker_id] = {
                    "status": "fallback_default",
                    "voice_id": self.default_voice_id,
                    "ref_audio": self._speaker_first_ref.get(speaker_id),
                    "reason": self._ivc_error_by_speaker.get(speaker_id, "unknown"),
                    "lines": lines,
                }
            else:
                # Sentences with no ref WAV looked up by the adapter.
                speakers[speaker_id] = {
                    "status": "no_ref_provided",
                    "voice_id": self.default_voice_id,
                    "lines": lines,
                }
        return {
            "candidate_id": self.id,
            "model_id": self.model_id,
            "default_voice_id": self.default_voice_id,
            "speakers": speakers,
            "silent_fallback_lines": self._silent_lines,
        }

    # ─── synth ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        *,
        ref_audio: Path | str | None = None,
        ref_text: str | None = None,
        guidance_scale: float | None = None,
        emotion: str | None = None,
        speaker_id: str | None = None,
        voice_id: str | None = None,
    ) -> bytes:
        """Return WAV bytes. `emotion` selects the voice_settings preset.

        Voice selection depends on mode:
          - clone=True  → `ref_audio` is IVC-cloned (uploads on first use).
          - clone=False → preset-pool: `speaker_id` is mapped to a stable pool
            voice. An explicit `voice_id` (e.g. from the speaker registry)
            overrides pool assignment in either mode.

        `ref_text` and `guidance_scale` accepted for adapter uniformity but
        unused — ElevenLabs has no ref-transcript input and `guidance_scale`
        is an OmniVoice concept.
        """
        del ref_text, guidance_scale

        # Identify the speaker: explicit kwarg (registry) wins; else infer from
        # the ref-clip path (diarize/refs/<speaker>/<emotion>.wav); else no-ref.
        if speaker_id is None:
            speaker_id = Path(str(ref_audio)).parent.name if ref_audio is not None else "_no_ref"
        self._speaker_lines[speaker_id] = self._speaker_lines.get(speaker_id, 0) + 1

        if not self.clone:
            # ── No-clone preset-pool mode ──────────────────────────────────
            # No upload, no captcha path. Stable voice per speaker.
            voice_id = voice_id or self._pool_voice_for_speaker(speaker_id)
        else:
            # ── IVC clone mode (original behavior) ─────────────────────────
            voice_id = voice_id or self.default_voice_id
            if ref_audio is not None:
                self._speaker_first_ref.setdefault(speaker_id, str(ref_audio))
                if speaker_id in self._ivc_failed_speakers:
                    # Already known to fail (tier issue, etc.) — skip retry,
                    # quietly use the default voice for this speaker's lines.
                    pass
                else:
                    try:
                        voice_id = self._voice_id_for_ref(ref_audio)
                    except Exception as exc:
                        self._ivc_failed_speakers.add(speaker_id)
                        self._ivc_error_by_speaker[speaker_id] = f"{type(exc).__name__}: {exc}"
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
        if self.language_code:
            payload["language_code"] = self.language_code
        try:
            resp = self._client.post(
                f"{_API_BASE}/text-to-speech/{voice_id}",
                headers={**self._headers, "Content-Type": "application/json"},
                params={"output_format": self.output_format},
                json=payload,
            )
        except httpx.HTTPError as exc:
            # A single slow request (read timeout / connection reset) used to
            # kill the entire pipeline mid-TTS — losing 20+ already-synthesized
            # sentences and the cached IVC voice_ids. Treat it as a per-sentence
            # failure: emit silence for this line, record it in `summarize()`,
            # let the pipeline finish so the rest of the job is recoverable.
            print(
                f"elevenlabs: synth request errored for text={text[:60]!r} "
                f"({type(exc).__name__}: {exc}); returning silent WAV for this line."
            )
            self._silent_lines += 1
            return _silent_wav(seconds=1.0, sample_rate=44100)
        if resp.status_code != 200:
            # Cloned-voice rejections (captcha / celebrity match / ToS) fail
            # with HTTP 403 + code=voice_access_denied or detected_captcha_voice.
            # These only ever fire for IVC-cloned voices, never for the
            # default library voice. Retry once with the default voice so
            # the dub isn't silenced for an entire video just because the
            # clone tripped the captcha (e.g. real-celebrity ref clips).
            is_voice_block = (
                resp.status_code == 403
                and voice_id != self.default_voice_id
                and ("voice_access_denied" in resp.text or "detected_captcha" in resp.text)
            )
            if is_voice_block:
                if speaker_id not in self._ivc_failed_speakers:
                    self._ivc_failed_speakers.add(speaker_id)
                    self._ivc_error_by_speaker[speaker_id] = (
                        f"voice_access_denied (HTTP 403, captcha/ToS) — clone "
                        f"of speaker {speaker_id} rejected; using default voice."
                    )
                    print(
                        f"elevenlabs: cloned voice {voice_id} for {speaker_id} "
                        f"rejected by captcha/ToS; retrying with default voice "
                        f"{self.default_voice_id} for this and subsequent lines."
                    )
                try:
                    retry = self._client.post(
                        f"{_API_BASE}/text-to-speech/{self.default_voice_id}",
                        headers={**self._headers, "Content-Type": "application/json"},
                        params={"output_format": self.output_format},
                        json=payload,
                    )
                except httpx.HTTPError as exc:
                    print(
                        f"elevenlabs: retry with default voice errored "
                        f"({type(exc).__name__}: {exc}); returning silent WAV."
                    )
                    self._silent_lines += 1
                    return _silent_wav(seconds=1.0, sample_rate=44100)
                if retry.status_code == 200:
                    resp = retry
                    voice_id = self.default_voice_id
                else:
                    print(
                        f"elevenlabs: retry with default voice also failed "
                        f"HTTP {retry.status_code}: {retry.text[:200]}; "
                        f"returning silent WAV."
                    )
                    self._silent_lines += 1
                    return _silent_wav(seconds=1.0, sample_rate=44100)
            else:
                print(
                    f"elevenlabs: synth HTTP {resp.status_code} for text={text[:60]!r} "
                    f"(voice_id={voice_id}): {resp.text[:200]}; returning silent WAV."
                )
                self._silent_lines += 1
                return _silent_wav(seconds=1.0, sample_rate=44100)

        audio_bytes = resp.content
        if self.output_format.startswith("mp3"):
            # Decode MP3 → PCM-16 mono 44.1k via ffmpeg subprocess (already
            # a hard project dependency for ingest + mux).
            try:
                return _mp3_to_wav(audio_bytes)
            except RuntimeError as exc:
                ctype = resp.headers.get("content-type", "?")
                head = audio_bytes[:80]
                print(
                    f"elevenlabs: MP3 decode failed for text={text[:60]!r} "
                    f"(voice_id={voice_id}, content-type={ctype}, "
                    f"bytes={len(audio_bytes)}, head={head!r}). "
                    f"Returning silent WAV. ({exc})"
                )
                return _silent_wav(seconds=1.0, sample_rate=44100)
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


def _silent_wav(seconds: float, sample_rate: int = 44100) -> bytes:
    """Build a silent PCM-16 mono WAV. Used as a fallback when ElevenLabs
    returns a non-decodable payload (rare — but seen when synthesizing
    long letter-runs that the model refuses to vocalize)."""
    n_samples = max(1, int(seconds * sample_rate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


def _mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Decode MP3 bytes → WAV bytes (PCM-16 mono 44.1k) via ffmpeg subprocess.

    Decodes to RAW PCM (`s16le`) and wraps it with Python's `wave` writer
    rather than asking ffmpeg for `-f wav pipe:1`. ffmpeg can't seek back on a
    pipe to fill the RIFF/data size field, so its piped WAV header carries a
    placeholder frame count — `wave.open(...).getnframes()` then reports a
    bogus multi-hour duration, which the TTS adapter stores as `wav_duration`
    and mux uses to time-stretch (→ wildly wrong stretch). Wrapping the raw
    PCM ourselves guarantees a correct header.
    """
    sr = 44100
    proc = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-f", "mp3", "-i", "pipe:0",
            "-ac", "1", "-ar", str(sr),
            "-f", "s16le", "-c:a", "pcm_s16le",
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
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(proc.stdout)
    return buf.getvalue()


def build(config: dict[str, Any]) -> ElevenLabsTTS:
    return ElevenLabsTTS(
        model_id=config.get("model_id", "eleven_v3"),
        api_key_env=config.get("api_key_env", "ELEVENLABS_API_KEY"),
        default_voice_id=config.get("default_voice_id", "21m00Tcm4TlvDq8ikWAM"),
        voice_settings=config.get("voice_settings"),
        timeout_seconds=float(config.get("timeout_seconds", 90.0)),
        cleanup_voices=bool(config.get("cleanup_voices", True)),
        output_format=config.get("output_format", "mp3_44100_128"),
        language_code=config.get("language_code", "hy"),
        clone=bool(config.get("clone", True)),
        voice_pool=config.get("voice_pool"),
    )
