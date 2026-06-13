"""ASR + diarization via ElevenLabs Scribe v2.

Scribe v2 returns one JSON document with word-level timestamps and a
`speaker_id` per word. We use it for BOTH the asr stage (this module) and
the diarize stage (`patmir.pipeline.stages.diarize.elevenlabs_scribe`,
which imports `ScribeV2Client` from here).

To keep cost at one API call per clip, the result is cached on disk in
`<job_dir>/scribe/result.json`. Diarize runs first (per `STAGE_ORDER`) and
triggers the upload; ASR runs later and just parses the cached JSON.

Endpoint: POST https://api.elevenlabs.io/v1/speech-to-text
Key request fields used:
  - model_id="scribe_v2"
  - diarize=true
  - timestamps_granularity="word"
  - num_speakers=<knob>  (optional; constrains clustering)
  - language_code=<iso>  (passes ctx.source_lang; defaults to "rus" for now)
  - tag_audio_events=false (we don't want [laughter] tokens in the transcript)

See `docs/architecture/decisions/2026-05-27-scribe-v2-for-asr-and-diarize.md`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Sentence, Transcript, Word
from patmir.pipeline.stages.asr._sentence import (
    group_into_sentences,
    is_non_lexical,
    write_transcript,
)

_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ScribeV2Client:
    """Shared API client. `fetch(ctx, audio_path)` returns the parsed JSON,
    making at most one HTTP call per job (cached to disk)."""

    def __init__(
        self,
        *,
        api_key_env: str = "ELEVENLABS_API_KEY",
        model_id: str = "scribe_v2",
        num_speakers: int = 0,
        diarization_threshold: float | None = None,
        timeout_seconds: float = 600.0,
        upload_sample_rate: int = 16000,
        upload_mono: bool = True,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"scribe_v2: env var {api_key_env!r} is not set. "
                f"Add ELEVENLABS_API_KEY=... to .env."
            )
        self.api_key = api_key
        self.model_id = model_id
        self.num_speakers = num_speakers
        self.diarization_threshold = diarization_threshold
        self.timeout = timeout_seconds
        # Scribe runs ~3x faster on a 16 kHz mono upload than on the 44.1 kHz
        # stereo Demucs vocal stem (the file is ~6x smaller and STT models work
        # at 16 kHz internally anyway). Word timestamps are in seconds, so the
        # downsample doesn't shift the timeline. Measured 2026-06-14: stereo
        # 30s slice ≈ 82–110s (2.8x realtime) vs 16k-mono ≈ 20s (0.68x). The
        # whole-file spike's fast Scribe number only held because its input
        # happened to be light; the streaming sections sent the heavy stem.
        self.upload_sample_rate = upload_sample_rate
        self.upload_mono = upload_mono

    def fetch(self, ctx: JobContext, audio_path: Path, *, language_hint: str | None = None) -> dict:
        cache = ctx.job_dir / "scribe" / "result.json"
        if cache.exists():
            return json.loads(cache.read_text())

        cache.parent.mkdir(exist_ok=True)
        data: dict[str, str] = {
            "model_id": self.model_id,
            "diarize": "true",
            "timestamps_granularity": "word",
            "tag_audio_events": "false",
        }
        if self.num_speakers and self.num_speakers > 0:
            data["num_speakers"] = str(self.num_speakers)
        if self.diarization_threshold is not None:
            data["diarization_threshold"] = str(self.diarization_threshold)
        # Scribe expects ISO-639-3; map common ISO-639-1 inputs.
        lang = language_hint or ctx.source_lang
        iso639_3 = _to_iso_639_3(lang)
        if iso639_3:
            data["language_code"] = iso639_3

        upload_path, cleanup = self._prepare_upload(audio_path)
        try:
            with open(upload_path, "rb") as f:
                files = {"file": (upload_path.name, f, "audio/wav")}
                resp = httpx.post(
                    _API_URL,
                    headers={"xi-api-key": self.api_key},
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
        finally:
            cleanup()
        if resp.status_code != 200:
            raise RuntimeError(
                f"scribe_v2: {resp.status_code} {resp.text[:500]}"
            )
        result = resp.json()
        cache.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    def _prepare_upload(self, audio_path: Path):
        """Transcode the upload to 16 kHz mono (the big Scribe speedup). Returns
        `(path, cleanup_fn)`; on any ffmpeg failure, uploads the original."""
        if not self.upload_sample_rate:
            return audio_path, lambda: None
        import subprocess
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        out = Path(tmp.name)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path),
                 "-ac", "1" if self.upload_mono else "2",
                 "-ar", str(self.upload_sample_rate),
                 "-c:a", "pcm_s16le", str(out)],
                check=True, capture_output=True,
            )
            return out, lambda: out.unlink(missing_ok=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            out.unlink(missing_ok=True)
            return audio_path, lambda: None


def _to_iso_639_3(code: str | None) -> str | None:
    """Map the common ISO-639-1 source-lang hints the project uses to ISO-639-3
    (what Scribe v2 expects). Falls through unchanged for 3-letter codes."""
    if not code:
        return None
    code = code.lower()
    mapping = {"ru": "rus", "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "hy": "hye"}
    if code in mapping:
        return mapping[code]
    if len(code) == 3:
        return code
    return None


class ScribeV2ASR:
    """ASR candidate. Pulls the Scribe v2 JSON from the shared cache and emits
    a Transcript with word-level timestamps + grouped sentences. Speaker
    tagging happens later in the runner's `apply_speaker_tags` step against
    the diarize output (which is also Scribe-derived in the standard
    Phase 2.x config)."""

    id = "elevenlabs_scribe_v2"

    def __init__(self, client: ScribeV2Client) -> None:
        self.client = client

    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript:
        ctx.progress("asr", 0.05, f"scribe_v2 ({self.client.model_id})")
        result = self.client.fetch(ctx, audio.path)
        language = (result.get("language_code") or ctx.source_lang or "ru")[:2]
        # Normalize "rus" → "ru" for downstream consumers that expect ISO-639-1.
        if language in {"rus", "eng", "spa", "fra", "deu", "hye"}:
            language = language[:2]
        ctx.source_lang = language
        ctx.progress("asr", 0.5, f"detected language={language}")

        words: list[Word] = []
        for w in result.get("words", []) or []:
            if w.get("type") != "word":
                continue
            text = (w.get("text") or "").strip()
            if not text:
                continue
            words.append(Word(
                text=text,
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", w.get("start", 0.0))),
            ))

        # Scribe doesn't emit prosodic segments; group on punctuation + gaps.
        raw_sentences = group_into_sentences(words)
        sentences: list[Sentence] = []
        for s in raw_sentences:
            if is_non_lexical(s.text):
                continue
            sentences.append(Sentence(
                idx=len(sentences),
                text=s.text,
                start=s.start,
                end=s.end,
                words=s.words,
            ))

        transcript = Transcript(language=language, words=words, sentences=sentences)
        write_transcript(ctx.job_dir, transcript)
        ctx.progress("asr", 1.0, "done")
        return transcript


def build(config: dict[str, Any]) -> ScribeV2ASR:
    client = ScribeV2Client(
        api_key_env=config.get("api_key_env", "ELEVENLABS_API_KEY"),
        model_id=config.get("model_id", "scribe_v2"),
        num_speakers=int(config.get("num_speakers", 0)),
        diarization_threshold=(
            float(config["diarization_threshold"])
            if config.get("diarization_threshold") is not None
            else None
        ),
        timeout_seconds=float(config.get("timeout_seconds", 600.0)),
        upload_sample_rate=int(config.get("upload_sample_rate", 16000)),
        upload_mono=bool(config.get("upload_mono", True)),
    )
    return ScribeV2ASR(client=client)
