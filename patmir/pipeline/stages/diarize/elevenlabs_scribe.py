"""Diarize candidate via ElevenLabs Scribe v2.

Same API call as the asr candidate next door — Scribe v2 returns words with
per-word `speaker_id` already attached. This stage parses those out into
`SpeakerTurn`s by collapsing consecutive same-speaker words. The shared
`ScribeV2Client` caches the response on disk so when the asr stage runs
later, it doesn't repeat the upload.

Limitations vs pyannote:
  - No overlap-interval detection. The ref_picker therefore won't skip
    overlapped windows. In practice Scribe handles overlap by attributing
    each word to its dominant speaker; the marginal regression for
    cartoon-style dialogue is small. If overlap-aware ref picking matters,
    keep pyannote and run Scribe ASR-only.

Returns speaker IDs in `SPEAKER_NN` format (matches pyannote's convention)
so the rest of the pipeline doesn't need to special-case Scribe.

See `docs/architecture/decisions/2026-05-27-scribe-v2-for-asr-and-diarize.md`.
"""
from __future__ import annotations

import json
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import Diarization, OverlapInterval, SeparatedAudio, SpeakerTurn
from patmir.pipeline.stages.asr.elevenlabs_scribe import ScribeV2Client


class ScribeV2Diarize:
    id = "elevenlabs_scribe_v2"

    def __init__(self, client: ScribeV2Client) -> None:
        self.client = client

    def diarize(self, separated: SeparatedAudio, ctx: JobContext) -> Diarization:
        ctx.progress("diarize", 0.05, f"scribe_v2 ({self.client.model_id})")
        result = self.client.fetch(ctx, separated.vocals)
        words = result.get("words") or []
        turns = _turns_from_words(words)

        diarize_dir = ctx.job_dir / "diarize"
        diarize_dir.mkdir(exist_ok=True)
        speakers = sorted({t.speaker_id for t in turns})
        (diarize_dir / "turns.json").write_text(json.dumps({
            "model": "scribe_v2",
            "device": "cloud",
            "speakers": speakers,
            "turns": [{"start": t.start, "end": t.end, "speaker_id": t.speaker_id} for t in turns],
            "overlaps": [],
        }, indent=2))

        ctx.progress("diarize", 1.0, f"{len(speakers)} speakers, {len(turns)} turns")
        return Diarization(
            turns=turns,
            overlaps=[],
            refs={},
            sample_rate=separated.sample_rate,
        )


def _turns_from_words(words: list[dict]) -> list[SpeakerTurn]:
    """Collapse consecutive same-speaker words into one SpeakerTurn.

    Spacing and audio_event entries are skipped — only "word" type entries
    carry speaker attribution.
    """
    turns: list[SpeakerTurn] = []
    cur_spk: str | None = None
    cur_start: float = 0.0
    cur_end: float = 0.0
    for w in words:
        if w.get("type") != "word":
            continue
        spk = _normalize_speaker_id(w.get("speaker_id") or "speaker_0")
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start))
        if spk != cur_spk:
            if cur_spk is not None and cur_end > cur_start:
                turns.append(SpeakerTurn(start=cur_start, end=cur_end, speaker_id=cur_spk))
            cur_spk = spk
            cur_start = start
        cur_end = end
    if cur_spk is not None and cur_end > cur_start:
        turns.append(SpeakerTurn(start=cur_start, end=cur_end, speaker_id=cur_spk))
    return turns


def _normalize_speaker_id(scribe_id: str) -> str:
    """`speaker_0` → `SPEAKER_00`. Matches pyannote's label format so the
    rest of the pipeline (refs.json, voice_cloning_summary.json, run artifact
    paths) reads consistently regardless of which diarizer was used."""
    if scribe_id.startswith("speaker_"):
        try:
            n = int(scribe_id.split("_", 1)[1])
            return f"SPEAKER_{n:02d}"
        except ValueError:
            pass
    return scribe_id.upper()


def build(config: dict[str, Any]) -> ScribeV2Diarize:
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
    )
    return ScribeV2Diarize(client=client)
