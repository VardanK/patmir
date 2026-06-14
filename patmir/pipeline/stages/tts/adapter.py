"""TTS stage adapter.

Wraps a Phase 0 TTS candidate (`synthesize(text) -> bytes`) with the typed
`TTSStage` protocol. Writes per-sentence WAV files to `<job_dir>/tts/<idx>.wav`
and a `synthesized.json` index for resume + UI.

Reads each WAV's sample rate and duration from the file header rather than
decoding the audio — Piper emits PCM WAV that wave.open can parse directly.

Per-speaker voice cloning: if `ctx.extra["diarization"]` is set (Phase 2),
each sentence's `speaker_id` is looked up in `diarization.refs`; the matching
`SpeakerRef.ref_audio` / `ref_text` are passed to the inner candidate's
`synthesize` call when the candidate accepts those kwargs. Candidates that
don't (e.g. Piper with a fixed voice) fall through to the bare call.
"""
from __future__ import annotations

import importlib
import inspect
import io
import json
import wave
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    Diarization,
    SynthesizedAudio,
    SynthesizedSentence,
    TranslatedTranscript,
)


class SentenceTTS:
    id = "tts_adapter"

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        try:
            sig = inspect.signature(inner.synthesize)
            params = sig.parameters
            self._supports_ref = "ref_audio" in params and "ref_text" in params
            self._supports_guidance = "guidance_scale" in params
            self._supports_emotion = "emotion" in params
            # Pool-voice candidates (ElevenLabs no-clone mode) map a stable
            # voice per character; they need the speaker_id directly rather
            # than inferring it from the ref-clip path. Clone candidates
            # (OmniVoice) don't have this param and the probe skips it.
            self._supports_speaker = "speaker_id" in params
            # Streaming mode: the speaker registry owns voice assignment
            # (gender/age-matched, consistent across sections), so the runner
            # passes an explicit voice_id per speaker via
            # `ctx.extra["voice_assignments"]`. Candidates that accept a
            # `voice_id` kwarg (ElevenLabs pool mode) use it; others skip it.
            self._supports_voice_id = "voice_id" in params
        except (TypeError, ValueError):
            self._supports_ref = False
            self._supports_guidance = False
            self._supports_emotion = False
            self._supports_speaker = False
            self._supports_voice_id = False
        # IVC-style candidates (ElevenLabs) clone from audio alone and don't
        # need the reference transcript. They set `ref_text_optional = True`
        # so we treat refs with empty `ref_text` as usable for them.
        # Zero-shot candidates that require the transcript (OmniVoice, Piper)
        # leave the default and still gate on a non-empty ref_text.
        self._ref_text_optional = bool(getattr(inner, "ref_text_optional", False))

    def _lookup_ref(self, diarization: Diarization | None, speaker_id: str | None, emotion: str | None):
        """Fallback chain (Phase 2.x): preserve speaker identity first,
        emotion match second.

          1. (speaker, emotion)             — exact match
          2. (speaker, "neutral")           — same speaker, neutral emotion
          3. any (speaker, *)               — same speaker, any emotion clip
                                              (prefer the longest by duration)
          4. None → caller falls back to the global default (Piper)

        Step 3 is what keeps us from regressing vs Phase 2's "one ref per
        speaker": when a speaker has only one emotion bucket's worth of
        clean clips, every sentence in any other emotion would otherwise
        fall through to a different character's voice (the global Piper
        fallback). Better to use the same-speaker clip in the "wrong"
        emotion than to swap voices mid-character.
        """
        if diarization is None or not speaker_id:
            return None

        def _is_usable(r) -> bool:
            if r is None:
                return False
            if self._ref_text_optional:
                return True
            return r.ref_text.strip() != ""

        emo = emotion or "neutral"
        ref = diarization.refs.get((speaker_id, emo))
        if _is_usable(ref):
            return ref
        if emo != "neutral":
            ref = diarization.refs.get((speaker_id, "neutral"))
            if _is_usable(ref):
                return ref
        # Any ref for this speaker — pick the longest source window for stability.
        same_speaker = [
            r for (spk, _bin), r in diarization.refs.items()
            if spk == speaker_id and _is_usable(r)
        ]
        if same_speaker:
            return max(same_speaker, key=lambda r: r.source_end - r.source_start)
        return None

    def synthesize(self, translated: TranslatedTranscript, ctx: JobContext) -> SynthesizedAudio:
        out_dir = ctx.job_dir / "tts"
        out_dir.mkdir(exist_ok=True)
        sentences: list[SynthesizedSentence] = []
        sample_rate = 0

        diar = ctx.extra.get("diarization")
        diarization: Diarization | None = diar if isinstance(diar, Diarization) else None
        voice_assignments = ctx.extra.get("voice_assignments") or {}

        # Build each sentence's synth kwargs up front (cheap, no model calls),
        # so the actual synthesize() calls — the slow, network-bound part for a
        # cloud TTS — can be dispatched concurrently below.
        jobs: list[tuple[Any, dict[str, Any]]] = []
        for s in translated.sentences:
            if not s.target_text.strip():
                continue
            ref = self._lookup_ref(diarization, s.speaker_id, s.emotion) if self._supports_ref else None
            ref_audio = ref.ref_audio if ref is not None else None
            ref_text = ref.ref_text if ref is not None else None

            kwargs: dict[str, Any] = {}
            if self._supports_ref:
                kwargs["ref_audio"] = ref_audio
                kwargs["ref_text"] = ref_text
            if self._supports_guidance and s.emotion:
                gs = ctx.options.emotion_guidance.get(s.emotion)
                if gs is not None:
                    kwargs["guidance_scale"] = gs
            if self._supports_emotion:
                # ElevenLabs (and any other candidate that accepts an
                # explicit `emotion` kwarg) drives its own per-emotion knobs
                # off this. OmniVoice/Piper don't have this param and the
                # block is skipped via the signature probe.
                kwargs["emotion"] = s.emotion
            if self._supports_speaker:
                kwargs["speaker_id"] = s.speaker_id
            if self._supports_voice_id and s.speaker_id in voice_assignments:
                kwargs["voice_id"] = voice_assignments[s.speaker_id]
            jobs.append((s, kwargs))

        total = len(jobs) or 1
        # Concurrency is the candidate's call: ElevenLabs pool mode sets 3
        # (network-bound, tier cap); OmniVoice/MLX leaves it at 1 (single GPU
        # stream). Output order is preserved regardless.
        concurrency = max(1, int(getattr(self.inner, "tts_concurrency", 1)))
        results: list[bytes | None] = [None] * len(jobs)

        def _synth(idx: int) -> None:
            s, kwargs = jobs[idx]
            results[idx] = self.inner.synthesize(s.target_text, **kwargs)

        if concurrency > 1 and len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(_synth, i): i for i in range(len(jobs))}
                done = 0
                for fut in as_completed(futs):
                    fut.result()  # re-raise any synth error
                    done += 1
                    ctx.progress("tts", done / total, f"sentence {done}/{total}")
        else:
            for i in range(len(jobs)):
                _synth(i)
                ctx.progress("tts", (i + 1) / total, f"sentence {i + 1}/{total}")

        # Write WAVs + build the index in source order.
        for (s, _kwargs), wav_bytes in zip(jobs, results):
            if wav_bytes is None:
                continue
            wav_path = out_dir / f"{s.idx:04d}.wav"
            wav_path.write_bytes(wav_bytes)
            sr, duration = _wav_header_meta(wav_bytes)
            sample_rate = sample_rate or sr
            sentences.append(SynthesizedSentence(
                idx=s.idx,
                wav_path=wav_path,
                wav_duration=duration,
                source_start=s.start,
                source_end=s.end,
                target_text=s.target_text,
            ))

        out = SynthesizedAudio(sentences=sentences, sample_rate=sample_rate)
        _write_index(ctx.job_dir, out)
        _maybe_write_cloning_summary(out_dir, self.inner)
        return out


def _wav_header_meta(wav_bytes: bytes) -> tuple[int, float]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        return sr, n / sr


def _write_index(job_dir: Path, synth: SynthesizedAudio) -> None:
    payload = {
        "sample_rate": synth.sample_rate,
        "sentences": [
            {
                "idx": s.idx,
                "wav_path": str(s.wav_path),
                "wav_duration": s.wav_duration,
                "source_start": s.source_start,
                "source_end": s.source_end,
                "target_text": s.target_text,
            }
            for s in synth.sentences
        ],
    }
    (job_dir / "synthesized.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _maybe_write_cloning_summary(tts_dir: Path, inner: Any) -> None:
    """Write `tts/voice_cloning_summary.json` if the candidate exposes
    `summarize()`. Candidates without IVC (Piper) skip this silently."""
    summarize = getattr(inner, "summarize", None)
    if not callable(summarize):
        return
    try:
        summary = summarize()
    except Exception as exc:
        print(f"tts: candidate summarize() raised {type(exc).__name__}: {exc}")
        return
    (tts_dir / "voice_cloning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )


def build(config: dict[str, Any]) -> SentenceTTS:
    module = importlib.import_module(config["module"])
    inner = module.build(config)
    if isinstance(inner, SentenceTTS):
        return inner
    return SentenceTTS(inner=inner)
