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
import time
import wave
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    Diarization,
    SpeakerRef,
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
        # Batched engines (Higgs on CUDA) synthesize many sentences in one
        # decode pass, because an AR decode that is memory-bandwidth-bound on
        # the model weights gets most of a second sentence for free. Such a
        # candidate exposes `synthesize_batch` + `tts_max_batch`; everything
        # else keeps the per-sentence loop below.
        #
        # The whole sentence list goes to the candidate in ONE call rather than
        # being chunked here, because the candidate sorts by expected length
        # before chunking — a chunk costs max(steps), not mean — and that sort
        # has to be global to be worth anything. In exchange the candidate calls
        # back per chunk (`on_chunk`) so progress and timing stay per-chunk
        # instead of going dark for the whole stage.
        batch_fn = getattr(inner, "synthesize_batch", None)
        self._batch_fn = batch_fn if callable(batch_fn) else None
        self._max_batch = max(1, int(getattr(inner, "tts_max_batch", 1)))
        self._batch_on_chunk = False
        if self._batch_fn is not None:
            try:
                self._batch_on_chunk = "on_chunk" in inspect.signature(self._batch_fn).parameters
            except (TypeError, ValueError):
                pass
        # IVC-style candidates (ElevenLabs) clone from audio alone and don't
        # need the reference transcript. They set `ref_text_optional = True`
        # so we treat refs with empty `ref_text` as usable for them.
        # Zero-shot candidates that require the transcript (OmniVoice, Piper)
        # leave the default and still gate on a non-empty ref_text.
        self._ref_text_optional = bool(getattr(inner, "ref_text_optional", False))
        # Stand-in pool state (quality issue #12): stable per-speaker fallback
        # refs for speakers with no usable clip of their own, spread least-used.
        self._standin: dict[str, SpeakerRef] = {}
        self._standin_use: dict[str, int] = {}

    def _usable(self, r) -> bool:
        if r is None:
            return False
        if self._ref_text_optional:
            return True
        return r.ref_text.strip() != ""

    def _lookup_ref(self, diarization: Diarization | None, speaker_id: str | None, emotion: str | None,
                    pin_speaker_ref: bool = False):
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

        # Any usable ref for this speaker, longest source window first.
        same_speaker = [
            r for (spk, _bin), r in diarization.refs.items()
            if spk == speaker_id and self._usable(r)
        ]
        # Pin one ref per speaker: ignore emotion entirely, always use the
        # longest/cleanest same-speaker clip so the cloned voice never swaps
        # mid-character. (Emotion still varies via guidance_scale.)
        if pin_speaker_ref:
            if same_speaker:
                return max(same_speaker, key=lambda r: r.source_end - r.source_start)
            return None

        emo = emotion or "neutral"
        ref = diarization.refs.get((speaker_id, emo))
        if self._usable(ref):
            return ref
        if emo != "neutral":
            ref = diarization.refs.get((speaker_id, "neutral"))
            if self._usable(ref):
                return ref
        # Any ref for this speaker — pick the longest source window for stability.
        if same_speaker:
            return max(same_speaker, key=lambda r: r.source_end - r.source_start)
        return None

    def _locked_ref(self, locked_refs: dict, speaker_id: str | None):
        """Streaming mode: the SpeakerRegistry locks ONE clone ref per global
        character (first clean clip seen). Prefer it over this section's clips
        so a character clones from the SAME audio in every section — the ref
        chosen per section otherwise varies with that section's diarization.
        Candidates that require a reference transcript skip a lock whose text
        is unknown and fall back to the normal per-section lookup chain."""
        if not speaker_id:
            return None
        entry = locked_refs.get(speaker_id)
        if not entry or not entry.get("ref_audio"):
            return None
        ref_text = entry.get("ref_text", "")
        if not self._ref_text_optional and not ref_text.strip():
            return None
        return SpeakerRef(
            speaker_id=speaker_id,
            ref_audio=Path(entry["ref_audio"]),
            ref_text=ref_text,
            source_start=0.0,
            source_end=0.0,
        )

    def _pool_ref(self, diarization: Diarization | None, speaker_id: str | None,
                  pool_cfg: list[dict], profiles: dict):
        """Stand-in ref for a speaker with no usable clip of its own (quality
        issue #12). Previously such speakers synthesized with NO reference —
        an anonymous default voice that also varied run to run. Instead:

          1. a clip from the curated `[[tts_ref_pool]]` (patmir.toml), gender-
             matched when the speaker's profile is known;
          2. no pool configured → reuse another character's pinned ref from
             this job (two characters sharing one voice beats a voice from
             nowhere — user-approved trade);
          3. nothing available → None (caller keeps the old default behavior).

        The choice is stable per speaker for this adapter instance and spreads
        least-used so distinct ref-less speakers land on distinct stand-ins.
        """
        if not speaker_id:
            return None
        if speaker_id in self._standin:
            return self._standin[speaker_id]

        candidates: list[SpeakerRef] = []
        if pool_cfg:
            gender = str((profiles.get(speaker_id) or {}).get("gender", "any"))
            tier = [e for e in pool_cfg
                    if gender == "any" or str(e.get("gender", "any")) in (gender, "any")]
            for e in (tier or pool_cfg):
                candidates.append(SpeakerRef(
                    speaker_id=speaker_id,
                    ref_audio=Path(e["path"]),
                    ref_text=str(e.get("text", "")),
                    source_start=0.0,
                    source_end=0.0,
                ))
        elif diarization is not None:
            best_by_speaker: dict[str, SpeakerRef] = {}
            for (spk, _emo), r in sorted(diarization.refs.items()):
                if spk == speaker_id or not self._usable(r):
                    continue
                cur = best_by_speaker.get(spk)
                if cur is None or (r.source_end - r.source_start) > (cur.source_end - cur.source_start):
                    best_by_speaker[spk] = r
            candidates = [best_by_speaker[k] for k in sorted(best_by_speaker)]

        candidates = [r for r in candidates if self._usable(r)]
        if not candidates:
            return None
        chosen = min(candidates, key=lambda r: (self._standin_use.get(str(r.ref_audio), 0), str(r.ref_audio)))
        self._standin_use[str(chosen.ref_audio)] = self._standin_use.get(str(chosen.ref_audio), 0) + 1
        self._standin[speaker_id] = chosen
        return chosen

    def synthesize(self, translated: TranslatedTranscript, ctx: JobContext) -> SynthesizedAudio:
        out_dir = ctx.job_dir / "tts"
        out_dir.mkdir(exist_ok=True)
        # Warm-cached instances (watch-along stage_cache) carry state across
        # jobs; reset per-job counters so voice_cloning_summary.json is per-job.
        reset = getattr(self.inner, "reset_summary", None)
        if callable(reset):
            reset()
        sentences: list[SynthesizedSentence] = []
        sample_rate = 0

        diar = ctx.extra.get("diarization")
        diarization: Diarization | None = diar if isinstance(diar, Diarization) else None
        voice_assignments = ctx.extra.get("voice_assignments") or {}
        locked_refs = ctx.extra.get("locked_refs") or {}
        speaker_profiles = ctx.extra.get("speaker_profiles") or {}
        ref_pool_cfg = list(getattr(ctx.options, "tts_ref_pool", None) or [])

        # Build each sentence's synth kwargs up front (cheap, no model calls),
        # so the actual synthesize() calls — the slow, network-bound part for a
        # cloud TTS — can be dispatched concurrently below.
        jobs: list[tuple[Any, dict[str, Any]]] = []
        for s in translated.sentences:
            if not s.target_text.strip():
                continue
            pin_ref = bool(getattr(ctx.options, "tts_pin_speaker_ref", True))
            ref = None
            if self._supports_ref:
                ref = self._locked_ref(locked_refs, s.speaker_id)
                if ref is None:
                    ref = self._lookup_ref(diarization, s.speaker_id, s.emotion, pin_speaker_ref=pin_ref)
                if ref is None:
                    ref = self._pool_ref(diarization, s.speaker_id, ref_pool_cfg, speaker_profiles)
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
        elapsed: list[float | None] = [None] * len(jobs)
        # Populated only on the batched path: one entry per decode chunk. Under
        # batching a per-SENTENCE wall does not exist — the chunk's cost is
        # shared — so timings are reported per chunk instead of inventing a
        # per-sentence number, which would multiply the true total by the batch
        # size in bench's `total_s` and `tts_rtf`.
        chunk_spans: list[tuple[list[int], float]] = []

        def _synth(idx: int) -> None:
            s, kwargs = jobs[idx]
            t0 = time.perf_counter()
            results[idx] = self.inner.synthesize(s.target_text, **kwargs)
            elapsed[idx] = time.perf_counter() - t0

        use_batch = self._batch_fn is not None and self._max_batch > 1 and len(jobs) > 1
        if use_batch:
            payload = [{"text": s.target_text, **kwargs} for s, kwargs in jobs]
            done = 0

            def _on_chunk(indices: list[int], wall_s: float) -> None:
                nonlocal done
                done += len(indices)
                chunk_spans.append((list(indices), wall_s))
                ctx.progress("tts", min(done / total, 1.0),
                             f"sentence {done}/{total} (batch of {len(indices)})")

            batch_kwargs: dict[str, Any] = {}
            if self._batch_on_chunk:
                batch_kwargs["on_chunk"] = _on_chunk
            t0 = time.perf_counter()
            outs = self._batch_fn(payload, **batch_kwargs)
            whole = time.perf_counter() - t0
            for i, wav_bytes in enumerate(outs[:len(jobs)]):
                results[i] = wav_bytes
            if not chunk_spans:
                # Candidate batches but doesn't report chunks: attribute the one
                # call as a single span so the totals stay right.
                chunk_spans.append((list(range(len(jobs))), whole))
            ctx.progress("tts", 1.0, f"{total} sentences in {len(chunk_spans)} batches")
        elif concurrency > 1 and len(jobs) > 1:
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
        bench_rows: list[dict[str, Any]] = []
        durations: list[float] = [0.0] * len(jobs)
        for job_i, ((s, _kwargs), wav_bytes, dt) in enumerate(zip(jobs, results, elapsed)):
            if wav_bytes is None:
                continue
            wav_path = out_dir / f"{s.idx:04d}.wav"
            wav_path.write_bytes(wav_bytes)
            sr, duration = _wav_header_meta(wav_bytes)
            sample_rate = sample_rate or sr
            durations[job_i] = duration
            sentences.append(SynthesizedSentence(
                idx=s.idx,
                wav_path=wav_path,
                wav_duration=duration,
                source_start=s.start,
                source_end=s.end,
                target_text=s.target_text,
            ))
            if dt is not None:
                bench_rows.append({
                    "idx": s.idx,
                    "elapsed_s": round(dt, 4),
                    "audio_s": round(duration, 4),
                    "chars": len(s.target_text),
                })

        if chunk_spans:
            # Per-chunk rows. `sum(elapsed_s)` is still the true synthesis wall
            # and `audio_s` the true audio, so bench's `total_s` / `tts_rtf`
            # remain correct; the distribution stats (mean/median/p90/max)
            # describe CHUNKS rather than sentences, which `kind` records.
            bench_rows = [
                {
                    "kind": "chunk",
                    "sentences": len(idxs),
                    "elapsed_s": round(wall_s, 4),
                    "audio_s": round(sum(durations[i] for i in idxs), 4),
                    "chars": sum(len(jobs[i][0].target_text) for i in idxs),
                }
                for idxs, wall_s in (
                    ([i for i in raw if 0 <= i < len(jobs)], w) for raw, w in chunk_spans
                )
                if idxs
            ]

        out = SynthesizedAudio(sentences=sentences, sample_rate=sample_rate)
        _write_index(ctx.job_dir, out)
        _maybe_write_cloning_summary(out_dir, self.inner)
        # Per-sentence latency trace — only when running under `patmir bench`
        # (ctx.extra["bench"]); a normal `patmir dub` leaves no timings file.
        if ctx.extra.get("bench") and bench_rows:
            with (out_dir / "timings.jsonl").open("w") as f:
                for row in bench_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
