"""Mux stage — layer AR voice over the accompaniment music bed.

For each synthesized sentence:
  1. Compute the source window from ASR timing.
  2. Stretch ratio = wav_duration / window, clamped to ±stretch_cap.
  3. Apply pyrubberband.time_stretch (pitch-preserving) if outside [0.98, 1.02].
  4. Position the AR voice at the sentence's source start — unless that would
     overlap the previous line's audio (common when the translated TTS line is
     longer than the source slot), in which case ripple it to just after the
     previous line (`voice_no_overlap`, default on). Drift resets at every
     source pause, so it stays local to a dense burst; lip sync is not a goal.

Per sentence, also scale the AR voice so its RMS matches the source vocals'
RMS in the same window (capped at ±12 dB). Keeps shouted lines shouted and
quiet lines quiet, so the cloned voice doesn't sit at a single flat level
out of step with the rest of the scene.

Then mix:
  out = accompaniment * music_bed_gain + voice * voice_gain
with a peak normalize to prevent clipping. Voice tracks live at the same
sample rate and channel count as the music bed (stereo, typically 44.1 kHz)
so the bed isn't downgraded by the voice path.

System requirement: `rubberband` CLI installed (`brew install rubberband`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    AudioTrack,
    MuxedAudio,
    SeparatedAudio,
    SynthesizedAudio,
)


_NO_STRETCH_BAND = (0.98, 1.02)
# Cap on the per-sentence loudness-match gain so a mostly-silent source
# window can't blast the AR voice to clipping (or, conversely, suppress it
# to silence). ±12 dB → linear factor between ~0.25× and ~4×.
_LEVEL_MATCH_GAIN_CAP_DB = 12.0
# Minimum RMS (linear, on float32 audio in [-1, 1]) below which we treat the
# signal as "essentially silent" and skip the match. Roughly -60 dBFS.
_LEVEL_MATCH_RMS_FLOOR = 1e-3



def _CLICK_GUARD_SAMPLES(sr: int) -> int:
    """Shortest fade that still removes a step discontinuity: 4 ms."""
    return max(1, int(0.004 * sr))


def _leading_quiet_samples(mono, rel_db: float = -30.0) -> int:
    """Samples before the clip's first real speech energy.

    Measured against the clip's own peak, so it does not care how loud the
    line is overall. Used to keep a fade-in inside the silence a clip
    actually has instead of ramping over its first consonant.
    """
    import numpy as np

    peak = float(np.abs(mono).max()) if len(mono) else 0.0
    if peak <= 0.0:
        return len(mono)
    thresh = peak * (10.0 ** (rel_db / 20.0))
    loud = np.flatnonzero(np.abs(mono) >= thresh)
    return int(loud[0]) if len(loud) else len(mono)


class FFmpegPadStretch:
    id = "ffmpeg_pad_stretch"

    def mux(
        self,
        synth: SynthesizedAudio,
        source: AudioTrack,
        separated: SeparatedAudio,
        ctx: JobContext,
    ) -> MuxedAudio:
        import soundfile as sf

        cap = ctx.options.stretch_cap
        voice_lin = 10.0 ** (ctx.options.voice_gain_db / 20.0)
        music_lin = ctx.options.music_bed_gain
        vocals_lin = float(getattr(ctx.options, "vocals_bed_gain", 0.12))

        # Music bed = accompaniment + full vocals stem ducked under the TTS
        # audio. Outside TTS spans, the original vocal expressions (laughs,
        # screams, breaths, un-ASR'd dialogue) survive. Inside, vocals are
        # crossfaded to zero so the AR dub plays alone. The Silero-derived
        # speech_vocals/non_speech_vocals split is unreliable on cartoon
        # dialogue and is no longer used here.
        bed, sr = sf.read(str(separated.accompaniment), dtype="float32", always_2d=True)
        n_samples = bed.shape[0]
        n_channels = bed.shape[1]

        vocals_arr: np.ndarray | None = None
        if separated.vocals is not None and separated.vocals.exists():
            v, vsr = sf.read(str(separated.vocals), dtype="float32", always_2d=True)
            if vsr == sr and v.shape == bed.shape:
                vocals_arr = v

        voice_track = np.zeros_like(bed)
        duck_mask = np.ones(n_samples, dtype="float32")
        fade = int(0.020 * sr)
        fade_in_curve = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade, dtype="float32")))
        fade_out_curve = fade_in_curve[::-1]
        # Per-sentence afade applied to the synthesized voice itself (not the
        # duck mask) to suppress the original-voice bleed reported as quality
        # issue #4. The cause is that ASR sentence boundaries are word-aligned
        # but the duck-mask boundaries are AR-audio-aligned — there's a small
        # window at the start/end of each synthesized line where the original
        # vocal isn't yet fully ducked but the AR voice has already begun /
        # has just ended. A short cosine fade on the AR audio itself softens
        # those transitions enough that the bleed becomes inaudible.
        sentence_fade_seconds = float(getattr(ctx.options, "sentence_afade_seconds", 0.05))
        sentence_fade = max(0, int(sentence_fade_seconds * sr))
        # Lead/tail pad (samples) on the duck window. ASR word-onset timestamps
        # run ~0.24s LATE vs the true speech onset, so ducking only from
        # source_start lets the original speaker leak at the start of every line
        # (quality issue #9). Open the duck this many samples early and hold it
        # past the line end. Only the original VOCALS stem is ducked (not the
        # music bed), so a generous lead doesn't dip the music.
        duck_lead = max(0, int(float(getattr(ctx.options, "duck_lead_seconds", 0.25)) * sr))
        duck_tail = max(0, int(float(getattr(ctx.options, "duck_tail_seconds", 0.10)) * sr))
        # No-overlap "ripple" placement: a line never starts before the prior
        # line's audio ends (+ a small gap). `voice_cursor` tracks the end of
        # the last placed line. A source pause naturally pushes the next line's
        # source_start past the cursor, so drift resets at every gap instead of
        # accumulating across the whole clip. See PipelineOptions.voice_no_overlap.
        no_overlap = bool(getattr(ctx.options, "voice_no_overlap", True))
        overlap_gap = max(0, int(float(getattr(ctx.options, "voice_overlap_gap_seconds", 0.06)) * sr))
        max_compress = max(1.0, float(getattr(ctx.options, "voice_max_compress", 1.5)))
        voice_cursor = 0
        placements: list[dict] = []
        # Next line's source start (samples) per index — the room a line has to
        # speak before the next one wants to. Last line has unbounded room.
        next_src = [int(synth.sentences[j + 1].source_start * sr) for j in range(len(synth.sentences) - 1)]
        next_src.append(n_samples)
        total = len(synth.sentences) or 1

        for i, s in enumerate(synth.sentences):
            with ctx.probe.section("read_s"):
                audio, file_sr = sf.read(str(s.wav_path), dtype="float32",
                                         always_2d=False)
            if file_sr != sr:
                audio = _simple_resample(audio, file_sr, sr)
            window = max(0.0, s.source_end - s.source_start)
            audio = _stretch_to_window(audio, sr, window, cap, probe=ctx.probe)

            start_sample = _placement_start(
                int(s.source_start * sr), voice_cursor, overlap_gap, no_overlap
            )
            # Hybrid: if the line still wouldn't fit before the next line wants
            # to speak, speed it up (pitch-preserving) toward that gap, capped
            # at `max_compress`. Whatever still doesn't fit is absorbed by the
            # ripple (the next line shifts a little). "A little of everything."
            if no_overlap and max_compress > 1.01:
                available = next_src[i] - start_sample
                if available > 0 and len(audio) > available:
                    ratio = min(len(audio) / available, max_compress)
                    if ratio > 1.02:
                        audio = _stretch_audio(audio, sr, ratio, probe=ctx.probe)
            end_sample = start_sample + len(audio)
            if start_sample >= n_samples:
                ctx.progress("mux", (i + 1) / total, f"sentence {i+1}/{total} (past end)")
                continue
            if end_sample > n_samples:
                audio = audio[: n_samples - start_sample]
                end_sample = n_samples

            mono = audio  # voice is mono

            # Per-sentence loudness match: scale the AR voice so its RMS in
            # this sentence's window matches the source vocals' RMS in the
            # same source window. Keeps a shouted line shouted and a quiet
            # line quiet, instead of every cloned line landing at ElevenLabs'
            # default loudness. Skipped when the vocals stem isn't available
            # or the source window is effectively silent (the AR voice plays
            # at its native level, with `voice_gain_db` as the only trim).
            match_gain = 1.0
            if vocals_arr is not None:
                src_start_sample = int(s.source_start * sr)
                src_end_sample = min(int(s.source_end * sr), n_samples)
                if src_end_sample > src_start_sample:
                    src_window = vocals_arr[src_start_sample:src_end_sample]
                    src_rms = float(np.sqrt(np.mean(src_window ** 2))) if src_window.size else 0.0
                    synth_rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
                    if src_rms > _LEVEL_MATCH_RMS_FLOOR and synth_rms > _LEVEL_MATCH_RMS_FLOOR:
                        raw = src_rms / synth_rms
                        cap_lin = 10.0 ** (_LEVEL_MATCH_GAIN_CAP_DB / 20.0)
                        match_gain = max(1.0 / cap_lin, min(cap_lin, raw))
                        mono = mono * match_gain

            # Apply per-sentence afade in/out on the AR voice — see issue #4.
            #
            # The fade-IN is shortened to whatever leading near-silence the
            # clip actually has, with a short anti-click floor. 50 ms was
            # chosen for lines that begin with silence, and most do — median
            # speech onset over 2961 measured lines is 0.062 s. But the
            # **first line of a job begins at sample 0 in 74% of jobs**
            # (against 26% for every other line), so the full ramp landed on
            # the first word's attack and chewed through its consonant. That
            # is audible on every video precisely because nothing precedes it
            # to mask it — the author's report was "the very first word starts
            # with noise". Quality issue #30.
            #
            # The floor is 4 ms: enough to stop the discontinuity the fade
            # exists to prevent, short enough to be inaudible on an attack.
            # The fade-OUT is untouched — utterances end by decaying, so the
            # tail has room the head does not.
            n_aud = len(mono)
            if sentence_fade > 0 and n_aud > 2 * sentence_fade:
                lead = _leading_quiet_samples(mono)
                fin_n = int(min(sentence_fade, max(_CLICK_GUARD_SAMPLES(sr), lead)))
                fin = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fin_n, dtype="float32")))
                fout = 0.5 * (1 - np.cos(np.linspace(0, np.pi, sentence_fade,
                                                     dtype="float32")))[::-1]
                mono = mono.copy()
                mono[:fin_n] *= fin
                mono[-sentence_fade:] *= fout
            # Spread mono voice across all bed channels.
            voice_track[start_sample:end_sample, :] += mono[:, None] * voice_lin
            voice_cursor = end_sample
            placements.append({
                "idx": s.idx,
                "source_start": round(s.source_start, 3),
                "placed_start": round(start_sample / sr, 3),
                "placed_end": round(end_sample / sr, 3),
                "drift": round((start_sample - int(s.source_start * sr)) / sr, 3),
            })

            # Duck the original vocals across the full ASR sentence window
            # (not just the TTS span). When TTS is shorter than the source
            # utterance — common with the ±20% stretch cap on long
            # sentences — the original speaker would otherwise continue
            # talking after the AR voice finishes. The duck window is the
            # union of [source_start, source_end] and [tts_start, tts_end].
            if vocals_arr is not None:
                src_end_sample = min(int(s.source_end * sr), n_samples)
                # Duck from the source utterance start (silence the original
                # speaker there) through the end of wherever the AR voice plays
                # — with ripple the AR span can sit later than the source window.
                # Pad earlier by duck_lead (to beat the late ASR onset) and later
                # by duck_tail so the original is gone before the line and stays
                # gone just after it (quality issue #9).
                duck_start = min(start_sample, int(s.source_start * sr)) - duck_lead
                duck_start = max(0, duck_start)
                duck_end = max(end_sample, src_end_sample) + duck_tail
                duck_end = min(duck_end, n_samples)
                a = min(fade, duck_end - duck_start)
                if a > 0:
                    duck_mask[duck_start:duck_start + a] = np.minimum(
                        duck_mask[duck_start:duck_start + a], fade_out_curve[:a]
                    )
                    duck_mask[duck_end - a:duck_end] = np.minimum(
                        duck_mask[duck_end - a:duck_end], fade_in_curve[:a]
                    )
                duck_mask[duck_start + a:duck_end - a] = 0.0

            ctx.progress("mux", (i + 1) / total, f"sentence {i + 1}/{total}")

        # Accompaniment (music/SFX) plays at music_bed_gain. The original vocals
        # stem plays only at vocals_bed_gain where it survives the duck (gaps);
        # inside ASR windows duck_mask is 0 so the AR dub plays alone. Keeping
        # vocals on a separate, lower gain stops un-transcribed original dialogue
        # from leaking at full music level (quality issue #9 / #10).
        out_audio = bed * music_lin
        if vocals_arr is not None:
            out_audio = out_audio + vocals_arr * duck_mask[:, None] * vocals_lin
        out_audio = out_audio + voice_track

        # Peak normalize to -1 dBFS to prevent clipping after the sum.
        peak = float(np.max(np.abs(out_audio)))
        if peak > 0:
            target = 10.0 ** (-1.0 / 20.0)
            if peak > target:
                out_audio = out_audio * (target / peak)

        out_path = ctx.job_dir / "dub.wav"
        with ctx.probe.section("write_s"):
            sf.write(str(out_path), out_audio, sr, subtype="PCM_16")
        _write_meta(ctx.job_dir, out_path, sr, n_channels)
        # Sidecar for inspection/debugging: where each line actually landed
        # (post compress + ripple) and how far it drifted from its source time.
        (ctx.job_dir / "mux_placements.json").write_text(
            json.dumps({"sample_rate": sr, "placements": placements}, indent=2)
        )
        return MuxedAudio(path=out_path, sample_rate=sr)


def _placement_start(source_start: int, voice_cursor: int, gap: int, no_overlap: bool) -> int:
    """Where (in samples) to place a line's voice. With `no_overlap`, never
    before the previous line's end + `gap`; otherwise at its source time. A
    source pause makes `source_start > voice_cursor`, so drift resets at gaps."""
    if no_overlap and source_start < voice_cursor + gap:
        return voice_cursor + gap
    return source_start


def _stretch_to_window(
    audio: np.ndarray,
    sr: int,
    window: float,
    cap: float,
    probe: Any = None,
) -> np.ndarray:
    if window <= 0:
        return audio
    duration = len(audio) / sr
    if duration <= 0:
        return audio
    desired_ratio = duration / window
    clamped = max(1 - cap, min(1 + cap, desired_ratio))
    if _NO_STRETCH_BAND[0] <= clamped <= _NO_STRETCH_BAND[1]:
        return audio
    return _stretch_audio(audio, sr, clamped, probe=probe)


def _stretch_audio(audio: np.ndarray, sr: int, ratio: float,
                   probe: Any = None) -> np.ndarray:
    """Pitch-preserving time-stretch by `ratio` (>1 = faster/shorter). No-op on
    a near-1 ratio or if rubberband isn't available (returns the input).

    Each real stretch writes a temp WAV, spawns the `rubberband` CLI and reads
    the result back, and a line can be stretched twice (to its window, then to
    fit before the next). `n_stretch_calls` beside `stretch_s` is what separates
    process-spawn cost from the stretch itself. Note the bare `except` also
    means a MISSING `rubberband` binary silently disables stretching, so a
    probe reporting 0 calls on dense dialogue is a finding, not noise."""
    if _NO_STRETCH_BAND[0] <= ratio <= _NO_STRETCH_BAND[1]:
        return audio
    try:
        import pyrubberband as pyrb  # lazy import
        if probe is not None:
            probe.bump("n_stretch_calls")
            with probe.section("stretch_s"):
                return pyrb.time_stretch(audio, sr, ratio)
        return pyrb.time_stretch(audio, sr, ratio)
    except Exception:
        return audio


def _simple_resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    ratio = to_sr / from_sr
    n = int(len(audio) * ratio)
    if n == 0:
        return np.zeros(0, dtype="float32")
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype("float32")


def _write_meta(job_dir: Path, path: Path, sample_rate: int, channels: int) -> None:
    meta = {"path": str(path), "sample_rate": sample_rate, "channels": channels}
    (job_dir / "dub.meta.json").write_text(json.dumps(meta, indent=2))


def build(config: dict[str, Any]) -> FFmpegPadStretch:
    return FFmpegPadStretch()
