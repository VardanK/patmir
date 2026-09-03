"""JobContext — per-run state passed to every stage.

Holds the working directory, source path, detected source language (set by
ASR), pipeline options, and a `progress(...)` callback that appends to
`<job_dir>/progress.jsonl` for the CLI tail + FastAPI SSE stream.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patmir.pipeline.probe import Probe


class JobCancelled(Exception):
    """Raised at a checkpoint when the job's cancel token has been set.

    A distinct type, not a generic failure: the runner marks the stage
    `cancelled` rather than `failed`, and the server reports `cancelled` so a
    deliberate stop does not render as a red error.

    Cancellation is necessarily **cooperative**. The heavy stages run in a
    worker thread via `asyncio.to_thread`, and a Python thread cannot be killed
    — `task.cancel()` only abandons the await while the thread runs on, holding
    the GPU *and* the shared accelerator lock (ADR
    `2026-08-25-one-accelerator-at-a-time`), so the next job would queue behind
    a job the user believes is dead. Hence a token checked at explicit
    checkpoints, and never `task.cancel()`.
    """


# --- Audience age guardrail --------------------------------------------------
#
# The age register the translator is steered toward. Prompt-only softening: the
# candidate is asked to choose safer WORDING, and is explicitly forbidden from
# dropping, blanking, bleeping or redacting a line. This dub is the only audio
# track its listener has, so a silent gap is strictly worse than a gentler word.

AUDIENCE_LEVELS: tuple[str, ...] = ("child", "teenager", "adult")

# What a submission gets when it says nothing. `teenager` -- NOT the same as the
# level that leaves the cache key alone. Keeping the two apart is deliberate; see
# NEUTRAL_AUDIENCE.
#
# Was `child` when the guardrail shipped (2026-08-28); moved to `teenager` the
# same day at the author's request. `child` remains one click away and is the
# right choice for cartoons; `teenager` is the safer *default* because it is the
# one that is wrong in the smaller direction across a mixed library -- too mild
# for an adult drama costs a little bite, where too coarse for a young viewer
# costs the thing this feature exists to prevent.
DEFAULT_AUDIENCE = "teenager"

# The level whose composed prompt is byte-identical to the pre-guardrail prompt,
# and therefore the only one that appends nothing to the cache key. Every dub
# cached before this feature existed was made with exactly this prompt, so its
# key must keep pointing at it. The `mode` suffix in ui/cache.py folds in on the
# *default*; here the two differ, which means shipping this re-dubs every video
# once at the new default -- accepted, because the alternative is serving
# adult-register audio under a child-labelled setting.
NEUTRAL_AUDIENCE = "adult"


def normalize_audience(value: str | None) -> str:
    """Validate an audience level. `None`/empty -> DEFAULT_AUDIENCE."""
    level = (value or DEFAULT_AUDIENCE).strip().lower()
    if level not in AUDIENCE_LEVELS:
        raise ValueError(
            f"unknown audience: {value!r} "
            f"(expected one of {', '.join(AUDIENCE_LEVELS)})"
        )
    return level


def resolve_audience(requested: str | None, options: "PipelineOptions") -> str:
    """The one place the precedence rule lives.

    A per-submission value wins; absent one, the server's
    `[pipeline.options].audience` applies. The resolved string drives BOTH the
    prompt and the cache-key suffix, so the two can never disagree -- resolve
    once per submission and reuse the result.
    """
    if requested:
        return normalize_audience(requested)
    return normalize_audience(getattr(options, "audience", None))


@dataclass
class PipelineOptions:
    stretch_cap: float = 0.20
    asr_int8: bool = True
    keep_intermediate: bool = True
    # Linear gain applied to the accompaniment bed when mixing the AR voice
    # on top. 1.0 = unchanged source mix volume; 0.6 = -4 dB (a touch quieter
    # so the dialogue sits above the music). The voice itself is normalized
    # to roughly match the original dialogue loudness.
    music_bed_gain: float = 0.6
    # Linear gain on the ORIGINAL VOCALS stem where it survives the duck (i.e.
    # in gaps between ASR sentences — un-transcribed dialogue, laughs, breaths).
    # Inside ASR windows the duck zeroes it so the AR dub plays alone; this floor
    # only governs the gaps. The old behavior folded vocals into the bed at
    # music_bed_gain (0.6), so any dialogue ASR missed leaked at full music
    # level (quality issue #9 / #10). 0.12 keeps non-verbal texture faint;
    # set 0.0 to fully remove the original voice.
    vocals_bed_gain: float = 0.12
    # Global trim (dB) on top of the mux per-sentence RMS loudness match.
    # Negative = quieter. -3.0 came out of the 2026-05-27 masha-2-5min ear-test.
    voice_gain_db: float = -3.0
    # Minimum duration (seconds) a turn must be — after 100 ms inward trim
    # on each side — to qualify as a reference clip for OmniVoice cloning.
    # Used by ref_picker, which runs in the runner post-emotion stage.
    min_ref_seconds: float = 1.5
    # Per-sentence afade duration (seconds) applied to the synthesized track
    # at sentence boundaries in mux. Suppresses the original-voice bleed at
    # boundaries reported as quality issue #4.
    sentence_afade_seconds: float = 0.05
    # Pin ONE reference clip per speaker (the longest/cleanest, used for every
    # line regardless of emotion) instead of routing emotion through a
    # per-(speaker, emotion) ref clip. Per-emotion refs made a character's
    # cloned voice audibly swap whenever the detected emotion changed — badly so
    # on chaotic cartoon audio where the per-emotion clips are short, noisy
    # snippets (screams/laughs). Pinning one ref keeps each character's voice
    # consistent. Trades away per-line emotion routing (emotion instead comes
    # from the TTS candidate's own control — an inline `<|emotion:..|>` tag on
    # the Higgs candidates). Set False to restore the Phase-2.x ref binning.
    tts_pin_speaker_ref: bool = True
    # Duck the original vocals stem this many seconds BEFORE each line's source
    # start (and after its end via duck_tail_seconds). ASR word-onset timestamps
    # land ~0.24s LATE vs the true speech onset, so without a lead the original
    # speaker is audible at the start of every line before the duck engages
    # (quality issue #9). The pre-roll silences the bed-vocals before the onset.
    duck_lead_seconds: float = 0.25
    duck_tail_seconds: float = 0.10
    # No-overlap voice placement in mux. When True, a synthesized line never
    # starts before the previous line's audio has finished — the mux "ripples"
    # a line later if its translated audio (often longer than the source slot,
    # e.g. ElevenLabs lines over fast cartoon dialogue) would otherwise pile on
    # top of the previous one. Drift self-resets at every speech gap (a pause
    # in the source puts the next line's start past the cursor), so it stays
    # local to dense bursts. Set False for the old place-at-source-time-with-
    # overrun behavior. Gap below is the minimum silence inserted between lines.
    voice_no_overlap: bool = True
    voice_overlap_gap_seconds: float = 0.06
    # Max pitch-preserving speed-up applied to a line to help it fit before the
    # next line, so dense dialogue (translated TTS often ~1.4× longer than the
    # source) is absorbed by a mix of mild compression AND shifting rather than
    # pure shifting (which packs lines back-to-back and truncates the tail).
    # 1.0 disables extra compression (pure ripple). 1.5 ≈ up to 50% faster.
    voice_max_compress: float = 1.5
    # Curated fallback ref-clip pool for clone TTS ([[tts_ref_pool]] tables in
    # patmir.toml → {path, gender?, text?}). A speaker with no usable ref used
    # to synthesize with NO reference — an anonymous default voice that also
    # varies run to run. With a pool configured, such speakers get a stable
    # stand-in clip from here (gender-matched when the speaker's profile is
    # known); with no pool configured the adapter falls back to reusing
    # another character's pinned ref from the same job before ever going
    # ref-less. Quality issue #12.
    tts_ref_pool: list[dict] = field(default_factory=list)
    # Minimum mean level (dBFS RMS) for a sliced turn to qualify as a ref clip.
    # Quiet slices are usually far-field / bleed-heavy and clone badly. Windows
    # below the floor lose to any louder eligible window; if ALL of a cell's
    # windows fail the floor, the longest still wins (never drop a speaker's
    # only ref). Quality issue #12.
    ref_min_rms_db: float = -33.0
    # Pinning one ref per speaker (`tts_pin_speaker_ref`) used to take the
    # longest clip regardless of its emotion bin, and Higgs clones prosody
    # from the reference — so a character whose longest clip was angry
    # delivered the whole film angrily. Prefer the neutral clip when it is at
    # least this share of the longest one's length; above 1.0 disables the
    # preference. Quality issue #15.
    tts_pin_neutral_fraction: float = 0.5
    # Watch-along only: seconds of the opening analyzed in the prelude to
    # establish the cast before playback starts. Smaller = faster time-to-first
    # -audio, but fewer characters are seen up front (later ones register on the
    # fly mid-stream). 40s balances first-audio latency against cast coverage.
    prelude_window_seconds: float = 40.0
    # Build every active stage once when the local server starts, instead of
    # letting the first real submission pay model load (plus, on MLX, the
    # one-time ~240s Metal-kernel JIT). Costs the full resident set up front —
    # ~10 GB with the current models — so set False on a memory-constrained box
    # or when the server is started just to browse finished runs.
    prewarm_stages: bool = True
    # Watch-along: analyse the next section while this one waits on the
    # translate round-trip. Translate is 44.2 s of a 153.9 s watch-along wall
    # (29%) as one cloud call per section, with the GPU idle throughout;
    # analysis is the GPU work that fits in that gap. A single lock keeps GPU
    # phases serialized, which the TTS CUDA-graph capture requires. False
    # restores the fully serial section loop.
    section_prefetch: bool = True
    # Age register the translator is steered toward: "child" | "teenager" |
    # "adult". Prompt-only softening -- gentler wording, never a dropped,
    # blanked or bleeped line.
    #
    # THE ONE EXCEPTION to "options never touch the cache key". Every other
    # field here is audio tuning and is deliberately excluded (ui/cache.py:10-12)
    # so an ear-test loop does not invalidate everything; this one changes the
    # Armenian TEXT, so re-picking a level must not replay the previous level's
    # audio. It stays out of `config_hash` (which hashes candidate ENTRIES, and
    # this is not a model selection -- it must not fragment the warm stage cache,
    # keyed `(stage, candidate_id)`) and is folded into `cache_key` as an
    # explicit `|audience=<level>` suffix instead.
    #
    # This is the DEFAULT for `patmir dub` and for any client that omits the
    # field. The Chrome extension sends its own value on every submission and
    # wins. ADR: 2026-08-28-audience-age-guardrail.
    audience: str = DEFAULT_AUDIENCE
    # NOTE (2026-08-24): `emotion_guidance` used to live here — a per-bin
    # OmniVoice `guidance_scale` lookup. Removed because it had been INERT
    # since 2026-06-20: the adapter only forwards it to candidates whose
    # `synthesize` accepts `guidance_scale`, and the active Higgs candidates
    # do not (they take an inline `<|emotion:..|>` tag instead). So quality
    # issue #11's "lowered guidance to 1.3-1.7 to stop over-emoting" stopped
    # applying the moment TTS moved to Higgs, and nothing said so for two
    # months. Registered OmniVoice fallbacks now use their own constructor
    # default (2.0) rather than a per-emotion override; if per-bin guidance is
    # ever wanted back, restore it WITH a warning when the active candidate
    # cannot accept it. ADR
    # 2026-08-24-asr-sentence-segmentation-and-cuda-baseline.


@dataclass
class JobContext:
    job_dir: Path
    source_path: Path
    options: PipelineOptions
    source_lang: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Cooperative cancel token, owned by the caller (the server's JobManager /
    # StreamManager keep one per job). Deliberately a separate field rather than
    # a key in `extra`: `extra` is the cross-stage data hand-off channel
    # (vad_intervals, diarization, emotion_predictions — see pipeline.md), and a
    # control signal is not that.
    cancel: threading.Event | None = None
    # Intra-stage timing, armed by `run_job` only when `extra["bench"]` is set
    # and reset+dumped per stage by the runner. A separate field for the same
    # reason `cancel` is: `extra` is the cross-stage DATA hand-off channel, and
    # an instrument is not data. Disabled by default and free on the hot path —
    # `section()` returns a shared no-op context manager.
    probe: Probe = field(default_factory=Probe)

    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def raise_if_cancelled(self) -> None:
        """Checkpoint. Call where aborting is cheap and leaves usable state."""
        if self.cancelled():
            raise JobCancelled()

    def progress(self, stage: str, pct: float, msg: str = "") -> None:
        record = {
            "ts": time.time(),
            "stage": stage,
            "pct": round(pct, 4),
            "msg": msg,
        }
        path = self.job_dir / "progress.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
