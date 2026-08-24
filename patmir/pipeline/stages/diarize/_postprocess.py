"""Shared post-processing for the Sortformer diarize candidates.

Sortformer reaches this pipeline through two runtimes — `mlx-audio` on Apple
Silicon (`sortformer_mlx.py`) and NVIDIA's own NeMo on CUDA
(`sortformer_nemo.py`). Both get raw per-frame speaker segments out of the same
model and then need identical clean-up before the result can satisfy the
`Diarization` contract:

  * normalize the speaker label to the `SPEAKER_NN` convention that
    `apply_speaker_tags` and the ref-picker expect;
  * merge the fragmented same-speaker pieces the model emits into turns;
  * fold together two output slots that are really one person (see
    `merge_duplicate_speakers`);
  * re-derive overlap intervals, which the model supports (it is end-to-end and
    emits concurrent speakers) but does not report directly;
  * drop turns with no intersection with the separate stage's Silero VAD windows.

These live here rather than in either candidate so the two runtimes cannot drift
apart in post-processing and be mistaken for a model difference — the whole
point of having both is that they are the same diarizer.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from patmir.pipeline.stage import OverlapInterval, SpeakerTurn


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def passes_vad(
    start: float,
    end: float,
    vad_intervals: list[tuple[float, float]] | None,
) -> bool:
    """True if `[start, end)` intersects any VAD speech window.

    With no VAD intervals available every turn passes — the filter is an
    improvement on the diarizer's output, not a precondition for it.
    """
    if not vad_intervals:
        return True
    for vs, ve in vad_intervals:
        if vs < end and ve > start:
            return True
    return False


def norm_speaker(spk: Any) -> str:
    """Normalize a Sortformer speaker label to `SPEAKER_NN`.

    The two runtimes label speakers differently — mlx-audio yields ints (0..3)
    or `speaker_0`, NeMo yields an integer speaker index (sometimes rendered as
    a string) — and downstream code keys refs and transcript tags off this
    string, so it has to be one convention.
    """
    s = str(spk)
    digits = "".join(c for c in s if c.isdigit())
    if digits:
        return f"SPEAKER_{int(digits):02d}"
    return s


def merge_same_speaker(turns: list[SpeakerTurn], gap: float) -> list[SpeakerTurn]:
    """Merge consecutive same-speaker turns separated by <= `gap` seconds.

    Sortformer decides per frame, so one continuous utterance comes back as many
    short segments. Left unmerged they would each be a candidate ref clip and
    none would clear `min_ref_seconds`.
    """
    merged: list[SpeakerTurn] = []
    for t in sorted(turns, key=lambda x: (x.start, x.end)):
        if merged and merged[-1].speaker_id == t.speaker_id and t.start - merged[-1].end <= gap:
            merged[-1] = SpeakerTurn(
                start=merged[-1].start,
                end=max(merged[-1].end, t.end),
                speaker_id=t.speaker_id,
            )
        else:
            merged.append(SpeakerTurn(start=t.start, end=t.end, speaker_id=t.speaker_id))
    return merged


def derive_overlaps(turns: list[SpeakerTurn]) -> list[OverlapInterval]:
    """Time regions where >=2 distinct speakers are simultaneously active.

    Within one speaker the merged turns are non-overlapping, so an active count
    of >=2 necessarily means >=2 distinct speakers.
    """
    events: list[tuple[float, int]] = []
    for t in turns:
        events.append((t.start, 1))
        events.append((t.end, -1))
    events.sort()
    overlaps: list[OverlapInterval] = []
    active = 0
    ov_start: float | None = None
    for time, delta in events:
        prev = active
        active += delta
        if prev < 2 <= active:
            ov_start = time
        elif prev >= 2 > active and ov_start is not None:
            if time > ov_start:
                overlaps.append(OverlapInterval(start=ov_start, end=time))
            ov_start = None
    return overlaps


def merge_duplicate_speakers(
    turns: list[SpeakerTurn],
    gap: float,
    *,
    min_containment: float = 0.4,
    min_shared_seconds: float = 1.0,
) -> tuple[list[SpeakerTurn], dict[str, str]]:
    """Fold together speaker slots that are really one person.

    Sortformer sometimes emits ONE speaker on TWO of its output slots, so the
    same actor comes back as two ids whose turns are near-simultaneous. That is
    not a harmless labelling quirk — it costs three things downstream:

      * the actor's lines split across two `(speaker, emotion)` ref cells, so
        the character's cloned voice changes mid-scene;
      * `derive_overlaps` reports the twin pair as overlapping speech, and
        `pick_refs` rejects any turn touching an overlap — so a heavily
        duplicated speaker can end up with NO eligible ref window and fall
        through to another character's clip;
      * `apply_speaker_tags` sees a speaker change a few hundred ms into a
        sentence and cannot split it cleanly.

    Measured on `runs/20260824-163841-sample_1` (Reacher S02E01, 5 min):
    SPEAKER_03 was 71% concurrent with SPEAKER_02 and 7 of its turn pairs
    coincided at 90-100% of the shorter turn (5 at total containment), while the
    two genuinely-distinct speakers on that clip sat at 8-10% ragged, partial
    overlap. One 7.16 s "turn" carried a four-way question-and-answer exchange.

    The test is aggregate containment: `shared / min(total_a, total_b)`. Real
    simultaneous speech cannot come near the threshold there — a person would
    have to spend most of their speaking time talking over exactly one other
    person and never hold a solo turn.

    **The threshold is 0.4 because the model is not deterministic.** Four runs of
    the identical file put the duplicated pair at 0.705, 0.594 and twice above
    0.6, while the highest a genuinely-distinct pair reached across those same
    runs was 0.259. A first cut at 0.6 was calibrated on one run and silently
    stopped firing on the next — the 0.594 run came back with four speakers and
    one of them again borrowing another character's reference clip. 0.4 sits in
    the middle of the observed gap rather than at the edge of one side of it.
    Both bounds are pinned by test.

    `min_shared_seconds` keeps a pair of very short turns from tripping the ratio
    on noise.

    Returns `(turns, aliases)` where `aliases` maps each absorbed id to the id
    that survived (the group member with the most speech, ties by lowest id).
    Relabelled turns are re-merged with `gap`, since two slots of one speaker
    become self-overlapping turns that every downstream consumer — including
    `derive_overlaps` — assumes cannot happen.

    `min_containment > 1.0` disables the pass.
    """
    if min_containment > 1.0:
        return list(turns), {}

    by_spk: dict[str, list[SpeakerTurn]] = defaultdict(list)
    for t in turns:
        by_spk[t.speaker_id].append(t)
    if len(by_spk) < 2:
        return list(turns), {}

    total = {spk: sum(t.end - t.start for t in ts) for spk, ts in by_spk.items()}
    ids = sorted(by_spk)

    parent = {spk: spk for spk in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            floor = min(total[a], total[b])
            if floor <= 0:
                continue
            shared = sum(
                _overlap(x.start, x.end, y.start, y.end)
                for x in by_spk[a] for y in by_spk[b]
            )
            if shared >= min_shared_seconds and shared / floor >= min_containment:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    groups: dict[str, list[str]] = defaultdict(list)
    for spk in ids:
        groups[find(spk)].append(spk)

    aliases: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = min(members, key=lambda s: (-total[s], s))
        for m in members:
            if m != canonical:
                aliases[m] = canonical
    if not aliases:
        return list(turns), {}

    relabeled = [
        SpeakerTurn(start=t.start, end=t.end,
                    speaker_id=aliases.get(t.speaker_id, t.speaker_id))
        for t in turns
    ]
    return merge_same_speaker(relabeled, gap), aliases
