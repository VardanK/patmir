"""Covered-interval tracker for the live streaming scheduler (Slice 3 #10).

The forward iterator (`stream_sections`) walks the video strictly in order.
Watch-along needs the runner to be *playhead-driven* instead: process the
section the viewer is about to reach first, jump to a new region on seek, and
keep filling forward while paused. The scheduler decides *which* section to
process next; this tiny structure is the bookkeeping it does that on.

`Coverage` is a set of merged `[start, end)` intervals (the global timeline
regions already dubbed). It answers two questions the scheduler asks every
tick:

  * `first_uncovered(after=t)` — the earliest still-undubbed point at or after
    `t`. The scheduler processes a section starting there. If `t` lands inside
    a covered region, this returns the end of that region (fill *forward* of
    the playhead); if `t` is in a gap (e.g. just after a seek into virgin
    territory), it returns `t` itself (process *at* the playhead).
  * `is_complete(duration)` — whether the whole video is dubbed, so the
    scheduler can stop.

Pure + deterministic; no audio, models, or I/O. Unit-tested directly.
"""
from __future__ import annotations

# Two covered intervals within this many seconds of each other are merged —
# absorbs the sub-second slop between a VAD-snapped section end and the next
# section's start so coverage doesn't fragment into adjacent-but-disjoint runs.
_MERGE_EPSILON = 0.5


class Coverage:
    def __init__(self) -> None:
        self._intervals: list[list[float]] = []  # sorted, non-overlapping [start, end]

    def add(self, start: float, end: float) -> None:
        """Mark `[start, end)` dubbed, merging into any touching interval."""
        if end <= start:
            return
        merged: list[list[float]] = []
        for iv in sorted(self._intervals + [[start, end]]):
            if merged and iv[0] <= merged[-1][1] + _MERGE_EPSILON:
                merged[-1][1] = max(merged[-1][1], iv[1])
            else:
                merged.append(list(iv))
        self._intervals = merged

    def covers(self, t: float) -> bool:
        """True if point `t` falls inside a covered interval."""
        return any(s <= t < e for s, e in self._intervals)

    def first_uncovered(self, after: float = 0.0, *, until: float | None = None) -> float | None:
        """Earliest undubbed point at or after `after`.

        Returns `after` if that point is already in a gap; otherwise the end of
        the covered run containing it. Returns None if everything from `after`
        to `until` (defaults to +inf) is already covered.
        """
        t = max(0.0, after)
        # Walk covered runs in order, pushing `t` past any run it sits in.
        for s, e in self._intervals:
            if t < s:
                break          # gap before this run starts → t is uncovered
            if t < e:
                t = e          # inside this run → jump to its end and re-check
        if until is not None and t >= until:
            return None
        return t

    def next_covered_start(self, after: float) -> float | None:
        """Start of the earliest covered run beginning strictly after `after`.

        The clamp the scheduler needs. `first_uncovered` says where to *begin* a
        section; it says nothing about where that section must *stop*, so a
        window opened at a seek target runs its full ~35s straight through an
        already-dubbed region. Both renderings of those seconds then land in the
        manifest (`append_section` only de-duplicates starts within 0.5s), and
        the client has to choose between them — heard as a repeated or skipped
        line at the join.

        Returns None when nothing is covered ahead, i.e. no clamp applies.
        """
        for s, _e in self._intervals:
            if s > after:
                return s
        return None

    def is_complete(self, duration: float, *, tail_tolerance: float = 1.0) -> bool:
        """True if `[0, duration)` is fully dubbed (within `tail_tolerance` of
        the end — the final VAD-snapped section rarely lands exactly on EOF)."""
        gap = self.first_uncovered(0.0, until=duration - tail_tolerance)
        return gap is None

    def intervals(self) -> list[tuple[float, float]]:
        return [(s, e) for s, e in self._intervals]
