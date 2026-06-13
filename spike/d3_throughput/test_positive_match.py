"""Positive cross-section identity proof (free / local ECAPA only).

The original full run has multiple refs for the SAME speaker (SPEAKER_00:
neutral/sad/angry — different times, different utterances) plus other speakers.
That's exactly a "returning character, different utterance" scenario. We:
  1. compute the full pairwise cosine matrix,
  2. confirm same-speaker pairs clear the 0.40 threshold and different speakers don't,
  3. simulate resolve(): seed a registry with one clip, resolve a *different* clip
     of the same speaker -> must return the same global id.

.venv/bin/python spike/d3_throughput/test_positive_match.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260528-010843-source"


def main():
    from patmir.pipeline.streaming.embeddings import embed, cosine
    from patmir.pipeline.streaming.registry import (
        DEFAULT_MATCH_THRESHOLD as TH,
        SpeakerProfile,
        SpeakerRegistry,
        build_pool,
    )
    from patmir.eval.candidates.tts.elevenlabs import _DEFAULT_VOICE_POOL

    refs = json.loads((RUN / "diarize/refs.json").read_text())
    # key -> (speaker_id, ref_audio); keep all emotion clips (different utterances)
    clips = {k: (r["speaker_id"], r["ref_audio"]) for k, r in refs.items()}
    embs = {k: embed(p) for k, (spk, p) in clips.items()}

    print(f"threshold = {TH}\n--- pairwise cosines ---")
    same_lo, diff_hi = 1.0, 0.0
    for a, b in combinations(clips, 2):
        spk_a, spk_b = clips[a][0], clips[b][0]
        c = cosine(embs[a], embs[b])
        same = spk_a == spk_b
        tag = "SAME-speaker" if same else "diff"
        ok = (c >= TH) if same else (c < TH)
        print(f"  {a:22} vs {b:22}: {c:.3f}  [{tag}] {'OK' if ok else 'FAIL'}")
        if same:
            same_lo = min(same_lo, c)
        else:
            diff_hi = max(diff_hi, c)

    print(f"\nlowest same-speaker cosine : {same_lo:.3f}  (must be >= {TH})")
    print(f"highest diff-speaker cosine: {diff_hi:.3f}  (must be <  {TH})")
    sep_ok = same_lo >= TH and diff_hi < TH

    # Simulate resolve(): seed registry with SPEAKER_00 neutral, resolve the
    # SPEAKER_00 *angry* clip (a different utterance) -> must match, not mint new.
    pool = build_pool(_DEFAULT_VOICE_POOL)
    reg = SpeakerRegistry(pool)
    reg.assign_voice("SPEAKER_00", SpeakerProfile("male", "adult"))
    reg.set_embedding("SPEAKER_00", embs["SPEAKER_00__neutral"])
    gid, is_new = reg.resolve_or_register(
        embs["SPEAKER_00__angry"], profile=SpeakerProfile("male", "adult"))
    print(f"\nresolve(SPEAKER_00 angry clip) -> {gid}, is_new={is_new}")
    match_ok = (gid == "SPEAKER_00" and not is_new)

    print(f"\n{'PASS' if sep_ok and match_ok else 'FAIL'}: "
          f"separation_ok={sep_ok}, returning-speaker-matches={match_ok}")


if __name__ == "__main__":
    main()
