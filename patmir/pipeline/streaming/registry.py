"""Speaker registry + gender/age-matched voice assignment (Slice 2 #7).

Keeps a stable identity → voice mapping across the whole video so a character
keeps one voice. For the ElevenLabs preset-pool TTS, each speaker is profiled
(gender + age, see `speaker_profile.py`) and assigned the best-matching pool
voice; for the OmniVoice clone TTS, the registry instead locks the speaker's
ref clip (set via `lock_ref`).

The cross-section *identity* resolution (deciding that `speaker_0` in a later
streamed section is the same person as one established in the prelude) is a
separate concern wired in when the streaming runner feeds sections (#9); this
module takes an already-resolved `speaker_id` and owns only the profile→voice
policy, which is pure and unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Ordered age buckets; adjacency is used to relax a match (a `young` speaker
# can take an `adult` voice before falling back to any gender match).
AGE_ORDER = ["child", "young", "adult", "elder"]

# Default ECAPA cosine threshold above which two clips are deemed the same
# speaker. Same-speaker pairs typically score >0.5, different speakers <0.3;
# 0.40 is a conservative middle that favours NOT merging distinct characters
# (a false-merge swaps a character's voice, which is more jarring than a
# false-split that just spawns an extra voice). Tune per content.
DEFAULT_MATCH_THRESHOLD = 0.40


@dataclass(frozen=True)
class VoiceEntry:
    id: str
    gender: str = "any"                  # "male" | "female" | "any"
    ages: tuple[str, ...] = ("any",)     # the set of age buckets this voice can cover

    def covers_age(self, age: str) -> bool:
        return "any" in self.ages or age == "any" or age in self.ages

    def age_distance(self, age: str) -> int:
        """Smallest distance from `age` to any age this voice covers."""
        return min((_age_distance(a, age) for a in self.ages), default=99)


@dataclass
class SpeakerProfile:
    gender: str = "any"
    age: str = "any"


def _age_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if a not in AGE_ORDER or b not in AGE_ORDER:
        return 99  # "any" or unknown → treat as far, only used as last resort
    return abs(AGE_ORDER.index(a) - AGE_ORDER.index(b))


def build_pool(raw: list) -> list[VoiceEntry]:
    """Normalize raw pool config (from patmir.toml) into VoiceEntry list.

    Each entry may be a plain voice-id string, or a dict with `id`, optional
    `gender`, and optional `age` — where `age` is either a single bucket
    ("adult") or a list of buckets the voice can cover (["young", "adult"]).
    A single voice covering multiple adjacent ages is the intended way to
    stretch a small pool across child/young/adult/elder.
    """
    pool: list[VoiceEntry] = []
    for e in raw:
        if isinstance(e, str):
            pool.append(VoiceEntry(id=e))
            continue
        age = e.get("age", "any")
        ages = tuple(age) if isinstance(age, (list, tuple)) else (age,)
        pool.append(VoiceEntry(id=e["id"], gender=e.get("gender", "any"), ages=ages or ("any",)))
    return pool


class SpeakerRegistry:
    """Stable per-speaker voice assignment from a gender/age-tagged pool."""

    def __init__(self, pool: list[VoiceEntry], default_voice_id: str | None = None):
        if not pool:
            raise ValueError("SpeakerRegistry needs a non-empty voice pool")
        self.pool = pool
        self.default_voice_id = default_voice_id or pool[0].id
        self._assigned: dict[str, str] = {}     # speaker_id → voice_id
        self._use_count: dict[str, int] = {v.id: 0 for v in pool}
        self._locked_ref: dict[str, str] = {}    # speaker_id → ref clip (OmniVoice path)
        self._embeddings: dict[str, object] = {}  # speaker_id → ECAPA embedding (np.ndarray)

    # ── ElevenLabs preset-pool path ────────────────────────────────────────

    def assign_voice(self, speaker_id: str, profile: SpeakerProfile | None = None) -> str:
        """Return a stable pool voice for `speaker_id`, matched to its profile.

        Match priority: exact (gender, age) → same gender, nearest age →
        same gender → nearest age (any gender) → least-used overall. Within a
        tier the least-used voice wins, so distinct characters spread across
        the pool instead of clumping on one voice.
        """
        if speaker_id in self._assigned:
            return self._assigned[speaker_id]

        profile = profile or SpeakerProfile()
        chosen = self._best_match(profile)
        self._assigned[speaker_id] = chosen.id
        self._use_count[chosen.id] += 1
        return chosen.id

    def _best_match(self, profile: SpeakerProfile) -> VoiceEntry:
        g, a = profile.gender, profile.age

        def least_used(entries: list[VoiceEntry]) -> VoiceEntry:
            return min(entries, key=lambda v: (self._use_count[v.id], self.pool.index(v)))

        gender_ok = [v for v in self.pool if g == "any" or v.gender in (g, "any")]

        # 1) same gender, voice covers this age → least-used wins (spread variety).
        tier = [v for v in gender_ok if v.covers_age(a)]
        if tier:
            return least_used(tier)
        # 2) same gender, nearest covered age.
        if gender_ok:
            return sorted(
                gender_ok,
                key=lambda v: (v.age_distance(a), self._use_count[v.id], self.pool.index(v)),
            )[0]
        # 3) any gender that covers this age.
        tier = [v for v in self.pool if v.covers_age(a)]
        if tier:
            return least_used(tier)
        # 4) anything, nearest covered age.
        return sorted(
            self.pool,
            key=lambda v: (v.age_distance(a), self._use_count[v.id], self.pool.index(v)),
        )[0]

    # ── OmniVoice clone path ────────────────────────────────────────────────

    def lock_ref(self, speaker_id: str, ref_audio: str) -> str:
        """Lock a speaker's clone reference on first sufficient clip so the
        cloned voice doesn't shift mid-video. Returns the locked ref."""
        return self._locked_ref.setdefault(speaker_id, ref_audio)

    def locked_ref(self, speaker_id: str) -> str | None:
        return self._locked_ref.get(speaker_id)

    # ── cross-section identity (ECAPA embeddings) ──────────────────────────

    def set_embedding(self, speaker_id: str, embedding) -> None:
        """Store a voice embedding for a known character (set in the prelude /
        when a new speaker is first registered)."""
        if embedding is not None:
            self._embeddings[speaker_id] = embedding

    def resolve(self, embedding, threshold: float = DEFAULT_MATCH_THRESHOLD) -> str | None:
        """Return the known speaker_id whose stored embedding best matches
        `embedding` (cosine ≥ threshold), or None if none qualify → caller
        registers a new character. Used by the streaming runner (#9) to map a
        later section's local speaker to an established character.
        """
        if embedding is None or not self._embeddings:
            return None
        from patmir.pipeline.streaming.embeddings import cosine

        best_id, best_score = None, -1.0
        for spk, emb in self._embeddings.items():
            score = cosine(embedding, emb)
            if score > best_score:
                best_id, best_score = spk, score
        return best_id if best_score >= threshold else None

    def has_embedding(self, speaker_id: str) -> bool:
        return speaker_id in self._embeddings

    # ── introspection ─────────────────────────────────────────────────────

    def assignments(self) -> dict[str, str]:
        return dict(self._assigned)

    def known_speakers(self) -> list[str]:
        return list(self._assigned.keys())
