"""Tests for the audience age guardrail.

The guardrail steers the translator's register toward `child` / `teenager` /
`adult`. Everything here defends one of three properties:

1. **`adult` is a no-op.** Its composed prompt is byte-identical to the
   pre-guardrail prompt, which is what lets `cache_key` leave existing keys
   alone. If this breaks, every cached dub on the machine silently becomes
   mislabelled.
2. **Nothing is ever dropped.** The clauses forbid deleting, blanking, bleeping
   or refusing a line, because this dub is its listener's only audio track. A
   test asserts that language is still present, so a future edit cannot quietly
   turn softening into redaction.
3. **A candidate that cannot steer says so.** The pipeline has a scar here —
   `emotion_guidance` sat inert for two months because the adapter silently
   dropped it. Warn, never fail, never stay silent.

ADR: `2026-08-28-audience-age-guardrail`.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from patmir.pipeline.context import (
    AUDIENCE_LEVELS,
    DEFAULT_AUDIENCE,
    NEUTRAL_AUDIENCE,
    JobContext,
    PipelineOptions,
    normalize_audience,
    resolve_audience,
)


def _set_env_key():
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


# --- vocabulary --------------------------------------------------------------


class AudienceVocabulary(unittest.TestCase):
    def test_default_is_softened_and_neutral_is_adult(self) -> None:
        """The two constants deliberately differ, unlike `mode`'s single
        DEFAULT_MODE: the default is a *softened* level, `adult` is the level
        that leaves the cache key alone (it reproduces the pre-guardrail
        prompt). Collapsing them either serves adult register under a
        softened-labelled setting, or re-dubs nothing when the level changes.

        The assertion that matters is the relationship, not the particular
        level: the default must be one of the softening levels and must not be
        the neutral one. Which of `child` / `teenager` it is, is a product call
        (it moved from `child` to `teenager` on 2026-08-28) and is pinned
        separately below so a deliberate change is one obvious edit."""
        self.assertIn(DEFAULT_AUDIENCE, ("child", "teenager"))
        self.assertEqual(NEUTRAL_AUDIENCE, "adult")
        self.assertNotEqual(DEFAULT_AUDIENCE, NEUTRAL_AUDIENCE)
        self.assertEqual(set(AUDIENCE_LEVELS), {"child", "teenager", "adult"})

    def test_the_current_default_is_teenager(self) -> None:
        """Pinned on its own so changing it is a deliberate one-line edit, and
        so the extension's copy of the value (sw.js, options.js) has something
        to be checked against."""
        self.assertEqual(DEFAULT_AUDIENCE, "teenager")

    def test_normalize_accepts_every_level(self) -> None:
        for lvl in AUDIENCE_LEVELS:
            self.assertEqual(normalize_audience(lvl), lvl)

    def test_normalize_defaults_and_trims_and_lowercases(self) -> None:
        self.assertEqual(normalize_audience(None), DEFAULT_AUDIENCE)
        self.assertEqual(normalize_audience(""), DEFAULT_AUDIENCE)
        self.assertEqual(normalize_audience("  CHILD "), "child")

    def test_normalize_rejects_unknown_and_names_the_options(self) -> None:
        with self.assertRaises(ValueError) as cm:
            normalize_audience("toddler")
        msg = str(cm.exception)
        for lvl in AUDIENCE_LEVELS:
            self.assertIn(lvl, msg)


class AudienceResolution(unittest.TestCase):
    def test_per_submission_wins(self) -> None:
        opts = PipelineOptions(audience="child")
        self.assertEqual(resolve_audience("adult", opts), "adult")

    def test_absent_falls_back_to_options(self) -> None:
        self.assertEqual(
            resolve_audience(None, PipelineOptions(audience="teenager")), "teenager")

    def test_unknown_request_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_audience("bogus", PipelineOptions())


# --- prompt composition ------------------------------------------------------


class AudienceClause(unittest.TestCase):
    def test_adult_contributes_nothing(self) -> None:
        from patmir.eval.candidates.translation.openrouter import (
            _SYSTEM, _audience_clause,
        )
        self.assertEqual(_audience_clause("adult"), "")
        self.assertEqual(_SYSTEM + _audience_clause("adult"), _SYSTEM)

    def test_absent_or_unknown_is_treated_as_adult(self) -> None:
        """The candidate is a Phase-0 object isolated from the pipeline's data
        contracts; validation lives at the edges (config loader, JobSpec, HTTP
        handlers), so an unknown value here degrades rather than raising in the
        middle of a run."""
        from patmir.eval.candidates.translation.openrouter import _audience_clause
        self.assertEqual(_audience_clause(None), "")
        self.assertEqual(_audience_clause(""), "")
        self.assertEqual(_audience_clause("bogus"), "")

    def test_child_and_teenager_are_distinct_and_non_empty(self) -> None:
        from patmir.eval.candidates.translation.openrouter import _audience_clause
        child, teen = _audience_clause("child"), _audience_clause("teenager")
        self.assertTrue(child.strip())
        self.assertTrue(teen.strip())
        self.assertNotEqual(child, teen)

    def test_case_and_whitespace_tolerant(self) -> None:
        from patmir.eval.candidates.translation.openrouter import _audience_clause
        self.assertEqual(_audience_clause(" CHILD "), _audience_clause("child"))

    def test_both_clauses_forbid_dropping_a_line(self) -> None:
        """Prompt-only softening is the whole design. If a future edit turns
        these clauses into redaction instructions, the listener gets a silent
        gap where a line should be — strictly worse than a gentler word, because
        the dub is the only audio track they have."""
        from patmir.eval.candidates.translation.openrouter import _audience_clause
        for level in ("child", "teenager"):
            clause = _audience_clause(level)
            with self.subTest(level=level):
                self.assertIn("Never delete a line", clause)
                self.assertIn("never bleep", clause.lower())
                self.assertIn("[...]", clause)
                self.assertIn("do not add a warning", clause.lower())
                self.assertIn("names", clause)

    def test_neither_clause_offers_redaction_vocabulary(self) -> None:
        """Deliberately never hand the model the words for censoring — that is
        how a safety-shaped refusal comes back as an empty line."""
        from patmir.eval.candidates.translation.openrouter import _audience_clause
        for level in ("child", "teenager"):
            clause = _audience_clause(level).lower()
            with self.subTest(level=level):
                self.assertNotIn("omit the line", clause)
                self.assertNotIn("skip the line", clause)
                self.assertNotIn("replace with [", clause)


class PerSentencePrompt(unittest.TestCase):
    def setUp(self) -> None:
        _set_env_key()

    def _translator(self, responses):
        from patmir.eval.candidates.translation.openrouter import OpenRouterTranslator
        from patmir.tests.test_translate_batch import FakeClient
        t = OpenRouterTranslator(model="google/gemini-3.5-flash")
        fake = FakeClient(responses)
        t._client = fake
        return t, fake

    def _system_of(self, fake, i=0) -> str:
        return fake.requests[i]["json"]["messages"][0]["content"]

    def _user_of(self, fake, i=0) -> str:
        return fake.requests[i]["json"]["messages"][1]["content"]

    def test_no_audience_is_the_bare_system_prompt(self) -> None:
        from patmir.eval.candidates.translation.openrouter import _SYSTEM
        from patmir.tests.test_translate_batch import _ok
        t, fake = self._translator([_ok("բարև")])
        t.translate("привет", source_lang="ru")
        self.assertEqual(self._system_of(fake), _SYSTEM)

    def test_child_appends_the_clause(self) -> None:
        from patmir.eval.candidates.translation.openrouter import (
            _SYSTEM, _audience_clause,
        )
        from patmir.tests.test_translate_batch import _ok
        t, fake = self._translator([_ok("բարև")])
        t.translate("привет", source_lang="ru", audience="child")
        self.assertEqual(
            self._system_of(fake), _SYSTEM + _audience_clause("child"))

    def test_audience_and_emotion_are_independent(self) -> None:
        """Two orthogonal per-call hints on different messages — the register
        clause on the system message, the emotion on the user message. Passing
        one must not disturb the other."""
        from patmir.eval.candidates.translation.openrouter import _audience_clause
        from patmir.tests.test_translate_batch import _ok
        t, fake = self._translator([_ok("բարև")])
        t.translate("привет", source_lang="ru", emotion="angry", audience="child")
        self.assertIn(_audience_clause("child"), self._system_of(fake))
        self.assertIn("angry", self._user_of(fake))


# --- adapter probing ---------------------------------------------------------


class _StubSentence:
    def __init__(self, idx: int, text: str, emotion: str = "neutral") -> None:
        self.idx = idx
        self.text = text
        self.start = float(idx)
        self.end = float(idx) + 1.0
        self.speaker_id = "SPEAKER_00"
        self.emotion = emotion


class _StubTranscript:
    language = "ru"

    def __init__(self, n: int = 2) -> None:
        self.sentences = [_StubSentence(i, f"ru_{i}") for i in range(n)]


class _RecordingCtx(JobContext):
    """A JobContext that captures progress lines instead of writing a file."""

    def __init__(self, job_dir: Path, audience: str) -> None:
        super().__init__(
            job_dir=job_dir,
            source_path=job_dir / "x.mkv",
            options=PipelineOptions(audience=audience),
            source_lang="ru",
        )
        self.lines: list[str] = []

    def progress(self, stage: str, pct: float, msg: str = "") -> None:
        self.lines.append(msg)


class _PlainInner:
    """No prompt surface at all — the NLLB / MADLAD / OPUS-MT shape."""
    id = "plain_inner"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def translate(self, text, source_lang):
        self.calls.append({"text": text, "source_lang": source_lang})
        return f"hy::{text}"


class _SteerableInner:
    id = "steerable_inner"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def translate(self, text, source_lang, emotion=None, audience=None):
        self.calls.append({
            "text": text, "source_lang": source_lang,
            "emotion": emotion, "audience": audience,
        })
        return f"hy::{text}"


class _PlainBatchInner:
    id = "plain_batch_inner"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def translate_batch(self, sentences, source_lang, *, progress=None):
        sents = list(sentences)
        self.calls.append({"source_lang": source_lang, "n": len(sents)})
        return {s.idx: f"hy::{s.text}" for s in sents}


class _SteerableBatchInner:
    id = "steerable_batch_inner"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def translate_batch(self, sentences, source_lang, *, progress=None, audience=None):
        sents = list(sentences)
        self.calls.append({"source_lang": source_lang, "audience": audience})
        return {s.idx: f"hy::{s.text}" for s in sents}


def _warnings(ctx) -> list[str]:
    return [m for m in ctx.lines if "NOT steered" in m]


class AdapterProbing(unittest.TestCase):
    """The adapter discovers the capability by signature, the same way it
    already does for `emotion` — candidates share no base class."""

    def _run(self, inner, audience: str):
        import tempfile
        from patmir.pipeline.stages.translate.adapter import SentenceTranslator
        with tempfile.TemporaryDirectory() as td:
            ctx = _RecordingCtx(Path(td), audience)
            out = SentenceTranslator(inner=inner).translate(_StubTranscript(), ctx)
            import json
            written = json.loads((Path(td) / "translated.json").read_text())
        return out, ctx, written

    def test_steerable_candidate_receives_the_level(self) -> None:
        inner = _SteerableInner()
        out, ctx, written = self._run(inner, "child")
        self.assertTrue(inner.calls)
        for c in inner.calls:
            self.assertEqual(c["audience"], "child")
        self.assertEqual(_warnings(ctx), [])
        self.assertTrue(written["audience_honored"])
        self.assertEqual(written["audience"], "child")

    def test_unsteerable_candidate_still_translates_and_warns_once(self) -> None:
        """Refusing would take out the offline fallback entirely; staying silent
        is the `emotion_guidance` failure. So: translate, and warn exactly
        once."""
        inner = _PlainInner()
        out, ctx, written = self._run(inner, "child")
        self.assertEqual(len(inner.calls), 2)  # both sentences still translated
        self.assertEqual(len(_warnings(ctx)), 1, ctx.lines)
        self.assertIn("plain_inner", _warnings(ctx)[0])
        self.assertIn("child", _warnings(ctx)[0])
        self.assertFalse(written["audience_honored"])

    def test_unsteerable_candidate_at_adult_does_not_warn(self) -> None:
        """Adult asks for nothing, so there is nothing to be unable to honour —
        warning here would cry wolf on every offline run."""
        inner = _PlainInner()
        _, ctx, _ = self._run(inner, "adult")
        self.assertEqual(_warnings(ctx), [])

    def test_batch_candidate_receives_the_level(self) -> None:
        inner = _SteerableBatchInner()
        _, ctx, written = self._run(inner, "teenager")
        self.assertEqual([c["audience"] for c in inner.calls], ["teenager"])
        self.assertEqual(_warnings(ctx), [])
        self.assertTrue(written["audience_honored"])

    def test_batch_candidate_without_the_kwarg_is_not_broken(self) -> None:
        """The batch branch probed nothing before this change, so passing the
        kwarg unconditionally would `TypeError` on every batch candidate that
        predates the guardrail."""
        inner = _PlainBatchInner()
        _, ctx, written = self._run(inner, "child")
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(len(_warnings(ctx)), 1, ctx.lines)
        self.assertFalse(written["audience_honored"])


class EmptyLineGuard(unittest.TestCase):
    """A provider that declines a line returns `content: null`, which becomes
    "" and is then skipped by the TTS adapter — a silent hole in the dub, and
    the most likely new failure a child-register prompt can introduce."""

    class _RefusingInner:
        id = "refusing_inner"

        def translate(self, text, source_lang, emotion=None, audience=None):
            return "" if text.endswith("1") else f"hy::{text}"

    def test_blank_results_are_counted_and_reported(self) -> None:
        import tempfile
        from patmir.pipeline.stages.translate.adapter import SentenceTranslator
        with tempfile.TemporaryDirectory() as td:
            ctx = _RecordingCtx(Path(td), "child")
            SentenceTranslator(inner=self._RefusingInner()).translate(
                _StubTranscript(), ctx)
        empty = [m for m in ctx.lines if "came back empty" in m]
        self.assertEqual(len(empty), 1, ctx.lines)
        self.assertIn("1 of 2", empty[0])
        self.assertIn("child", empty[0])

    def test_no_report_when_every_line_landed(self) -> None:
        import tempfile
        from patmir.pipeline.stages.translate.adapter import SentenceTranslator
        with tempfile.TemporaryDirectory() as td:
            ctx = _RecordingCtx(Path(td), "child")
            SentenceTranslator(inner=_SteerableInner()).translate(
                _StubTranscript(), ctx)
        self.assertEqual([m for m in ctx.lines if "came back empty" in m], [])

    class _DeadInner:
        """Every line comes back empty with a 200 — an outage that arrives
        without a status code (or from a candidate that has none)."""
        id = "dead_inner"

        def translate(self, text, source_lang, emotion=None, audience=None):
            return ""

    def test_every_line_empty_is_an_outage_not_a_report(self) -> None:
        """The 2026-09-13 shape: "12 of 12 line(s) came back empty" logged 226
        times over two days while the stage said `done`, so the TTS had nothing
        to say and the "dub" was the music bed plus the original vocals at the
        gap floor. All-empty must raise, and must NOT write `translated.json`
        as if the stage had succeeded (quality issue #35)."""
        import tempfile
        from patmir.pipeline.context import TranslatorUnavailable
        from patmir.pipeline.stages.translate.adapter import SentenceTranslator
        with tempfile.TemporaryDirectory() as td:
            ctx = _RecordingCtx(Path(td), "teenager")
            with self.assertRaises(TranslatorUnavailable) as cm:
                SentenceTranslator(inner=self._DeadInner()).translate(
                    _StubTranscript(n=12), ctx)
            self.assertFalse((Path(td) / "translated.json").exists())
        self.assertIn("12 lines", cm.exception.message)
        self.assertIn("teenager", cm.exception.message)
        self.assertEqual([m for m in ctx.lines if "came back empty" in m], [])

    def test_a_single_empty_line_is_still_only_reported(self) -> None:
        """The ADR's case survives: one line, one refusal, one reported gap —
        not a dead stream. `ALL_EMPTY_MIN_LINES` is the boundary."""
        import tempfile
        from patmir.pipeline.stages.translate.adapter import (
            ALL_EMPTY_MIN_LINES, SentenceTranslator,
        )
        self.assertEqual(ALL_EMPTY_MIN_LINES, 2)
        with tempfile.TemporaryDirectory() as td:
            ctx = _RecordingCtx(Path(td), "child")
            out = SentenceTranslator(inner=self._DeadInner()).translate(
                _StubTranscript(n=1), ctx)
            self.assertTrue((Path(td) / "translated.json").exists())
        self.assertEqual(out.sentences[0].target_text, "")
        empty = [m for m in ctx.lines if "came back empty" in m]
        self.assertEqual(len(empty), 1, ctx.lines)
        self.assertIn("1 of 1", empty[0])

    def test_batch_candidate_all_empty_is_an_outage_too(self) -> None:
        # The batch path builds target_text from `results.get(idx) or ""`, so
        # a dict of empty strings — exactly what the 402 storm produced — has
        # to hit the same rule.
        import tempfile
        from patmir.pipeline.context import TranslatorUnavailable
        from patmir.pipeline.stages.translate.adapter import SentenceTranslator

        class _DeadBatch:
            id = "dead_batch"

            def translate_batch(self, sentences, source_lang, progress=None,
                                audience=None):
                return {s.idx: "" for s in sentences}

        with tempfile.TemporaryDirectory() as td:
            ctx = _RecordingCtx(Path(td), "teenager")
            with self.assertRaises(TranslatorUnavailable):
                SentenceTranslator(inner=_DeadBatch()).translate(
                    _StubTranscript(n=3), ctx)


# --- submission validation ---------------------------------------------------


class JobSpecValidation(unittest.TestCase):
    def test_unknown_level_raises(self) -> None:
        from patmir.ui.jobs import JobSpec
        with self.assertRaises(ValueError):
            JobSpec(source_url="https://x/y", audience="toddler")

    def test_each_level_is_accepted_and_normalized(self) -> None:
        from patmir.ui.jobs import JobSpec
        for lvl in AUDIENCE_LEVELS:
            self.assertEqual(JobSpec(source_url="https://x/y", audience=lvl).audience, lvl)
        self.assertEqual(
            JobSpec(source_url="https://x/y", audience="CHILD").audience, "child")

    def test_absent_stays_none_so_the_server_default_applies(self) -> None:
        """None is meaningfully different from a level here: it means "the
        client did not choose", which is what lets `[pipeline.options].audience`
        be the default for the CLI while the extension always wins."""
        from patmir.ui.jobs import JobSpec
        self.assertIsNone(JobSpec(source_url="https://x/y").audience)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
