"""Tests for `patmir.eval.candidates.translation.openrouter_batch`.

Pure-Python unit tests covering the contract pieces that can break
without an OpenRouter key:

  - chunk splitting math, including the last short chunk.
  - JSON parsing tolerates the model wrapping the object in prose.
  - `idx` set mismatch → fallback path is taken.
  - empty `text` → fallback path is taken.
  - happy path: chunk_size + overlap produces a well-formed prior-context
    field on the second call.

The HTTP layer is faked via a `httpx.Client` subclass so we never hit the
network.
"""
from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest.mock import patch


def _set_env_key():
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key")


class _FakeResp:
    def __init__(self, status_code: int, payload: dict | str):
        self.status_code = status_code
        if isinstance(payload, dict):
            self._json = payload
            self.text = json.dumps(payload)
        else:
            self._json = None
            self.text = payload

    def json(self) -> dict:
        if self._json is None:
            raise ValueError("not json")
        return self._json


def _ok(content: str) -> _FakeResp:
    return _FakeResp(200, {"choices": [{"message": {"content": content}}]})


def _json_response_for(idxes: list[int]) -> _FakeResp:
    body = {"translations": [{"idx": i, "text": f"hy_{i}"} for i in idxes]}
    return _ok(json.dumps(body))


def _items_from(idxes: list[int]):
    return [{"idx": i, "text": f"ru_{i}", "speaker_id": "SPEAKER_00", "emotion": "neutral"} for i in idxes]


class FakeClient:
    """Stand-in for the `httpx.Client` the candidate uses. Records every
    request and returns canned responses in order."""

    def __init__(self, responses: list[_FakeResp]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def post(self, url, *, headers, json):  # noqa: A002 — match httpx kwarg
        self.requests.append({"url": url, "headers": headers, "json": json})
        if not self.responses:
            raise RuntimeError("no more canned responses")
        return self.responses.pop(0)


class ChunkSplit(unittest.TestCase):
    def test_exact_multiple(self) -> None:
        from patmir.eval.candidates.translation.openrouter_batch import (
            _split_into_chunks, to_items,
        )
        items = to_items(_items_from(list(range(20))))
        chunks = _split_into_chunks(items, 10)
        self.assertEqual([len(c) for c in chunks], [10, 10])

    def test_short_last_chunk(self) -> None:
        from patmir.eval.candidates.translation.openrouter_batch import (
            _split_into_chunks, to_items,
        )
        items = to_items(_items_from(list(range(23))))
        chunks = _split_into_chunks(items, 10)
        self.assertEqual([len(c) for c in chunks], [10, 10, 3])

    def test_single_chunk(self) -> None:
        from patmir.eval.candidates.translation.openrouter_batch import (
            _split_into_chunks, to_items,
        )
        items = to_items(_items_from([0, 1, 2]))
        chunks = _split_into_chunks(items, 20)
        self.assertEqual([len(c) for c in chunks], [3])


class JsonParseTolerance(unittest.TestCase):
    def test_clean_object(self) -> None:
        from patmir.eval.candidates.translation.openrouter_batch import _parse_translations
        body = '{"translations":[{"idx":0,"text":"abc"},{"idx":1,"text":"def"}]}'
        arr = _parse_translations(body)
        self.assertEqual([t["idx"] for t in arr], [0, 1])

    def test_wrapped_in_prose(self) -> None:
        from patmir.eval.candidates.translation.openrouter_batch import _parse_translations
        body = 'Here you go: {"translations":[{"idx":7,"text":"x"}]}\n\nLet me know if you need anything else.'
        arr = _parse_translations(body)
        self.assertEqual(arr, [{"idx": 7, "text": "x"}])

    def test_no_translations_array(self) -> None:
        from patmir.eval.candidates.translation.openrouter_batch import (
            _parse_translations, _BatchError,
        )
        with self.assertRaises(_BatchError):
            _parse_translations('{"results":[]}')

    def test_no_json_at_all(self) -> None:
        from patmir.eval.candidates.translation.openrouter_batch import (
            _parse_translations, _BatchError,
        )
        with self.assertRaises(_BatchError):
            _parse_translations("Sorry, I can't help with that.")


class BatchTranslateE2E(unittest.TestCase):
    def setUp(self) -> None:
        _set_env_key()

    def _build(self, fake: FakeClient, **kwargs):
        from patmir.eval.candidates.translation.openrouter_batch import OpenRouterBatchTranslator
        t = OpenRouterBatchTranslator(
            model="google/gemini-3.5-flash",
            chunk_size=kwargs.pop("chunk_size", 3),
            overlap=kwargs.pop("overlap", 1),
            **kwargs,
        )
        t._client = fake
        # Make the per-sentence fallback also use the fake so we don't hit
        # the network when triggering it.
        t._fallback._client = fake
        return t

    def test_happy_path_single_chunk(self) -> None:
        fake = FakeClient([_json_response_for([0, 1, 2])])
        t = self._build(fake)
        out = t.translate_batch(_items_from([0, 1, 2]), source_lang="ru")
        self.assertEqual(out, {0: "hy_0", 1: "hy_1", 2: "hy_2"})
        self.assertEqual(len(fake.requests), 1)

    def test_two_chunks_with_overlap_context(self) -> None:
        # 4 sentences, chunk_size=3, overlap=1.
        # Chunk 1: [0,1,2]   → translated.
        # Chunk 2: [3]       → with prior=[(item2, "hy_2")] as context.
        fake = FakeClient([
            _json_response_for([0, 1, 2]),
            _json_response_for([3]),
        ])
        t = self._build(fake, chunk_size=3, overlap=1)
        out = t.translate_batch(_items_from([0, 1, 2, 3]), source_lang="ru")
        self.assertEqual(out, {0: "hy_0", 1: "hy_1", 2: "hy_2", 3: "hy_3"})

        # Inspect the second request: it must carry the overlap context.
        self.assertEqual(len(fake.requests), 2)
        second_payload = fake.requests[1]["json"]
        user_content = second_payload["messages"][1]["content"]
        # Context block contains the *prior* sentence (idx=2) + its target.
        self.assertIn('"idx": 2', user_content)
        self.assertIn('"text_target": "hy_2"', user_content)
        # And the second chunk's input is idx=3.
        self.assertIn('"idx": 3', user_content)

    def test_idx_mismatch_falls_back_per_sentence(self) -> None:
        # First chunk returns the wrong idx set — should fall back to per-
        # sentence. Per-sentence path goes through the same fake client
        # (we routed t._fallback._client = fake in _build) and we need to
        # supply one response per sentence in the chunk.
        fake = FakeClient([
            _json_response_for([99, 100, 101]),       # mismatch — triggers fallback
            _ok("ՀՅ-0"), _ok("ՀՅ-1"), _ok("ՀՅ-2"),  # per-sentence fallback
        ])
        t = self._build(fake, chunk_size=3, overlap=0)
        out = t.translate_batch(_items_from([0, 1, 2]), source_lang="ru")
        self.assertEqual(out, {0: "ՀՅ-0", 1: "ՀՅ-1", 2: "ՀՅ-2"})

    def test_empty_text_triggers_fallback(self) -> None:
        empty_resp = _ok(json.dumps({
            "translations": [
                {"idx": 0, "text": "hy_0"},
                {"idx": 1, "text": ""},   # empty triggers _BatchError → fallback
                {"idx": 2, "text": "hy_2"},
            ],
        }))
        fake = FakeClient([
            empty_resp,
            _ok("a"), _ok("b"), _ok("c"),
        ])
        t = self._build(fake, chunk_size=3, overlap=0)
        out = t.translate_batch(_items_from([0, 1, 2]), source_lang="ru")
        self.assertEqual(out, {0: "a", 1: "b", 2: "c"})

    def test_progress_callback_fires_per_sentence(self) -> None:
        fake = FakeClient([_json_response_for([0, 1, 2])])
        t = self._build(fake)
        ticks = []
        t.translate_batch(
            _items_from([0, 1, 2]), source_lang="ru",
            progress=lambda done, total, msg: ticks.append((done, total)),
        )
        self.assertEqual(ticks, [(1, 3), (2, 3), (3, 3)])


if __name__ == "__main__":
    unittest.main()


class AudienceBatchPrompt(unittest.TestCase):
    """The batch candidate is what watch-along runs, and it shares `_SYSTEM`
    with the per-sentence one by import — so the register clause must reach it
    through the same import rather than being duplicated."""

    def setUp(self) -> None:
        _set_env_key()

    def _build(self, fake: FakeClient, **kwargs):
        from patmir.eval.candidates.translation.openrouter_batch import OpenRouterBatchTranslator
        t = OpenRouterBatchTranslator(
            model="google/gemini-3.5-flash",
            chunk_size=kwargs.pop("chunk_size", 3),
            overlap=kwargs.pop("overlap", 1),
            **kwargs,
        )
        t._client = fake
        t._fallback._client = fake
        return t

    def _system_of(self, fake: FakeClient, i: int = 0) -> str:
        return fake.requests[i]["json"]["messages"][0]["content"]

    def test_adult_system_prompt_is_unchanged(self) -> None:
        from patmir.eval.candidates.translation.openrouter import _SYSTEM
        from patmir.eval.candidates.translation.openrouter_batch import (
            _BATCH_SYSTEM_SUFFIX,
        )
        fake = FakeClient([_json_response_for([0, 1, 2])])
        t = self._build(fake)
        t.translate_batch(_items_from([0, 1, 2]), source_lang="ru")
        self.assertEqual(
            self._system_of(fake), _SYSTEM + "\n\n" + _BATCH_SYSTEM_SUFFIX,
        )

    def test_no_audience_argument_matches_adult(self) -> None:
        """Callers that never heard of the guardrail (the eval harness) keep the
        exact prompt they had."""
        fake_a = FakeClient([_json_response_for([0, 1, 2])])
        self._build(fake_a).translate_batch(_items_from([0, 1, 2]), source_lang="ru")
        fake_b = FakeClient([_json_response_for([0, 1, 2])])
        self._build(fake_b).translate_batch(
            _items_from([0, 1, 2]), source_lang="ru", audience="adult",
        )
        self.assertEqual(self._system_of(fake_a), self._system_of(fake_b))

    def test_child_clause_sits_between_system_and_batch_suffix(self) -> None:
        """Order matters: register guidance belongs with `_SYSTEM`, and the
        JSON-shape rules stay last where the model weights them most."""
        from patmir.eval.candidates.translation.openrouter import (
            _SYSTEM, _audience_clause,
        )
        from patmir.eval.candidates.translation.openrouter_batch import (
            _BATCH_SYSTEM_SUFFIX,
        )
        fake = FakeClient([_json_response_for([0, 1, 2])])
        t = self._build(fake)
        t.translate_batch(_items_from([0, 1, 2]), source_lang="ru", audience="child")
        content = self._system_of(fake)
        clause = _audience_clause("child")
        self.assertTrue(content.startswith(_SYSTEM))
        self.assertIn(clause, content)
        self.assertTrue(content.endswith(_BATCH_SYSTEM_SUFFIX))
        self.assertLess(content.index(clause), content.index(_BATCH_SYSTEM_SUFFIX))

    def test_fallback_chunk_carries_the_same_audience(self) -> None:
        """A chunk whose response fails validation degrades to per-sentence.
        If the level did not ride along, one malformed response would silently
        revert those lines to the adult prompt — which on a child submission is
        the whole failure this feature exists to prevent."""
        from patmir.eval.candidates.translation.openrouter import _audience_clause
        # First response has the wrong idx set -> _BatchError -> per-sentence.
        fake = FakeClient([
            _json_response_for([99]),
            _ok("hy_0"), _ok("hy_1"), _ok("hy_2"),
        ])
        t = self._build(fake)
        out = t.translate_batch(
            _items_from([0, 1, 2]), source_lang="ru", audience="child",
        )
        self.assertEqual(out, {0: "hy_0", 1: "hy_1", 2: "hy_2"})
        self.assertEqual(len(fake.requests), 4)
        clause = _audience_clause("child")
        for i in range(1, 4):  # the three per-sentence retries
            self.assertIn(clause, self._system_of(fake, i), f"request {i}")



class RetryPolicy(unittest.TestCase):
    """Transient failures are retried; permanent ones are not.

    Before 2026-09-02 the per-sentence candidate raised any non-200 straight
    out of the adapter's thread pool, failing the whole translate stage and
    discarding every sentence already translated in that job — for one 429.
    The batched candidate did not fail the stage but degraded the chunk to
    `chunk_size` per-sentence round-trips, which is the opposite of what a rate
    limiter wants.
    """

    def _sentence_translator(self, responses):
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter import (
            OpenRouterTranslator,
        )
        t = OpenRouterTranslator(model="m", max_attempts=3, retry_backoff_s=0.0)
        client = FakeClient(responses)
        t._client = client
        return t, client

    def test_429_is_retried_then_succeeds(self):
        t, client = self._sentence_translator(
            [_FakeResp(429, "slow down"), _ok("բարև")])
        out = t.translate("privet", source_lang="ru")
        self.assertEqual(out, "բարև")
        self.assertEqual(len(client.requests), 2)

    def test_provider_500_is_retried(self):
        t, client = self._sentence_translator(
            [_FakeResp(500, "boom"), _FakeResp(503, "nope"), _ok("ok")])
        self.assertEqual(t.translate("x", source_lang="ru"), "ok")
        self.assertEqual(len(client.requests), 3)

    def test_400_is_not_retried(self):
        # A bad request fails identically next time; retrying only delays the
        # real error.
        t, client = self._sentence_translator([_FakeResp(400, "bad model")])
        with self.assertRaises(RuntimeError):
            t.translate("x", source_lang="ru")
        self.assertEqual(len(client.requests), 1)

    def test_attempts_are_bounded(self):
        t, client = self._sentence_translator([_FakeResp(429, "no")] * 5)
        with self.assertRaises(RuntimeError):
            t.translate("x", source_lang="ru")
        self.assertEqual(len(client.requests), 3)      # max_attempts

    def test_transport_error_is_retried(self):
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter import (
            OpenRouterTranslator,
        )
        import httpx
        t = OpenRouterTranslator(model="m", max_attempts=2, retry_backoff_s=0.0)
        calls = {"n": 0}

        class _Flaky:
            def post(self, url, *, headers, json):  # noqa: A002
                calls["n"] += 1
                if calls["n"] == 1:
                    raise httpx.ConnectError("dropped")
                return _ok("recovered")

        t._client = _Flaky()
        self.assertEqual(t.translate("x", source_lang="ru"), "recovered")
        self.assertEqual(calls["n"], 2)

    def test_batch_chunk_retries_before_falling_back(self):
        # One 429 must not turn a single batched call into chunk_size
        # per-sentence ones.
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter_batch import (
            OpenRouterBatchTranslator,
        )
        t = OpenRouterBatchTranslator(model="m", chunk_size=20, overlap=0,
                                      max_attempts=3, retry_backoff_s=0.0)
        client = FakeClient([_FakeResp(429, "slow"),
                             _json_response_for([0, 1, 2])])
        t._client = client
        out = t.translate_batch(_items_from([0, 1, 2]), source_lang="ru")
        self.assertEqual(out, {0: "hy_0", 1: "hy_1", 2: "hy_2"})
        self.assertEqual(len(client.requests), 2)      # retried, not fell back

    def test_batch_permanent_error_still_falls_back(self):
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter_batch import (
            OpenRouterBatchTranslator,
        )
        t = OpenRouterBatchTranslator(model="m", chunk_size=20, overlap=0,
                                      max_attempts=3, retry_backoff_s=0.0)
        t._client = FakeClient([_FakeResp(400, "bad")])
        calls = []

        class _Fallback:
            translate_concurrency = 1

            def translate(self, text, *, source_lang, emotion=None,
                          audience=None):
                calls.append(text)
                return f"fb_{text}"

        t._fallback = _Fallback()
        out = t.translate_batch(_items_from([0, 1]), source_lang="ru")
        self.assertEqual(out, {0: "fb_ru_0", 1: "fb_ru_1"})
        self.assertEqual(len(calls), 2)


class OutagePolicy(unittest.TestCase):
    """A translator that cannot translate ANY line raises, it does not degrade.

    On 2026-09-13 the OpenRouter account ran out of credits. Every call
    answered 402; 402 was not retryable, so the batched candidate raised
    `_BatchError`, fell back to per-sentence calls, each of those 402'd too and
    `_fallback_chunk` turned each into "". `translate_batch` returned a dict of
    empty strings, the adapter logged "12 of 12 line(s) came back empty" and
    the stage said `done`. Two days of watch-along sections were the music bed
    plus the original vocals at the gap floor (quality issue #35). These tests
    pin the boundary: 401/402/404 raise `TranslatorUnavailable` from the first
    response, with no per-sentence storm; a 403 (per-input moderation) is still
    a per-line matter and still falls back.
    """

    def _sentence_translator(self, responses):
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter import (
            OpenRouterTranslator,
        )
        t = OpenRouterTranslator(model="m", max_attempts=3, retry_backoff_s=0.0)
        client = FakeClient(responses)
        t._client = client
        return t, client

    def _batch_translator(self, responses):
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter_batch import (
            OpenRouterBatchTranslator,
        )
        t = OpenRouterBatchTranslator(model="m", chunk_size=20, overlap=0,
                                      max_attempts=3, retry_backoff_s=0.0)
        client = FakeClient(responses)
        t._client = client
        return t, client

    def test_402_is_unavailable_on_the_sentence_candidate(self):
        from patmir.pipeline.context import TranslatorUnavailable
        t, client = self._sentence_translator(
            [_FakeResp(402, '{"error":{"message":"Insufficient credits"}}')])
        with self.assertRaises(TranslatorUnavailable) as cm:
            t.translate("x", source_lang="ru")
        self.assertEqual(cm.exception.status, 402)
        self.assertIn("402", cm.exception.message)
        self.assertIn("credits exhausted", cm.exception.message)
        self.assertIn("Insufficient credits", cm.exception.message)
        # Not retried: it fails identically next time.
        self.assertEqual(len(client.requests), 1)

    def test_401_and_404_are_unavailable_too(self):
        from patmir.pipeline.context import TranslatorUnavailable
        for status in (401, 404):
            with self.subTest(status=status):
                t, client = self._sentence_translator([_FakeResp(status, "no")])
                with self.assertRaises(TranslatorUnavailable) as cm:
                    t.translate("x", source_lang="ru")
                self.assertEqual(cm.exception.status, status)
                self.assertEqual(len(client.requests), 1)

    def test_the_message_is_one_readable_line(self):
        # The extension prints `error` under the button; a traceback there is
        # noise. Multi-line provider bodies are flattened and truncated.
        from patmir.pipeline.context import TranslatorUnavailable
        t, _ = self._sentence_translator([_FakeResp(402, "line one\nline two\n" + "x" * 500)])
        with self.assertRaises(TranslatorUnavailable) as cm:
            t.translate("x", source_lang="ru")
        self.assertNotIn("\n", cm.exception.message)
        self.assertLess(len(cm.exception.message), 320)

    def test_400_is_still_a_plain_error(self):
        # Usually our own payload; not an outage, not `TranslatorUnavailable`.
        from patmir.pipeline.context import TranslatorUnavailable
        t, _ = self._sentence_translator([_FakeResp(400, "bad")])
        with self.assertRaises(RuntimeError) as cm:
            t.translate("x", source_lang="ru")
        self.assertNotIsInstance(cm.exception, TranslatorUnavailable)

    def test_batch_402_raises_without_the_per_sentence_storm(self):
        """The measured shape of the incident: 1 chunk call + 12 fallback calls
        for a 12-sentence section, every one a 402. Now: one call, one raise."""
        from patmir.pipeline.context import TranslatorUnavailable
        t, client = self._batch_translator([_FakeResp(402, "Insufficient credits")])
        fallback_calls: list[str] = []

        class _Fallback:
            translate_concurrency = 12

            def translate(self, text, *, source_lang, emotion=None, audience=None):
                fallback_calls.append(text)
                return ""

        t._fallback = _Fallback()
        with self.assertRaises(TranslatorUnavailable) as cm:
            t.translate_batch(_items_from(list(range(12))), source_lang="ru")
        self.assertEqual(cm.exception.status, 402)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(fallback_calls, [], "fell back per sentence on an outage")

    def test_batch_403_is_per_line_and_still_falls_back(self):
        # OpenRouter answers 403 when the provider's moderation flags one
        # input. That is the refusal the audience ADR routes to an empty line,
        # so it must keep degrading to per-sentence calls, not fail the dub.
        t, client = self._batch_translator([_FakeResp(403, "flagged")])
        calls: list[str] = []

        class _Fallback:
            translate_concurrency = 1

            def translate(self, text, *, source_lang, emotion=None, audience=None):
                calls.append(text)
                return f"fb_{text}"

        t._fallback = _Fallback()
        out = t.translate_batch(_items_from([0, 1]), source_lang="ru")
        self.assertEqual(out, {0: "fb_ru_0", 1: "fb_ru_1"})
        self.assertEqual(len(calls), 2)

    def test_fallback_reraises_unavailable_but_blanks_other_failures(self):
        """`_fallback_chunk._one` catches everything and returns "" — that is
        the per-line refusal contract. An outage raised INSIDE the fallback
        (the chunk call failed for another reason, then the credits ran out)
        must escape it; any other exception still becomes an empty line."""
        from patmir.pipeline.context import TranslatorUnavailable

        for exc, expect_raise in ((TranslatorUnavailable("gone", status=402), True),
                                  (RuntimeError("one bad line"), False)):
            with self.subTest(exc=type(exc).__name__):
                t, _ = self._batch_translator([_FakeResp(400, "bad")])  # -> fallback

                class _Fallback:
                    translate_concurrency = 1

                    def translate(self, text, *, source_lang, emotion=None,
                                  audience=None):
                        raise exc

                t._fallback = _Fallback()
                if expect_raise:
                    with self.assertRaises(TranslatorUnavailable):
                        t.translate_batch(_items_from([0, 1]), source_lang="ru")
                else:
                    out = t.translate_batch(_items_from([0, 1]), source_lang="ru")
                    self.assertEqual(out, {0: "", 1: ""})

    def test_fallback_reraises_unavailable_when_concurrent(self):
        # The thread-pool path: `ex.map` must surface the worker's raise.
        from patmir.pipeline.context import TranslatorUnavailable
        t, _ = self._batch_translator([_FakeResp(400, "bad")])

        class _Fallback:
            translate_concurrency = 4

            def translate(self, text, *, source_lang, emotion=None, audience=None):
                raise TranslatorUnavailable("gone", status=402)

        t._fallback = _Fallback()
        with self.assertRaises(TranslatorUnavailable):
            t.translate_batch(_items_from([0, 1, 2, 3]), source_lang="ru")

    def test_shared_fatal_set(self):
        # One outage policy for both candidates, like the retry set.
        from patmir.eval.candidates.translation import openrouter, openrouter_batch
        self.assertEqual(openrouter_batch._FATAL, openrouter.OpenRouterTranslator._FATAL)
        self.assertEqual(openrouter.OpenRouterTranslator._FATAL, frozenset({401, 402, 404}))
        self.assertNotIn(403, openrouter.OpenRouterTranslator._FATAL)
        self.assertFalse(openrouter.OpenRouterTranslator._FATAL
                         & openrouter.OpenRouterTranslator._RETRYABLE)


class ConcurrencyKnob(unittest.TestCase):
    def test_concurrency_is_configurable_per_instance(self):
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter import (
            OpenRouterTranslator,
            build,
        )
        self.assertEqual(OpenRouterTranslator(model="m").translate_concurrency, 6)
        self.assertEqual(
            OpenRouterTranslator(model="m", concurrency=12).translate_concurrency, 12)
        # ...and reachable from patmir.toml, which is the point: the number
        # that decides ~13% of batch wall was previously code-only.
        self.assertEqual(
            build({"model": "m", "concurrency": 16}).translate_concurrency, 16)

    def test_concurrency_floor_is_one(self):
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter import (
            OpenRouterTranslator,
        )
        self.assertEqual(
            OpenRouterTranslator(model="m", concurrency=0).translate_concurrency, 1)

    def test_fallback_chunk_runs_concurrently(self):
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter_batch import (
            OpenRouterBatchTranslator,
        )
        t = OpenRouterBatchTranslator(model="m", chunk_size=20, overlap=0)
        import threading
        peak = {"n": 0, "cur": 0}
        lock = threading.Lock()
        gate = threading.Barrier(3, timeout=5)

        class _Fallback:
            translate_concurrency = 4

            def translate(self, text, *, source_lang, emotion=None,
                          audience=None):
                with lock:
                    peak["cur"] += 1
                    peak["n"] = max(peak["n"], peak["cur"])
                # Blocks until three calls are in flight — the assertion IS
                # that they overlap; a serial implementation deadlocks here.
                gate.wait()
                with lock:
                    peak["cur"] -= 1
                return f"fb_{text}"

        t._fallback = _Fallback()
        t._client = FakeClient([_FakeResp(400, "bad")])
        out = t.translate_batch(_items_from([0, 1, 2]), source_lang="ru")
        self.assertEqual(len(out), 3)
        self.assertGreaterEqual(peak["n"], 3)

    def test_fallback_stays_serial_when_the_inner_is_serial(self):
        _set_env_key()
        from patmir.eval.candidates.translation.openrouter_batch import (
            OpenRouterBatchTranslator,
        )
        t = OpenRouterBatchTranslator(model="m", chunk_size=20, overlap=0)
        order = []

        class _Fallback:
            translate_concurrency = 1

            def translate(self, text, *, source_lang, emotion=None,
                          audience=None):
                order.append(text)
                return "x"

        t._fallback = _Fallback()
        t._client = FakeClient([_FakeResp(400, "bad")])
        t.translate_batch(_items_from([0, 1, 2]), source_lang="ru")
        self.assertEqual(order, ["ru_0", "ru_1", "ru_2"])


class CacheKeyIgnoresSchedulingKnobs(unittest.TestCase):
    """Tuning throughput must not invalidate a library of finished dubs.

    `config_hash` hashes every key of a candidate entry, on the sound reasoning
    that a model id, a voice, a dtype or a sampling knob all change output. But
    concurrency and retry policy do not, and putting them in `patmir.toml` —
    which is the only way to make them tunable — would otherwise re-dub
    everything for a change that cannot alter a sample.
    """

    def _cfg(self, tts_entry=None, translate_entry=None):
        from patmir.pipeline.stage import STAGE_ORDER

        class _Cfg:
            entries = {name: {"id": f"c_{name}"} for name in STAGE_ORDER}

        c = _Cfg()
        c.entries = dict(c.entries)
        if tts_entry:
            c.entries["tts"] = {**c.entries["tts"], **tts_entry}
        if translate_entry:
            c.entries["translate"] = {**c.entries["translate"], **translate_entry}
        return c

    def test_concurrency_and_retries_do_not_move_the_key(self):
        from patmir.ui.cache import config_hash
        base = config_hash(self._cfg())
        for entry in ({"concurrency": 12}, {"max_attempts": 5},
                      {"retry_backoff_s": 2.0},
                      {"concurrency": 16, "max_attempts": 1}):
            self.assertEqual(config_hash(self._cfg(translate_entry=entry)), base,
                             f"{entry} changed the cache key")

    def test_cuda_graphs_DOES_move_the_key(self):
        """Inverted 2026-09-07. This test previously asserted the opposite, on
        the 2026-09-02 claim that graph replay is byte-identical to eager
        stepping (md5-for-md5 on 80 WAVs). **That claim is false for the decode
        loop as it runs**, and this test was pinning it in place.

        Measured — 24 real sentences, one pinned reference, separate processes:

            graphs ON  vs ON  (repeat): identical 24/24   <- control
            graphs OFF vs OFF (repeat): identical 24/24   <- control
            graphs ON  vs OFF:          identical  0/24

        The controls establish determinism across processes, so the flag alone
        accounts for it, and `graph_captures` read 2 vs 0. The original check
        compared `StaticCache` against `DynamicCache` and replay against eager
        in isolation, never exercising the mask width the real loop uses:
        the graph attends over `seq + max_cap` columns against the fallback's
        `seq + step + 1`, so the softmax reduces a different width.
        """
        from patmir.ui.cache import config_hash
        base = config_hash(self._cfg())
        self.assertNotEqual(
            config_hash(self._cfg(tts_entry={"cuda_graphs": True})), base,
            "cuda_graphs re-renders every line; excluding it from the key lets "
            "one key hold two different renderings")

    def test_decode_attention_DOES_move_the_key(self):
        # The near miss: also "just" a kernel choice, but its reduction order
        # differs and it re-rolls every line, so cached dubs must regenerate.
        from patmir.ui.cache import config_hash
        base = config_hash(self._cfg())
        self.assertNotEqual(
            config_hash(self._cfg(tts_entry={"decode_attention": "gqa"})), base)

    def test_output_affecting_knobs_still_move_the_key(self):
        from patmir.ui.cache import config_hash
        base = config_hash(self._cfg())
        for entry in ({"model": "other"}, {"temperature": 0.9}, {"seed": 7},
                      {"max_tokens": 1024}, {"dtype": "float16"},
                      {"max_batch": 12}):
            self.assertNotEqual(config_hash(self._cfg(tts_entry=entry)), base,
                                f"{entry} must change the cache key")


class BuiltFallbackKnobsTest(unittest.TestCase):
    """`build()` must hand the throughput/retry knobs to the per-sentence
    fallback — the candidate itself makes one call per chunk, so the fallback
    is the only place `concurrency` acts, and watch-along runs THIS candidate.
    Before this the fallback was constructed with the class defaults whatever
    `patmir.toml` said, so the 6 -> 12 tuning never reached watch-along."""

    def test_build_forwards_knobs_to_the_fallback(self):
        from patmir.eval.candidates.translation import openrouter_batch
        t = openrouter_batch.build({
            "model": "m", "concurrency": 12, "max_attempts": 4, "retry_backoff_s": 0.7,
        })
        self.assertEqual(t._fallback.translate_concurrency, 12)
        self.assertEqual(t._fallback.max_attempts, 4)
        self.assertEqual(t._fallback.retry_backoff_s, 0.7)
        self.assertEqual(t.max_attempts, 4)

    def test_the_toml_entry_carries_the_knobs(self):
        import tomllib
        from pathlib import Path
        cfg = tomllib.loads((Path(__file__).resolve().parents[2] / "patmir.toml").read_text())
        entry = next(c for c in cfg["eval"]["translation"]["candidates"]
                     if c["id"] == "openrouter_gemini_flash_batch")
        self.assertEqual(entry["concurrency"], 12)
        self.assertIn("max_attempts", entry)
