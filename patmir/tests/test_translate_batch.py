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

