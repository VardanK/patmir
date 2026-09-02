"""Tests for the batched decode in the torch Higgs TTS candidate.

Batching is the largest available win on the TTS stage (~69% of batch wall): the
decode is memory-bandwidth-bound on 4B weights, so reading them once to advance
B sentences is close to free. Upstream's `generate_speech` is batch-1, so the
loop is ours — which means the batching mechanics are ours to get right.

Two layers of testing, because the risks are different:

* **Equivalence, on a real model.** A tiny randomly-initialized
  `HiggsMultimodalQwen3` on CPU, driven through the actual decode path. This is
  what catches the mechanical mistakes that batching invites: left padding
  corrupting a row, `position_ids` drifting from the mask so RoPE sees the wrong
  positions, or a finished row's masking leaking into a live one. A row decoded
  inside a batch must equal the same row decoded alone. These tests skip when
  the checkpoint's snapshot isn't cached locally — they need its config and
  tokenizer, not its weights.

* **Orchestration, on stubs.** Chunking, cap-sorting, output ordering, the
  overflow retry and the silence fallback are pure bookkeeping around the decode
  and are tested without a model at all. Ordering is the one worth stating
  plainly: requests are REORDERED for batching efficiency, so an off-by-one in
  the index mapping would hand sentences the wrong audio — an error that would
  never crash, just quietly desynchronize the dub.
"""
from __future__ import annotations

import os
import unittest
import wave
from pathlib import Path

from patmir.eval.candidates.tts import higgs_v3_torch
from patmir.eval.candidates.tts.higgs_v3_torch import (
    SR,
    HiggsV3Torch,
    _balanced_chunks,
    _Request,
    _request_seed,
    _sample_codes,
    token_cap,
)

_REPO_CACHE_DIR = "models--multimodalart--higgs-audio-v3-tts-4b-transformers"


def _hub_cache() -> Path:
    """The resolved HuggingFace hub cache directory.

    Must honour `HF_HUB_CACHE` / `HF_HOME` rather than assuming
    `~/.cache/huggingface`. Hardcoding the default meant the CUDA workstation —
    whose cache lives on a separate drive — silently SKIPPED every test in
    `BatchEquivalenceTest`, i.e. the ones that prove the batched decode is
    correct, on the only machine where that decode actually runs. A skip is not a
    pass, and eight of them hid behind a green summary.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 - fall back to the documented env precedence
        if os.environ.get("HF_HUB_CACHE"):
            return Path(os.environ["HF_HUB_CACHE"])
        if os.environ.get("HF_HOME"):
            return Path(os.environ["HF_HOME"]) / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"


def _snapshot() -> Path | None:
    snaps = _hub_cache() / _REPO_CACHE_DIR / "snapshots"
    if not snaps.is_dir():
        return None
    for child in sorted(snaps.iterdir()):
        if (child / "config.json").exists() and (child / "tokenizer.json").exists():
            return child
    return None


def _upstream_module():
    """The checkpoint's own `modeling_higgs_multimodal_qwen3` module.

    Resolved through transformers' dynamic-module machinery — the same path the
    candidate takes (`sys.modules[type(self.model).__module__]`) — so the state
    machine used below as an oracle is the one the model would actually run,
    not a second transcription of it.
    """
    import sys

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    snap = _snapshot()
    if snap is None:
        raise RuntimeError("Higgs checkpoint snapshot not in the HF cache")
    cls = get_class_from_dynamic_module(
        "modeling_higgs_multimodal_qwen3.HiggsMultimodalQwen3ForConditionalGeneration",
        str(snap),
    )
    return sys.modules[cls.__module__]


def _tiny_model(snapshot: str):
    """A 2-layer randomly-initialized HiggsMultimodalQwen3 on CPU in float32.

    Small enough to build in a unit test, real enough to exercise the actual
    decode path and the checkpoint's own tied-weights declaration.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
    tc = cfg.text_config
    tc.hidden_size = 64
    tc.num_hidden_layers = 2
    tc.num_attention_heads = 2
    tc.num_key_value_heads = 1
    tc.head_dim = 32
    tc.intermediate_size = 128
    cfg.audio_encoder_config = dict(cfg.audio_encoder_config)
    cfg.audio_encoder_config["out_dim"] = 64
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).to(torch.float32).eval()


def _bare(max_batch: int = 8, max_retries: int = 0) -> HiggsV3Torch:
    """A candidate with no model, for testing the orchestration around decode."""
    obj = object.__new__(HiggsV3Torch)
    obj.tts_max_batch = max_batch
    obj.max_retries = max_retries
    obj.base_seed = 0
    obj._ref_cache = {}
    obj._summary = {"sentences": 0, "overflows": 0, "batches": 0}
    return obj


def _wav_duration(data: bytes) -> float:
    import io
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


class TokenCapTest(unittest.TestCase):
    """The cap bounds a non-terminating run AND is the batch sort key, so its
    monotonicity in text length is load-bearing twice over."""

    def test_floor_and_ceiling(self):
        self.assertEqual(token_cap(""), 200)
        self.assertEqual(token_cap("x" * 10_000), 1024)

    def test_proportional_between_the_bounds(self):
        self.assertEqual(token_cap("x" * 100), 400)

    def test_monotonic_in_length(self):
        caps = [token_cap("x" * n) for n in (10, 60, 120, 200, 400)]
        self.assertEqual(caps, sorted(caps))


class ChunksTest(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(_balanced_chunks([0, 1, 2, 3], 2), [[0, 1], [2, 3]])

    def test_ragged_tail(self):
        self.assertEqual(_balanced_chunks([0, 1, 2], 2), [[0, 1], [2]])

    def test_batch_larger_than_input(self):
        self.assertEqual(_balanced_chunks([0], 8), [[0]])

    def test_empty(self):
        self.assertEqual(_balanced_chunks([], 4), [])

    # --- the tail defect (spec 2026-08-25-1931, finding B1) ----------------

    def test_nine_items_do_not_leave_a_batch_of_one(self):
        """The measured case. A chunk costs max(steps), so a trailing chunk of
        one is the single longest line paying for a whole chunk: 17.3 s of the
        prelude's 41.6 s TTS stage, for one sentence."""
        sizes = [len(c) for c in _balanced_chunks(list(range(9)), 8)]
        self.assertEqual(sizes, [5, 4])

    def test_never_exceeds_the_batch_ceiling(self):
        """Balancing must not buy speed with memory — max_batch is a VRAM
        bound, and this box already peaks at 16.4 GB of 24."""
        for n in range(1, 40):
            for size in (1, 2, 4, 8):
                for c in _balanced_chunks(list(range(n)), size):
                    self.assertLessEqual(len(c), size, f"n={n} size={size}")

    def test_uses_the_fewest_possible_chunks(self):
        for n in range(1, 40):
            for size in (1, 2, 4, 8):
                got = len(_balanced_chunks(list(range(n)), size))
                self.assertEqual(got, -(-n // size), f"n={n} size={size}")

    def test_partitions_the_input_in_order(self):
        """Contiguity in cap-sorted order is what makes batching pay, and the
        caller maps results back by slot — losing or reordering an item would
        give a sentence someone else's audio."""
        for n in range(0, 40):
            items = list(range(n))
            flat = [i for c in _balanced_chunks(items, 8) for i in c]
            self.assertEqual(flat, items, f"n={n}")

    def test_chunk_sizes_differ_by_at_most_one(self):
        for n in range(1, 40):
            for size in (2, 4, 8):
                sizes = [len(c) for c in _balanced_chunks(list(range(n)), size)]
                self.assertLessEqual(max(sizes) - min(sizes), 1, f"n={n} size={size}")

    def test_matches_greedy_when_the_size_divides_evenly(self):
        """The common full-batch case must be untouched, so this change cannot
        move audio for a job whose sentence count is already a multiple."""
        for n, size in ((8, 8), (16, 8), (24, 8), (4, 2)):
            greedy = [list(range(i, i + size)) for i in range(0, n, size)]
            self.assertEqual(_balanced_chunks(list(range(n)), size), greedy)


class OrchestrationTest(unittest.TestCase):
    """Chunking / ordering / retry, with `_decode_chunk` stubbed out."""

    def _run(self, cand, texts, decode):
        cand._decode_chunk = decode  # type: ignore[method-assign]
        return cand.synthesize_batch([{"text": t} for t in texts])

    def test_output_order_matches_input_despite_reordering(self):
        """Requests are sorted by cap for batching efficiency. The returned list
        must still line up with the input, or sentences get the wrong audio."""
        import numpy as np

        # Deliberately not in length order.
        texts = ["x" * 300, "x" * 10, "x" * 200, "x" * 50]
        # Encode each request's identity in its waveform via the cap, which is a
        # function of its text length.
        def decode(reqs, attempt=0):
            return [(np.full(8, r.cap / 10_000.0, dtype=np.float32), False) for r in reqs]

        out = self._run(_bare(max_batch=2), texts, decode)
        self.assertEqual(len(out), 4)
        for text, data in zip(texts, out):
            import io
            with wave.open(io.BytesIO(data), "rb") as wf:
                first = int.from_bytes(wf.readframes(1), "little", signed=True)
            expected = int(token_cap(text) / 10_000.0 * 32767)
            self.assertAlmostEqual(first, expected, delta=2, msg=text)

    def test_chunks_are_contiguous_in_sorted_cap_order(self):
        """Wall time for a chunk is max(steps), not mean, so grouping similar
        lengths is the whole reason batching pays.

        The guarantee is that chunks are consecutive slices of the cap-sorted
        order — not that every chunk is internally tight, since with any chunk
        size one boundary group can still straddle a jump in length."""
        import numpy as np

        seen: list[list[int]] = []

        def decode(reqs, attempt=0):
            seen.append([r.cap for r in reqs])
            return [(np.zeros(8, dtype=np.float32), False) for _ in reqs]

        texts = ["x" * n for n in (250, 60, 200, 70, 225, 65)]
        self._run(_bare(max_batch=2), texts, decode)

        caps_sorted = sorted(token_cap(t) for t in texts)
        self.assertEqual([c for chunk in seen for c in chunk], caps_sorted)
        for chunk in seen:
            self.assertEqual(chunk, sorted(chunk))
        # And the pathological pairing is genuinely avoided: the shortest line
        # is never batched with the longest.
        for chunk in seen:
            self.assertFalse(
                caps_sorted[0] in chunk and caps_sorted[-1] in chunk, seen
            )

    def test_respects_max_batch(self):
        import numpy as np

        sizes: list[int] = []

        def decode(reqs, attempt=0):
            sizes.append(len(reqs))
            return [(np.zeros(8, dtype=np.float32), False) for _ in reqs]

        self._run(_bare(max_batch=3), ["a"] * 7, decode)
        # The ceiling and total are the contract this test is named for.
        self.assertTrue(all(n <= 3 for n in sizes), sizes)
        self.assertEqual(sum(sizes), 7)
        # Shape changed deliberately (spec 2026-08-25-1931, B1): this used to
        # assert [1, 3, 3], pinning the greedy split's trailing chunk of ONE —
        # which is the defect, not the contract. A chunk costs max(steps), so a
        # lone trailing row pays a full chunk by itself.
        self.assertEqual(sorted(sizes), [2, 2, 3])

    def test_overflow_is_retried_and_counted(self):
        import numpy as np

        calls = {"n": 0}

        def decode(reqs, attempt=0):
            calls["n"] += 1
            if calls["n"] == 1:
                return [(np.zeros(8, dtype=np.float32), True)]      # overflowed
            return [(np.ones(8, dtype=np.float32), False)]           # clean

        cand = _bare(max_batch=4, max_retries=2)
        out = self._run(cand, ["a"], decode)
        self.assertEqual(calls["n"], 2, "overflow should have triggered one retry")
        self.assertEqual(cand._summary["overflows"], 1)
        self.assertEqual(cand._summary["sentences"], 1)
        # The clean retry is what got returned, not the overflowed first pass.
        self.assertGreater(_wav_duration(out[0]), 0.0)

    def test_only_the_failed_rows_are_retried(self):
        import numpy as np

        seen: list[int] = []

        def decode(reqs, attempt=0):
            seen.append(len(reqs))
            # First pass: middle row overflows. Retry pass: it succeeds.
            if len(reqs) == 3:
                return [
                    (np.zeros(8, dtype=np.float32), False),
                    (np.zeros(8, dtype=np.float32), True),
                    (np.zeros(8, dtype=np.float32), False),
                ]
            return [(np.zeros(8, dtype=np.float32), False) for _ in reqs]

        cand = _bare(max_batch=4, max_retries=2)
        self._run(cand, ["a", "b", "c"], decode)
        self.assertEqual(seen, [3, 1], "the retry must not re-run the clean rows")

    def test_persistent_overflow_falls_back_to_silence(self):
        import numpy as np

        def decode(reqs, attempt=0):
            return [(np.ones(8, dtype=np.float32), True) for _ in reqs]

        cand = _bare(max_batch=4, max_retries=1)
        out = self._run(cand, ["a"], decode)
        # 2 attempts, both overflowed, then the 0.2 s silence fallback.
        self.assertEqual(cand._summary["overflows"], 2)
        self.assertAlmostEqual(_wav_duration(out[0]), 0.2, places=3)

    def test_collapsed_generation_falls_back_to_silence(self):
        """`None` waveform (fewer rows than codebooks) with no overflow flag."""
        def decode(reqs, attempt=0):
            return [(None, False) for _ in reqs]

        cand = _bare(max_batch=4, max_retries=0)
        out = self._run(cand, ["a"], decode)
        self.assertEqual(cand._summary["overflows"], 0)
        self.assertAlmostEqual(_wav_duration(out[0]), 0.2, places=3)

    def test_empty_request_list(self):
        def decode(reqs, attempt=0):  # pragma: no cover - must never be called
            raise AssertionError("no chunks expected")

        self.assertEqual(self._run(_bare(), [], decode), [])

    def test_emotion_tag_is_prefixed_onto_the_prompt(self):
        """Tests the prefixing MECHANISM, deliberately via a bin that still
        carries a tag. Which bins are tagged is policy set by ear-test and is
        pinned in `test_config_profiles.EmotionTagPolicyTest` — this test used
        to hardcode `angry -> anger` and broke when that bin was untagged on
        2026-08-24, which is the wrong reason for this test to fail.
        """
        seen: list[str] = []

        def decode(reqs, attempt=0):
            seen.extend(r.prompt_text for r in reqs)
            return [(None, False) for _ in reqs]

        tagged_bin = next(
            b for b, t in higgs_v3_torch.EMOTION_TAGS.items() if t is not None
        )
        tag = higgs_v3_torch.EMOTION_TAGS[tagged_bin]

        cand = _bare()
        cand._decode_chunk = decode  # type: ignore[method-assign]
        cand.synthesize_batch([
            {"text": "hello", "emotion": tagged_bin},
            {"text": "hello", "emotion": "neutral"},
            {"text": "hello"},
        ])
        self.assertIn(f"<|emotion:{tag}|>hello", seen)

    def test_untagged_bins_reach_the_prompt_unprefixed(self):
        """An untagged bin must pass the text through verbatim — no empty tag,
        no stray marker. `angry` and `happy` are untagged as of 2026-08-24."""
        seen: list[str] = []

        def decode(reqs, attempt=0):
            seen.extend(r.prompt_text for r in reqs)
            return [(None, False) for _ in reqs]

        untagged = [b for b, t in higgs_v3_torch.EMOTION_TAGS.items() if t is None]
        self.assertTrue(untagged, "expected at least one untagged bin")

        cand = _bare()
        cand._decode_chunk = decode  # type: ignore[method-assign]
        cand.synthesize_batch([{"text": "hello", "emotion": b} for b in untagged])
        self.assertEqual(seen, ["hello"] * len(untagged))
        self.assertFalse(any("<|emotion:" in p for p in seen))


@unittest.skipIf(
    _snapshot() is None,
    f"Higgs checkpoint snapshot not found under {_hub_cache()} "
    "(set HF_HOME/HF_HUB_CACHE, or run the TTS smoke test to populate it)",
)
class BatchEquivalenceTest(unittest.TestCase):
    """A row decoded in a batch must equal the same row decoded alone.

    Runs a tiny randomly-initialized model on CPU at temperature 0 (greedy).
    Temperature 0 is deliberate and is NOT because rows share an RNG stream —
    since 2026-08-24 each row draws from its own generator. It is because
    batching is not bitwise-invariant *numerically*: batched matmuls and the
    left-padded prefill do not give identical logits at different batch shapes,
    and sampling amplifies a 1-bit logit difference into a different draw, while
    argmax absorbs it. `SamplingReproducibilityTest` covers what sampling does
    guarantee.

    Random weights never learn to emit a stop token, so rows terminate on their
    own cap; the EOC path is forced separately below.
    """

    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, Qwen2TokenizerFast

        snap = _snapshot()
        cfg = AutoConfig.from_pretrained(str(snap), trust_remote_code=True)
        tc = cfg.text_config
        tc.hidden_size = 64
        tc.num_hidden_layers = 2
        tc.num_attention_heads = 2
        tc.num_key_value_heads = 1
        tc.head_dim = 32
        tc.intermediate_size = 128
        cfg.audio_encoder_config = dict(cfg.audio_encoder_config)
        cfg.audio_encoder_config["out_dim"] = 64

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        cls.model = model.to(torch.float32).eval()

        # Stub codec: makes the returned waveform a faithful fingerprint of the
        # generated codes, so comparing waveforms compares the decode exactly.
        class _Codec:
            def decode(self, codes_BNT):
                class _Out:
                    audio_values = codes_BNT.float().mean(dim=1, keepdim=True) / 2048.0
                return _Out()

        cls.model._audio_codec = _Codec()
        cls.tok = Qwen2TokenizerFast.from_pretrained(str(snap))

    def _candidate(self, **kw):
        return HiggsV3Torch(
            device="cpu", temperature=0.0, max_retries=0,
            _model=self.model, _tokenizer=self.tok, **kw,
        )

    def test_batched_rows_match_standalone_rows(self):
        """The core property. Different prompt lengths, so every row is padded
        differently — the case that breaks if `position_ids` drift from the
        attention mask."""
        import torch

        cand = self._candidate(max_batch=8)
        reqs = [
            _Request("Բարև ձեզ", None, None, cap=30),
            _Request("Ես գնում եմ տուն հիմա", None, None, cap=30),
            _Request("Ինչ", None, None, cap=30),
        ]
        alone = [cand._decode_chunk([r])[0] for r in reqs]
        together = cand._decode_chunk(reqs)

        for i, ((a, _), (b, _)) in enumerate(zip(alone, together)):
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            self.assertEqual(a.shape, b.shape, f"row {i} length differs")
            self.assertTrue(torch.equal(a, b), f"row {i} content differs")

    def test_a_reference_conditioned_row_batches_with_plain_rows(self):
        """A ref clip adds audio-placeholder tokens, so its prompt is a
        different length AND a different shape from a text-only prompt."""
        import torch

        cand = self._candidate(max_batch=8)
        torch.manual_seed(7)
        ref = torch.randint(0, 1024, (5, 8))
        reqs = [
            _Request("no reference", None, None, cap=30),
            _Request("with a reference", ref, "ref transcript", cap=30),
        ]
        alone = [cand._decode_chunk([r])[0] for r in reqs]
        together = cand._decode_chunk(reqs)
        for i, ((a, _), (b, _)) in enumerate(zip(alone, together)):
            self.assertEqual(a.shape, b.shape, f"row {i}")
            self.assertTrue(torch.equal(a, b), f"row {i}")

    def test_a_row_finishing_early_does_not_disturb_live_rows(self):
        """Divergent termination: row 0's small cap retires it mid-batch while
        the others keep going. Its slot still occupies the batch, so this is the
        masking path — if it leaked, the live rows would diverge."""
        import torch

        cand = self._candidate(max_batch=8)
        reqs = [
            _Request("finishes first", None, None, cap=12),
            _Request("keeps going for a while", None, None, cap=40),
            _Request("also keeps going", None, None, cap=40),
        ]
        alone = [cand._decode_chunk([r])[0] for r in reqs]
        together = cand._decode_chunk(reqs)

        # Row 0 really did stop early, or this test proves nothing.
        self.assertLess(together[0][0].shape[0], together[1][0].shape[0])
        for i, ((a, _), (b, _)) in enumerate(zip(alone, together)):
            self.assertEqual(a.shape, b.shape, f"row {i}")
            self.assertTrue(torch.equal(a, b), f"row {i}")

    def test_hitting_the_cap_is_reported_as_overflow(self):
        cand = self._candidate(max_batch=8)
        (_, overflowed), = cand._decode_chunk([_Request("x", None, None, cap=12)])
        self.assertTrue(overflowed, "a run that reaches its cap is an overflow")

    def test_stop_token_terminates_before_the_cap_without_overflow(self):
        """Random weights never emit EOC, so force it: once past the delay
        ramp-up, codebook 0's argmax is the stop id. Exercises the EOC
        wind-down and, unlike every other case here, a clean termination."""
        import torch

        cand = self._candidate(max_batch=8)
        up = cand._up
        real_head = self.model.audio_head
        n, v = cand.n_codebooks, int(self.model.codebook_vocab_size)

        class _ForcedHead(torch.nn.Module):
            def forward(self, hidden):
                logits = torch.zeros(hidden.shape[0], n, v)
                logits[:, 0, up.EOC_ID] = 100.0    # codebook 0 -> stop
                return logits

        self.model.audio_head = _ForcedHead()
        try:
            wav, overflowed = cand._decode_chunk(
                [_Request("plenty of room", None, None, cap=200)]
            )[0]
        finally:
            self.model.audio_head = real_head

        self.assertFalse(overflowed, "terminated by stop token, not by the cap")
        self.assertIsNotNone(wav)
        # Delay ramp (N steps) + EOC detection + (N-2) wind-down, far below 200.
        self.assertLess(wav.shape[0], 30)

    def test_audio_head_is_tied_to_the_embedding(self):
        """The checkpoint omits `audio_head.weight` and relies on `tie_weights()`,
        so transformers reports it as MISSING/newly-initialized on EVERY load.
        That report is a red herring — but only while the tie actually holds."""
        self._candidate(max_batch=8)  # constructing it runs the check
        self.assertEqual(
            self.model.audio_head.weight.data_ptr(),
            self.model.audio_embedding.weight.data_ptr(),
        )

    def test_an_untied_audio_head_is_refused(self):
        """If a transformers change ever re-initialized that key after tying, the
        head behind every audio logit would be random — and nothing would notice:
        the decode runs, sentences terminate, overflows reads 0, RTF looks fine.
        The only symptom is that the output is noise, so refuse to start."""
        import torch

        cand = self._candidate(max_batch=8)
        real = self.model.audio_head.weight
        try:
            self.model.audio_head.weight = torch.nn.Parameter(real.detach().clone())
            with self.assertRaises(RuntimeError) as ctx:
                cand._check_audio_head_tied()
            self.assertIn("noise", str(ctx.exception))
        finally:
            self.model.audio_head.weight = real

    def test_single_entry_point_goes_through_the_batch_path(self):
        """`synthesize` is a batch of one on purpose — one decode loop, not two
        that can drift."""
        cand = self._candidate(max_batch=8)
        data = cand.synthesize("Բարև")
        self.assertGreater(len(data), 44)  # more than a WAV header
        import io
        with wave.open(io.BytesIO(data), "rb") as wf:
            self.assertEqual(wf.getframerate(), SR)
            self.assertEqual(wf.getnchannels(), 1)
        self.assertEqual(cand._summary["sentences"], 1)
        self.assertEqual(cand._summary["batches"], 1)


@unittest.skipIf(
    _snapshot() is None,
    f"Higgs checkpoint snapshot not found under {_hub_cache()}",
)
class ListTolerantLoadTest(unittest.TestCase):
    """The class-level normalization only helps if `from_pretrained` uses the
    class object we patched. Dynamic remote code is re-copied and re-imported
    under a hashed module name, so the loader can hold a DIFFERENT object still
    carrying the shipped list. This patch coerces the attribute where the loader
    reads it, so class identity stops mattering."""

    def setUp(self):
        import transformers.modeling_utils as mu
        self.mu = mu
        self._saved_helper = mu._get_tied_weight_keys
        self.snapshot = str(_snapshot())

    def tearDown(self):
        self.mu._get_tied_weight_keys = self._saved_helper

    def test_loading_helper_tolerates_the_list_without_touching_the_class(self):
        from patmir.eval.candidates.tts.higgs_v3_torch import (
            tolerate_list_tied_weights_keys,
        )

        model = _tiny_model(self.snapshot)
        # Simulate the loader's own class object, untouched by normalization.
        type(model)._tied_weights_keys = ["audio_head.weight"]
        with self.assertRaises(AttributeError):
            self.mu._get_tied_weight_keys(model)

        self.assertTrue(tolerate_list_tied_weights_keys())
        self.assertEqual(
            self.mu._get_tied_weight_keys(_tiny_model(self.snapshot)),
            ["audio_head.weight"],
        )

    def test_patch_is_idempotent(self):
        from patmir.eval.candidates.tts.higgs_v3_torch import (
            tolerate_list_tied_weights_keys,
        )

        tolerate_list_tied_weights_keys()
        self.assertFalse(tolerate_list_tied_weights_keys())

    def test_a_dict_is_left_alone(self):
        """It must only ever widen list -> dict, never the reverse."""
        from patmir.eval.candidates.tts.higgs_v3_torch import (
            tolerate_list_tied_weights_keys,
        )

        tolerate_list_tied_weights_keys()
        model = _tiny_model(self.snapshot)
        mapping = {"audio_head.weight": "audio_embedding.weight"}
        type(model)._tied_weights_keys = dict(mapping)
        self.mu._get_tied_weight_keys(model)
        self.assertEqual(type(model)._tied_weights_keys, mapping)


class AdapterBatchDispatchTest(unittest.TestCase):
    """`SentenceTTS` has to actually route to `synthesize_batch`, hand it the
    WHOLE sentence list (the candidate's length sort is only worth anything if
    it is global), and keep the results aligned to the input order."""

    def _transcript(self, n: int):
        from patmir.pipeline.stage import TranslatedSentence, TranslatedTranscript

        return TranslatedTranscript(
            source_lang="en", target_lang="hy",
            sentences=[
                TranslatedSentence(
                    idx=i, source_text=f"s{i}", target_text="Ա" * (i + 1),
                    start=float(i), end=float(i + 1), speaker_id="SPEAKER_00",
                )
                for i in range(n)
            ],
        )

    def _ctx(self, tmp, *, bench=False):
        from patmir.pipeline.context import JobContext, PipelineOptions

        return JobContext(
            job_dir=Path(tmp),
            source_path=Path(tmp) / "src.wav",
            options=PipelineOptions(),
            extra={"bench": True} if bench else {},
        )

    @staticmethod
    def _wav(seconds: float, sr: int = 16000) -> bytes:
        import io
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b"\x00\x00" * int(seconds * sr))
        return buf.getvalue()

    def _batched_inner(self, outer, max_batch=4, on_chunk_support=True):
        wav = self._wav

        class _Batched:
            ref_text_optional = True
            tts_concurrency = 1
            tts_max_batch = max_batch

            def __init__(self):
                self.batch_calls: list[list[dict]] = []
                self.single_calls: list[str] = []

            def synthesize(self, text, *, ref_audio=None, ref_text=None, emotion=None):
                self.single_calls.append(text)
                return wav(0.1)

            if on_chunk_support:
                def synthesize_batch(self, requests, *, on_chunk=None):
                    self.batch_calls.append(list(requests))
                    # Two chunks, to prove the adapter records each.
                    half = max(1, len(requests) // 2)
                    if on_chunk is not None:
                        on_chunk(list(range(half)), 1.0)
                        on_chunk(list(range(half, len(requests))), 3.0)
                    # Duration encodes position so ordering is checkable.
                    return [wav(0.1 * (i + 1)) for i in range(len(requests))]
            else:
                def synthesize_batch(self, requests):
                    self.batch_calls.append(list(requests))
                    return [wav(0.1 * (i + 1)) for i in range(len(requests))]

        return _Batched()

    def test_batch_path_is_taken_and_gets_every_sentence_at_once(self):
        import tempfile

        from patmir.pipeline.stages.tts.adapter import SentenceTTS

        inner = self._batched_inner(self)
        tts = SentenceTTS(inner)
        self.assertIsNotNone(tts._batch_fn)
        self.assertTrue(tts._batch_on_chunk)
        with tempfile.TemporaryDirectory() as tmp:
            tts.synthesize(self._transcript(5), self._ctx(tmp))
        self.assertEqual(len(inner.batch_calls), 1, "one call, not one per chunk")
        self.assertEqual(len(inner.batch_calls[0]), 5)
        self.assertEqual(inner.single_calls, [], "per-sentence path must not run")

    def test_results_stay_aligned_with_sentence_order(self):
        import tempfile

        from patmir.pipeline.stages.tts.adapter import SentenceTTS

        inner = self._batched_inner(self)
        tts = SentenceTTS(inner)
        with tempfile.TemporaryDirectory() as tmp:
            out = tts.synthesize(self._transcript(4), self._ctx(tmp))
            for i, sent in enumerate(sorted(out.sentences, key=lambda s: s.idx)):
                self.assertEqual(sent.idx, i)
                # The stub's i-th output is 0.1*(i+1) seconds long.
                self.assertAlmostEqual(sent.wav_duration, 0.1 * (i + 1), places=3)
                self.assertEqual(sent.wav_path.name, f"{i:04d}.wav")

    def test_text_and_kwargs_reach_the_batch_request(self):
        import tempfile

        from patmir.pipeline.stages.tts.adapter import SentenceTTS

        inner = self._batched_inner(self)
        tts = SentenceTTS(inner)
        with tempfile.TemporaryDirectory() as tmp:
            tts.synthesize(self._transcript(2), self._ctx(tmp))
        reqs = inner.batch_calls[0]
        self.assertEqual([r["text"] for r in reqs], ["Ա", "ԱԱ"])
        for r in reqs:
            # The adapter probes `synthesize`'s signature, which offers these.
            self.assertIn("ref_audio", r)
            self.assertIn("emotion", r)

    def test_max_batch_of_one_uses_the_per_sentence_loop(self):
        """A candidate can expose the batch API and still opt out by size."""
        import tempfile

        from patmir.pipeline.stages.tts.adapter import SentenceTTS

        inner = self._batched_inner(self, max_batch=1)
        tts = SentenceTTS(inner)
        with tempfile.TemporaryDirectory() as tmp:
            tts.synthesize(self._transcript(3), self._ctx(tmp))
        self.assertEqual(inner.batch_calls, [])
        self.assertEqual(len(inner.single_calls), 3)

    def test_candidate_without_the_batch_api_is_unaffected(self):
        import tempfile

        from patmir.pipeline.stages.tts.adapter import SentenceTTS

        wav = self._wav

        class _Plain:
            def synthesize(self, text):
                return wav(0.1)

        tts = SentenceTTS(_Plain())
        self.assertIsNone(tts._batch_fn)
        with tempfile.TemporaryDirectory() as tmp:
            out = tts.synthesize(self._transcript(3), self._ctx(tmp))
        self.assertEqual(len(out.sentences), 3)

    def test_bench_timings_are_per_chunk_and_totals_stay_true(self):
        """The reason chunk rows exist: attributing a chunk's wall to each of
        its sentences would multiply bench's `total_s` and `tts_rtf` by the
        batch size."""
        import json
        import tempfile

        from patmir.pipeline.stages.tts.adapter import SentenceTTS

        inner = self._batched_inner(self)
        tts = SentenceTTS(inner)
        with tempfile.TemporaryDirectory() as tmp:
            tts.synthesize(self._transcript(4), self._ctx(tmp, bench=True))
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "tts" / "timings.jsonl").read_text().splitlines()
            ]
        self.assertEqual([r["kind"] for r in rows], ["chunk", "chunk"])
        self.assertEqual([r["sentences"] for r in rows], [2, 2])
        # The stub reported 1.0 s and 3.0 s; the total must be exactly 4.0.
        self.assertAlmostEqual(sum(r["elapsed_s"] for r in rows), 4.0, places=3)
        # And audio must total the four stub durations, 0.1+0.2+0.3+0.4.
        self.assertAlmostEqual(sum(r["audio_s"] for r in rows), 1.0, places=2)

    def test_a_batched_candidate_without_on_chunk_still_records_one_span(self):
        import json
        import tempfile

        from patmir.pipeline.stages.tts.adapter import SentenceTTS

        inner = self._batched_inner(self, on_chunk_support=False)
        tts = SentenceTTS(inner)
        self.assertFalse(tts._batch_on_chunk)
        with tempfile.TemporaryDirectory() as tmp:
            tts.synthesize(self._transcript(3), self._ctx(tmp, bench=True))
            rows = [
                json.loads(line)
                for line in (Path(tmp) / "tts" / "timings.jsonl").read_text().splitlines()
            ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sentences"], 3)

    def test_progress_is_reported_per_chunk_not_once_at_the_end(self):
        import tempfile

        from patmir.pipeline.stages.tts.adapter import SentenceTTS

        inner = self._batched_inner(self)
        tts = SentenceTTS(inner)
        seen: list[float] = []
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            ctx.progress = lambda stage, frac, msg="": seen.append(frac)  # type: ignore[assignment]
            tts.synthesize(self._transcript(4), ctx)
        # Two chunk callbacks plus the final 1.0 — the stage must not go dark.
        self.assertGreaterEqual(len(seen), 3)
        self.assertLessEqual(max(seen), 1.0)
        self.assertEqual(seen[-1], 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class RequestSeedTest(unittest.TestCase):
    """The per-row RNG seed is derived from the request's own content.

    This is what makes a run reproducible: the seed must not depend on batch
    position, call order, or process identity.
    """

    def test_the_same_request_always_gets_the_same_seed(self):
        a = _request_seed("Վրան արյուն կա", "reference text")
        b = _request_seed("Վրան արյուն կա", "reference text")
        self.assertEqual(a, b)

    def test_the_seed_is_stable_across_processes(self):
        """Pinned literal. `hash()` is salted per process, so using it here would
        make every run irreproducible in exactly the way this exists to prevent —
        a regression to it would change this number."""
        self.assertEqual(_request_seed("Վրան արյուն կա", "ref"), 1437796720965076717)

    def test_text_and_reference_both_move_the_seed(self):
        base = _request_seed("Վրան արյուն կա", "ref")
        self.assertNotEqual(base, _request_seed("Վրան արյուն կա։", "ref"))
        self.assertNotEqual(base, _request_seed("Վրան արյուն կա", "other ref"))
        self.assertNotEqual(base, _request_seed("Վրան արյուն կա", None))

    def test_the_text_reference_boundary_cannot_be_smeared(self):
        """Without a separator these two would hash identically."""
        self.assertNotEqual(_request_seed("ab", "c"), _request_seed("a", "bc"))

    def test_the_seed_fits_a_generator(self):
        for text in ("", "a", "Վրան արյուն կա" * 50):
            seed = _request_seed(text, None)
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 1 << 62)

    def test_prepare_stamps_the_content_seed_onto_the_request(self):
        cand = HiggsV3Torch.__new__(HiggsV3Torch)
        cand._ref_cache = {}
        req = cand._prepare("Վրան արյուն կա", None, "ref", None)
        self.assertEqual(req.seed, _request_seed("Վրան արյուն կա", "ref"))


class SampleCodesTest(unittest.TestCase):
    """Our copy of the checkpoint's sampler, which adds the generator it lacks."""

    def _logits(self, rows=3, vocab=16):
        import torch
        torch.manual_seed(0)
        return torch.randn(rows, vocab)

    def test_zero_temperature_is_argmax(self):
        import torch
        logits = self._logits()
        out = _sample_codes(logits, temperature=0.0, top_p=None, top_k=None,
                            generator=None)
        self.assertTrue(torch.equal(out, logits.argmax(dim=-1)))

    def test_the_same_seed_draws_the_same_codes(self):
        import torch
        logits = self._logits()
        g1 = torch.Generator(); g1.manual_seed(1234)
        g2 = torch.Generator(); g2.manual_seed(1234)
        a = _sample_codes(logits, temperature=0.8, top_p=0.95, top_k=None, generator=g1)
        b = _sample_codes(logits, temperature=0.8, top_p=0.95, top_k=None, generator=g2)
        self.assertTrue(torch.equal(a, b))

    def test_different_seeds_draw_differently(self):
        import torch
        logits = self._logits(rows=8, vocab=256)
        draws = []
        for seed in (1, 2, 3, 4):
            g = torch.Generator(); g.manual_seed(seed)
            draws.append(tuple(_sample_codes(
                logits, temperature=1.0, top_p=None, top_k=None,
                generator=g).tolist()))
        self.assertGreater(len(set(draws)), 1)

    def test_the_draw_does_not_touch_the_global_rng(self):
        """A row's draw must not advance the stream any other row reads."""
        import torch
        logits = self._logits()
        torch.manual_seed(99)
        before = torch.rand(4)
        torch.manual_seed(99)
        g = torch.Generator(); g.manual_seed(7)
        _sample_codes(logits, temperature=0.8, top_p=0.95, top_k=None, generator=g)
        after = torch.rand(4)
        self.assertTrue(torch.equal(before, after))

    def test_top_k_restricts_the_support(self):
        import torch
        logits = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
        for seed in range(8):
            g = torch.Generator(); g.manual_seed(seed)
            out = _sample_codes(logits, temperature=1.0, top_p=None, top_k=1,
                                generator=g)
            self.assertEqual(out.tolist(), [1])

    def test_top_p_keeps_at_least_the_top_token(self):
        """A nucleus small enough to exclude everything must still leave one."""
        import torch
        logits = torch.tensor([[0.0, 5.0, 1.0, 2.0]])
        for seed in range(8):
            g = torch.Generator(); g.manual_seed(seed)
            out = _sample_codes(logits, temperature=1.0, top_p=0.01, top_k=None,
                                generator=g)
            self.assertEqual(out.tolist(), [1])


class ChunkLoopCancellationTest(unittest.TestCase):
    """Cancellation granularity inside TTS.

    TTS is ~55% of batch wall (~77s on CUDA), so checking the cancel token only
    at stage boundaries would make Stop take over a minute. The adapter instead
    raises from the `on_chunk` callback the candidate already invokes after every
    decode chunk (~7.7 sentences, ~8s), which needs NO cooperation from the
    candidate — provided an exception raised in that callback really does abort
    the loop rather than being swallowed. That is what these pin down.
    """

    def _decode(self, calls):
        import numpy as np

        def decode(reqs, attempt=0):
            calls.append(len(reqs))
            return [(np.zeros(8, dtype=np.float32), False) for _ in reqs]
        return decode

    def test_raising_from_on_chunk_aborts_the_loop_early(self):
        from patmir.pipeline.context import JobCancelled

        calls: list[int] = []
        cand = _bare(max_batch=2)
        cand._decode_chunk = self._decode(calls)  # type: ignore[method-assign]

        def on_chunk(indices, wall_s):
            raise JobCancelled()

        # 8 sentences at max_batch=2 would be 4 chunks if it ran to completion.
        with self.assertRaises(JobCancelled):
            cand.synthesize_batch([{"text": "x" * 50} for _ in range(8)],
                                  on_chunk=on_chunk)
        self.assertEqual(calls, [2], "loop did not stop at the first chunk")

    def test_on_chunk_is_called_per_chunk(self):
        """The granularity claim itself: one callback per chunk, so the check
        happens ~every `max_batch` sentences rather than once per stage."""
        calls: list[int] = []
        seen: list[list[int]] = []
        cand = _bare(max_batch=2)
        cand._decode_chunk = self._decode(calls)  # type: ignore[method-assign]
        cand.synthesize_batch([{"text": "x" * 50} for _ in range(8)],
                              on_chunk=lambda idx, w: seen.append(list(idx)))
        self.assertEqual(len(seen), 4, f"expected 4 chunks, got {seen}")
        self.assertEqual(sorted(i for c in seen for i in c), list(range(8)))

    def test_uncancelled_batch_completes(self):
        """Control: the hook must not perturb the normal path."""
        calls: list[int] = []
        cand = _bare(max_batch=2)
        cand._decode_chunk = self._decode(calls)  # type: ignore[method-assign]
        out = cand.synthesize_batch([{"text": "x" * 50} for _ in range(8)],
                                    on_chunk=lambda idx, w: None)
        self.assertEqual(len(out), 8)
        self.assertEqual(len(calls), 4)


class BatchedFilterEquivalenceTest(unittest.TestCase):
    """`_filter_probs` over a chunk must equal it applied to each row alone.

    This is the load-bearing claim behind vectorizing the sampler. The DRAW
    stays per row from that row's own generator, so if the filtering is
    identical then the whole sampling path is, and no line's audio can move.
    Every operation involved is independent along the last dim; these tests pin
    that it stays true, including on the tie-heavy logits a bf16 audio head
    actually produces, where a sort's tie-break order is the thing that could
    in principle depend on batch shape.
    """

    def _check(self, logits, *, temperature, top_p, top_k):
        import torch

        from patmir.eval.candidates.tts.higgs_v3_torch import _filter_probs
        batched = _filter_probs(logits, temperature=temperature, top_p=top_p,
                                top_k=top_k)
        for b in range(logits.shape[0]):
            alone = _filter_probs(logits[b], temperature=temperature,
                                  top_p=top_p, top_k=top_k)
            self.assertTrue(torch.equal(batched[b], alone),
                            f"row {b} diverged batched vs alone")

    def test_top_p_only_matches_per_row(self):
        import torch
        for seed in range(20):
            torch.manual_seed(seed)
            self._check(torch.randn(8, 8, 1026), temperature=0.8, top_p=0.95,
                        top_k=None)

    def test_top_k_and_top_p_match_per_row(self):
        import torch
        for seed in range(10):
            torch.manual_seed(100 + seed)
            self._check(torch.randn(8, 8, 1026), temperature=0.8, top_p=0.95,
                        top_k=50)

    def test_bf16_valued_logits_with_many_ties_match(self):
        import torch
        for seed in range(10):
            torch.manual_seed(200 + seed)
            logits = torch.randn(8, 8, 1026).to(torch.bfloat16).to(torch.float32)
            self._check(logits, temperature=0.8, top_p=0.95, top_k=None)

    def test_single_row_batch_matches_unbatched(self):
        import torch
        torch.manual_seed(7)
        self._check(torch.randn(1, 8, 1026), temperature=0.8, top_p=0.95,
                    top_k=None)

    def test_sample_codes_delegates_to_the_same_filter(self):
        import torch

        from patmir.eval.candidates.tts.higgs_v3_torch import (
            _filter_probs,
            _sample_codes,
        )
        torch.manual_seed(0)
        logits = torch.randn(8, 1026)
        probs = _filter_probs(logits, temperature=0.8, top_p=0.95, top_k=None)
        g1 = torch.Generator().manual_seed(11)
        g2 = torch.Generator().manual_seed(11)
        via_helper = _sample_codes(logits, temperature=0.8, top_p=0.95,
                                   top_k=None, generator=g1)
        direct = probs.multinomial(num_samples=1, generator=g2).squeeze(-1)
        self.assertTrue(torch.equal(via_helper, direct))

    def test_zero_temperature_is_argmax_and_consumes_no_randomness(self):
        # The equivalence test compares batched against standalone rows at
        # temperature 0, which only works if the argmax path leaves the
        # generator untouched — as upstream's does.
        import torch

        from patmir.eval.candidates.tts.higgs_v3_torch import _sample_codes
        torch.manual_seed(0)
        logits = torch.randn(8, 1026)
        g = torch.Generator().manual_seed(5)
        before = g.get_state()
        out = _sample_codes(logits, temperature=0.0, top_p=0.95, top_k=None,
                            generator=g)
        self.assertTrue(torch.equal(out, logits.argmax(dim=-1)))
        self.assertTrue(torch.equal(g.get_state(), before))


class SamplerStateVectorizationTest(unittest.TestCase):
    """`_DelayState` must match the checkpoint's `_sampler_step`, exactly.

    The decode loop used to call upstream's function once per row per step. It
    mutates a `_SamplerState` and reads `codes_N[0].item()` — a device->host
    sync per row per step, ~8 on each of ~1300 steps.

    Forking a state machine costs more than copying a sampler, which is why the
    module docstring used to say the machine stays upstream's. So `_DelayState`
    is the ONE implementation the decode loop runs, and it is driven here
    against upstream's own function as an oracle: random code streams, EOC
    injected at random steps INCLUDING inside the delay ramp where upstream
    ignores it, and random caps. A checkpoint that changes the machine fails
    this test rather than silently changing every dub.
    """

    @classmethod
    def setUpClass(cls):
        try:
            cls.up = _upstream_module()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"checkpoint snapshot unavailable: {exc}")

    def _oracle(self, streams, caps, N):
        """Upstream's machine, driven row by row exactly as the old loop did."""
        import torch

        rows, overflow = [], []
        for stream, cap in zip(streams, caps):
            state = self.up._SamplerState(num_codebooks=N)
            emitted, over = [], False
            for step in range(len(stream)):
                # A one-hot at temperature 0 sends `_sampler_step` down its
                # argmax path, so it returns exactly these codes and then runs
                # its own bookkeeping — which is how the old loop drove it.
                onehot = torch.zeros(N, 1026)
                onehot.scatter_(-1, stream[step].unsqueeze(-1).long(), 1.0)
                out = self.up._sampler_step(onehot, state, temperature=0.0,
                                            top_p=None, top_k=None)
                if state.generation_done:
                    break
                emitted.append(out.clone())
                if len(emitted) >= cap:
                    over = True
                    break
            rows.append(emitted)
            overflow.append(over)
        return rows, overflow

    def _vectorized(self, streams, caps, N):
        """`_DelayState` as `_decode_chunk` drives it."""
        import torch

        from patmir.eval.candidates.tts.higgs_v3_torch import _DelayState
        B = len(streams)
        max_cap = max(caps)
        state = _DelayState(
            batch=B, num_codebooks=N,
            caps=torch.tensor(caps, dtype=torch.long),
            boc_id=self.up.BOC_ID, eoc_id=self.up.EOC_ID, device=None)
        emitted = torch.zeros(B, max_cap, N, dtype=torch.long)
        for step in range(max_cap):
            codes = torch.stack([s[step].long() for s in streams])
            state.step(codes, step)
            emitted[:, step] = codes
            if state.all_done():
                break
        return emitted, state.length.tolist(), state.overflowed.tolist()

    def _compare(self, streams, caps, N):
        import torch
        ref_rows, ref_over = self._oracle(streams, caps, N)
        emitted, lengths, over = self._vectorized(streams, caps, N)
        for b, ref in enumerate(ref_rows):
            self.assertEqual(lengths[b], len(ref),
                             f"row {b}: emitted {lengths[b]}, upstream {len(ref)}")
            for s, frame in enumerate(ref):
                self.assertTrue(torch.equal(emitted[b, s], frame),
                                f"row {b} frame {s} differs from upstream")
        self.assertEqual(over, ref_over)

    def test_random_streams_with_eoc_anywhere(self):
        import torch
        N = 8
        for seed in range(60):
            torch.manual_seed(seed)
            B = int(torch.randint(1, 5, (1,)).item())
            caps = [int(torch.randint(N + 2, 40, (1,)).item()) for _ in range(B)]
            width = max(caps) + 2
            streams = []
            for cap in caps:
                st = torch.randint(0, 1000, (width, N))
                if bool(torch.rand(1) < 0.8):
                    at = int(torch.randint(0, cap, (1,)).item())
                    st[at, 0] = self.up.EOC_ID
                streams.append(st)
            self._compare(streams, caps, N)

    def test_eoc_during_the_delay_ramp_is_ignored(self):
        import torch
        N = 8
        st = torch.randint(0, 1000, (32, N))
        st[2, 0] = self.up.EOC_ID          # inside the ramp
        self._compare([st], [30], N)
        _e, lengths, over = self._vectorized([st], [30], N)
        self.assertEqual(lengths[0], 30)   # ran to its cap, did not stop at 8
        self.assertTrue(over[0])

    def test_eoc_after_the_ramp_retires_the_row_six_steps_later(self):
        import torch
        N = 8
        st = torch.randint(0, 1000, (40, N))
        st[10, 0] = self.up.EOC_ID
        self._compare([st], [35], N)
        _e, lengths, over = self._vectorized([st], [35], N)
        # Detected at step 10 and still emitted, then five more, retiring at 16.
        self.assertEqual(lengths[0], 16)
        self.assertFalse(over[0])

    def test_boc_masking_matches_the_ramp_position(self):
        import torch
        N = 8
        st = torch.randint(0, 1000, (12, N))
        emitted, lengths, _ = self._vectorized([st], [12], N)
        for step in range(N - 1):
            self.assertTrue(
                bool((emitted[0, step, step + 1:] == self.up.BOC_ID).all()),
                f"step {step} did not mask codebooks past the ramp position")
        self.assertEqual(lengths[0], 12)

    def test_rows_retire_independently(self):
        import torch
        N = 8
        a = torch.randint(0, 1000, (40, N))
        b = torch.randint(0, 1000, (40, N))
        a[9, 0] = self.up.EOC_ID
        self._compare([a, b], [35, 35], N)

    def test_cap_is_exact_not_inferred_from_length(self):
        import torch
        N = 8
        st = torch.randint(0, 1000, (40, N))   # no EOC at all
        self._compare([st], [N + 3], N)
        _e, lengths, over = self._vectorized([st], [N + 3], N)
        self.assertEqual(lengths[0], N + 3)
        self.assertTrue(over[0])
