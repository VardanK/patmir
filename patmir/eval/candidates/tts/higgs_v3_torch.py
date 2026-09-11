"""Higgs Audio v3 TTS (4B) on PyTorch — the CUDA path, with batched decode.

Sibling of `higgs_v3_mlx.py`. Same model, same voice cloning, same inline
emotion tags; the difference is who runs the autoregressive decode.

On Apple Silicon we hand-ported the AR graph to MLX because torch-MPS cannot do
fast quantized matmul, and the decode is memory-bandwidth-bound on the 4B
weights (~69 ms/step at torch-MPS bf16 vs ~28 ms/step at MLX q8). On a discrete
NVIDIA GPU that argument dissolves: bf16 is native and bandwidth is ~4.7x.

**Why this file owns a decode loop at all.** Upstream ships a complete
implementation (`generate_speech`), and the first cut of this candidate simply
called it. But `generate_speech` is batch-1, and batch-1 is precisely the wrong
shape for a bandwidth-bound decode: every step reads all 4B weights to produce
one sentence's next frame. Reading them once to advance B sentences is close to
free, so batching is the single largest lever available on this stage — TTS is
~69% of batch wall. That cannot be reached through upstream's public API, so the
loop lives here.

It is still not a second port. Everything model-specific is reused from the
loaded checkpoint's own module: the delay-pattern helpers, the codec constants,
and — importantly — `_SamplerState` / `_sampler_step`, so the delay ramp-up and
EOC wind-down state machine is upstream's, run once per row. What this file adds
is only the batching mechanics: left-padded prefill, per-row caps and stop
state, and per-row de-delay + codec decode.

Notes on the batching:

* **Left padding.** Prompts differ in length (ref-clip codes + text), so they
  are left-padded and `position_ids` derived from the attention mask. Left, not
  right, so the final position is a real token for every row and one
  `last_hidden_state[:, -1]` serves the whole batch.
* **Sorted chunking.** Wall time for a chunk is `max(steps)`, not `mean`, so a
  short line batched with a long one waits. Requests are sorted by their token
  cap (proportional to text length, which predicts generation length) before
  chunking, which keeps `sum/max` within a chunk close to the batch size.
* **Exact overflow.** Owning the loop means row counts are known, so the
  non-termination guardrail no longer has to infer overflow from output length
  the way the `generate_speech` version did.
* **Sampling is reproducible** (since 2026-08-24), which it was not. Each row
  draws from its own `torch.Generator`, seeded from a content hash of the
  request, so the same sentence set yields byte-identical audio on every run
  regardless of call order, and two identical requests in one chunk now produce
  identical audio. Before this, rows shared one global RNG stream seeded once per
  chunk, so a row's draw depended on its slot index and on how many draws its
  chunk-mates had made — repeated runs of one clip put the same sentence's
  utterance-final level anywhere across a 26 dB range, heard by the author as
  wrong emphasis on some lines on some runs. The module used to call that
  "expected for sampled decoding"; it was measured instead. See `_sample_codes`.
* **It is still not batch-INVARIANT, and that is not an RNG problem.** Changing
  which sentences share a chunk still changes a line's audio, because batched
  matmuls and the left-padded prefill (`seq = max` over the chunk) do not produce
  bitwise-identical logits at different batch shapes, and at temperature 0.8 a
  1-bit logit difference can flip a multinomial draw. Fixing that would mean
  bitwise-stable kernels across shapes, not a seeding change. This is why
  `test_batched_rows_match_standalone_rows` pins temperature to 0 — argmax
  tolerates the numerical drift that sampling amplifies. Practical consequence:
  re-running a clip reproduces exactly, but adding or removing one sentence can
  redraw unrelated lines.

Both Higgs candidates stay registered; `[pipeline].tts` (or a machine profile)
picks one. ADRs: `2026-06-20-higgs-v3-tts-mlx-q8` for the model,
`2026-08-23-cuda-port-machine-profiles` for this runtime.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

from patmir.pipeline.probe import Probe

REPO_DEFAULT = "multimodalart/higgs-audio-v3-tts-4b-transformers"

# Output sample rate, fixed by the checkpoint's codec.
SR = 24000

# Pipeline emotion bins -> Higgs inline emotion tags. Kept identical to
# `higgs_v3_mlx.EMOTION_TAGS` so a runtime swap can't change delivery
# (asserted in test_config_profiles.RuntimeParityTest).
EMOTION_TAGS = {
    # Ear-test 2026-08-24 (`runs/emotion-ab/`, 5 policies x 5 real lines on the
    # real pinned refs) — angry and happy are deliberately UNTAGGED.
    #
    # On three angry lines the author never once preferred `<|emotion:anger|>`:
    # no-tag was the top pick on two and the mildest tagged variant
    # (`bitterness` + `expressive_low`) on the third. On the happy line nothing
    # read as happy at all — `elation` did not register, and `contentment` added
    # an unwanted flirtatious tone. Since 35% of that clip's sentences were
    # tagged angry and 9% happy, dropping these two covers ~43% of all lines
    # and is most of the fix for quality issue #15.
    #
    # Emotion for these bins now comes from the cloned reference clip alone,
    # which is the character actually being tense — the bound described in ADR
    # 2026-05-26-emotion-routed-through-ref-clip-binning, reached from the other
    # direction.
    #
    # `sad` and `fearful` were ear-tested separately (`runs/emotion-ab2/`,
    # 5 policies x 4 lines) and BOTH moved off their original tags too:
    #   sad     sadness -> helplessness   (won 2 of 3 lines outright; the third
    #                                      preferred helplessness + expressive_low,
    #                                      so helplessness won all three)
    #   fearful fear    -> shame          (won the fearful line; fear + low second)
    #
    # Across both rounds the shipped mapping was never preferred on ANY line, in
    # ANY bin — so this table is a full replacement of the original
    # happy/sad/angry/fearful -> elation/sadness/anger/fear guesses, which were
    # chosen as the obvious name matches and never listened to.
    #
    # `<|prosody:expressive_low|>` is NOT used. It measurably changes delivery
    # (identical prompt +/- that token yields different audio, verified by md5),
    # but it only won one line of nine, and on another the author judged it
    # "too emotional" — i.e. it made the line MORE intense, the opposite of the
    # name. Treat every tag name here as a label, not a description: nothing
    # upstream documents what any of them do acoustically (model card lists
    # names; there is no paper). 21 emotion tags exist in the checkpoint --
    # see stages/tts/higgs-audio-v3.md before changing any of this.
    "happy": None,
    "sad": "helplessness",
    "angry": None,
    "fearful": "shame",
    "neutral": None,
}


def _resolve_device(requested: str) -> str:
    if requested in ("cpu", "cuda", "mps"):
        return requested
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def token_cap(prompt_text: str) -> int:
    """Length-proportional cap on generated frames.

    Legit lines use ~2.3 steps/char; cap at 4x/char so a non-terminating run is
    caught in about a second instead of running to a fixed 1024. Same formula as
    the MLX candidate, and it doubles as the sort key for batch chunking.
    """
    return min(1024, max(200, int(len(prompt_text) * 4)))


@dataclass
class _Request:
    """One prepared synthesis request (prompt assembled, reference encoded)."""
    prompt_text: str
    ref_codes: Any | None
    ref_text: str | None
    cap: int
    # Per-row RNG seed; `_prepare` derives it from the request's content. The
    # default keeps hand-built requests (tests) constructible.
    seed: int = 0


def _request_seed(prompt_text: str, ref_text: str | None) -> int:
    """Stable per-request RNG seed derived from the request's own content.

    Content-addressed, so the same sentence with the same reference draws the
    same sampling stream in any job, in any batch position, on any run. blake2b
    rather than `hash()`, which is salted per process and would make runs
    irreproducible in exactly the way this exists to prevent.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(prompt_text.encode("utf-8"))
    h.update(b"\x00")
    h.update((ref_text or "").encode("utf-8"))
    return int.from_bytes(h.digest(), "big") & ((1 << 62) - 1)


def _filter_probs(logits: Any, *, temperature: float, top_p: float | None,
                  top_k: int | None) -> Any:
    """Temperature / top-k / top-p filtering, then softmax -> probabilities.

    Rank-agnostic (`...` indexing), so ONE implementation serves a single row's
    `[N, V]` and a whole chunk's `[B, N, V]`. Every operation is independent
    along the last dim, which is why filtering the chunk in one call gives each
    row bitwise the same probabilities as filtering it alone — verified on fp32
    and on the tie-heavy logits a bf16 audio head actually produces, and pinned
    by `BatchedFilterEquivalenceTest`.

    Kept in step with the checkpoint's own `_sample`. The ONLY divergence is
    that upstream's `multinomial` draws from the global RNG, which is what made
    a sentence's audio depend on its batch-mates; the draw therefore stays with
    the caller, one `multinomial` per row from that row's own generator.
    """
    import torch

    out = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, out.size(-1))
        kth = out.topk(k, dim=-1).values[..., -1:]
        out = torch.where(out < kth, float("-inf"), out)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scatter = torch.zeros_like(remove)
        scatter.scatter_(-1, sorted_idx, remove)
        out = torch.where(scatter, float("-inf"), out)
    return out.softmax(dim=-1)


def _sample_codes(logits_NV: Any, *, temperature: float, top_p: float | None,
                  top_k: int | None, generator: Any) -> Any:
    """One row's draw. Delegates filtering to `_filter_probs` so the single-row
    and batched paths cannot drift apart."""
    if temperature <= 1e-5:
        return logits_NV.argmax(dim=-1)
    probs = _filter_probs(logits_NV, temperature=temperature, top_p=top_p,
                          top_k=top_k)
    return probs.multinomial(num_samples=1, generator=generator).squeeze(-1)


class _DelayState:
    """Upstream's per-row delay/EOC sampler state machine, over a `[B]` state.

    The decode loop used to call the checkpoint's own `_sampler_step` once per
    row per step. That function mutates one `_SamplerState` and reads
    `codes_N[0].item()` — a device->host sync per row per step, ~8 on every one
    of ~1300 steps.

    Forking a state machine is more expensive than copying a sampler, which is
    why the module docstring used to say the machine stays upstream's. So this
    is ONE implementation, exercised by the decode loop and driven directly by
    `SamplerStateVectorizationTest` against upstream's `_sampler_step` as an
    oracle — random code streams, EOC injected at random steps including inside
    the delay ramp where upstream ignores it, and random caps. A checkpoint
    that changes the machine fails that test.

    Upstream's control flow, reproduced exactly:

      delay_count < N   mask codebooks past the ramp position; NO end-of-codebook
                        check on this branch
      countdown armed   decrement; retire at <= 0, BEFORE emitting the frame
      else EOC drawn    arm the countdown at N-2 (retire at once when N <= 2);
                        this frame IS still emitted

    `delay_count == step` holds for every row because all rows start together
    and the delay branch can never retire one, so no row is retired while
    `step < N`.
    """

    def __init__(self, *, batch: int, num_codebooks: int, caps: Any,
                 boc_id: int, eoc_id: int, device: Any) -> None:
        import torch

        self.N = int(num_codebooks)
        self.BOC = int(boc_id)
        self.EOC = int(eoc_id)
        self.caps = caps
        self.length = torch.zeros(batch, dtype=torch.long, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.overflowed = torch.zeros(batch, dtype=torch.bool, device=device)
        # -1 stands for upstream's `eoc_countdown is None`.
        self.countdown = torch.full((batch,), -1, dtype=torch.long, device=device)
        self._eoc_fill = torch.full((batch,), self.N - 2, dtype=torch.long,
                                    device=device)

    def step(self, codes: Any, step: int) -> Any:
        """Advance one decode step. Mutates `codes` in place on the delay ramp
        and returns the `[B]` bool mask of rows that emitted this frame."""
        import torch

        N = self.N
        alive = ~self.done
        if step < N:
            if step + 1 < N:
                codes[:, step + 1:] = self.BOC
            newly_done = torch.zeros_like(self.done)
        else:
            counting = alive & (self.countdown >= 0)
            self.countdown = torch.where(counting, self.countdown - 1,
                                         self.countdown)
            newly_done = counting & (self.countdown <= 0)
            eoc = alive & ~counting & (codes[:, 0] == self.EOC)
            if N <= 2:
                newly_done = newly_done | eoc
            else:
                self.countdown = torch.where(eoc, self._eoc_fill, self.countdown)

        emitting = alive & ~newly_done
        self.length = torch.where(emitting, step + 1, self.length)
        # Ran to its own cap without emitting a stop. Owning the loop makes this
        # an exact signal, not an inference from the output's length.
        capped = emitting & (self.caps <= step + 1)
        self.overflowed |= capped
        self.done = self.done | newly_done | capped
        return emitting

    def all_done(self) -> bool:
        """Costs one device->host read; the loop asks every `_SYNC_EVERY` steps."""
        return bool(self.done.all())


# transformers' attention name for our single-token decode kernel. Registered
# once, lazily, and only used when the candidate asks for it by name.
_GQA_ATTN = "patmir_gqa_decode"
_gqa_registered = False


def _register_gqa_attention() -> bool:
    """Register a grouped-query attention kernel for the q_len == 1 decode step.

    Why this exists: transformers' sdpa path sets `use_gqa_in_sdpa` to False
    whenever an attention mask is present (`integrations/sdpa_attention.py`),
    so `repeat_kv` materializes K and V four times over — 8 KV heads expanded
    to 32 query heads — on every layer of every decode step. At B=8 that copy
    dominates the layer: measured 0.48 ms per layer against 0.17 ms for a
    grouped matmul, which over 36 layers is ~11 ms of a ~32 ms step.

    Only the single-token case is handled. Prefill (q_len > 1) is delegated
    straight back to transformers' own sdpa forward, so the long, expensive,
    numerically-delicate path is untouched and this is not a second attention
    implementation to maintain.

    `F.scaled_dot_product_attention(enable_gqa=True)` was the obvious
    alternative and is worse: with a mask it falls to the cuDNN backend and
    measured 0.99-1.80 ms per layer, slower than the copy it avoids.
    """
    global _gqa_registered
    if _gqa_registered:
        return True
    try:
        import torch
        from transformers import AttentionInterface, AttentionMaskInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.masking_utils import sdpa_mask
    except ImportError:
        return False

    def _gqa_decode(module, query, key, value, attention_mask,
                    scaling=None, **kwargs):
        if query.shape[2] != 1 or attention_mask is None:
            return sdpa_attention_forward(module, query, key, value,
                                          attention_mask, scaling=scaling,
                                          **kwargs)
        B, H, _q, D = query.shape
        kv_heads = key.shape[1]
        group = H // kv_heads
        # [B, KVH, G, D] @ [B, KVH, D, L] -> [B, KVH, G, L]: the group axis
        # rides along as extra rows of the matmul, so the KV heads are read
        # once instead of being expanded `group` times.
        scores = torch.matmul(query.view(B, kv_heads, group, D),
                              key.transpose(-1, -2)).float()
        scores = scores * (scaling if scaling is not None else D ** -0.5)
        # The mask arrives as [B, 1, q, L]; q is 1 here.
        scores = scores.masked_fill(
            ~attention_mask[:, :, 0].unsqueeze(1).bool(), float("-inf"))
        probs = scores.softmax(dim=-1).to(value.dtype)
        out = torch.matmul(probs, value)                    # [B, KVH, G, D]
        # transformers expects [B, q, H, D].
        return out.view(B, H, 1, D).transpose(1, 2), None

    try:
        AttentionInterface.register(_GQA_ATTN, _gqa_decode)
        # Masks are built exactly as sdpa's are; only the kernel differs.
        AttentionMaskInterface.register(_GQA_ATTN, sdpa_mask)
    except Exception:  # noqa: BLE001
        return False
    _gqa_registered = True
    return True


class _DecodeGraph:
    """One captured decode step, replayed once per token.

    What is captured: audio embedding of the fed codes, the 36-layer backbone
    forward against a `StaticCache`, the audio head, and the position
    increment. What stays OUTSIDE: sampling. Per-row generators must keep
    advancing on the host — capturing `multinomial` would freeze its RNG
    offsets and reintroduce exactly the batch-position dependence that ADR
    2026-08-24 removed.

    Three things make the capture legal, and all three came from the
    "worth 0 ms" groundwork commit: the mask lives at a fixed address, nothing
    inside reads back to the host, and the state machine no longer calls
    `.item()`.

    Equivalence is measured, not assumed. Graph replay is bitwise identical to
    eager stepping and `StaticCache` is bitwise identical to `DynamicCache`
    (verified over 12 steps with ragged left-padded prompts), so this path
    changes no sample of audio. `GraphedDecodeEquivalenceTest` pins it.

    Capture is per chunk and costs a warmup plus one step, ~0.2 s, against
    ~14 ms saved on each of a chunk's ~120 steps.
    """

    __slots__ = ("graph", "logits", "codes_buf", "pos_buf", "cache",
                 "_cum0", "_pos0", "_mask", "_seq")

    def __init__(self, *, graph, logits, codes_buf, pos_buf, cache,
                 cum0, pos0, mask, seq) -> None:
        self.graph = graph
        self.logits = logits          # static output, [B, N, V] float32
        self.codes_buf = codes_buf    # static input, [B, N] long
        self.pos_buf = pos_buf        # static input, [B, 1] long
        self.cache = cache
        self._cum0 = cum0             # per-layer cumulative_length after prefill
        self._pos0 = pos0
        self._mask = mask
        self._seq = seq

    def rewind(self) -> None:
        """Undo what warmup and capture advanced.

        Both run the step for real, so the cache write offset, the position ids
        and the mask columns they touched have to go back to their
        post-prefill values or the first replayed step would write into the
        wrong cache slot.
        """
        for layer, saved in zip(self.cache.layers, self._cum0):
            layer.cumulative_length.copy_(saved)
        self.pos_buf.copy_(self._pos0)
        self._mask[:, self._seq:] = 0

    def replay(self) -> None:
        self.graph.replay()


# How often the decode loop asks whether every row has finished. The question
# costs a device->host read, and asking it per step is what the per-row `.cpu()`
# used to hide the cost of. At 4, a chunk runs at most 3 forwards past the point
# every row is done — nothing about the emitted frames changes, since a retired
# row stops emitting. Each row's own cap and the loop's `range(max_cap)` bound
# are still exact.
_SYNC_EVERY = 4

# Cost per decode step, in ms, that the three real renderers hit on the
# reference 3090 (graph + gqa / graph / eager). `_note_chunk_cost` warns above
# `_SLOW_STEP_MS`, which clears the slowest of them by 4x so that an ordinary
# graph fallback is never reported as a degraded accelerator.
_HEALTHY_STEP_MS = 45
_SLOW_STEP_MS = 200.0

# --- fixed decode geometry ------------------------------------------------
#
# Chunks are formed by sorting a job's sentences by cap, so consecutive chunks
# differ in batch width, prompt length AND cap — every chunk is a new geometry,
# so every chunk captured its own CUDA graph. Measured 2026-09-07: ~9 captures
# per job, VRAM climbing 15.4 -> 19.1 GB over four jobs and never coming back,
# ending in `dxgkio_reserve_gpu_va` EOVERFLOW and an 86x decode slowdown. A
# graph's memory pool is never returned to the general allocator, so this grows
# whether the pool is shared or private — the fix is to stop minting geometries.
#
# Rounding each dimension onto a coarse grid collapses a job's chunks onto a
# handful of shapes, and — the part that actually matters — those shapes RECUR
# across jobs, so the captured set stops growing after the first job or two
# instead of growing forever.
#
# Grids are a deliberate trade, not tuning knobs discovered by sweep:
#   B    padding a row costs its attention width for the whole chunk, and
#        watch-along sections can carry as few as 3 sentences, so a coarse
#        grid here would tax the latency-critical path. 4 keeps the worst case
#        at +3 rows.
#   seq  prompt lengths cluster tightly (one pinned ref per speaker), so 128
#        usually collapses a whole job to one or two buckets.
#   cap  `token_cap` is already clamped to [200, 1024], so 256 gives at most 4.
_GEOM_B_GRID = 4
_GEOM_SEQ_GRID = 128
_GEOM_CAP_GRID = 256
#: Cap given to a padding row: it must decode something well-defined (a fully
#: masked row makes softmax produce NaN and would poison the batch), but it
#: should retire immediately. Its output is discarded.
_PAD_ROW_CAP = 8


def _bucket_up(n: int, grid: int, hard_max: int | None = None) -> int:
    """Round `n` up onto `grid`, never below `n`, clamped to `hard_max`.

    Never below `n` is the load-bearing half: the value feeds the attention
    width and the cache length, so rounding DOWN would truncate a real row's
    prompt or cap.
    """
    out = ((n + grid - 1) // grid) * grid
    if hard_max is not None:
        out = min(max(out, n), hard_max)
    return max(out, n)


def _balanced_chunks(items: list[int], size: int) -> list[list[int]]:
    """Split into the FEWEST chunks of at most `size`, balanced in length.

    A chunk costs `max(steps)` over its rows, so a greedy fixed-size split is
    actively harmful at the tail: with 9 cap-sorted requests and `size=8` it
    yields [8, 1], and that final chunk is the single LONGEST line paying a full
    chunk on its own. Measured on the author's 2026-08-25 prelude: a batch of 8
    took 24.3 s and the trailing batch of 1 took **17.3 s**, 42% of the whole
    TTS stage for one sentence.

    Balancing gives [5, 4] instead. Same number of chunks and the same memory
    ceiling (no chunk exceeds `size`), but the first chunk's cost drops from
    "cap of the 8th-shortest" to "cap of the 5th-shortest" while the last chunk
    is dominated by the longest line either way. Because chunks stay contiguous
    slices of the cap-sorted order and every boundary moves earlier or stays
    put, the total is never worse than the greedy split and is usually better.

    Agrees with a greedy split whenever `size` divides the input evenly, so the
    common full-batch case is unchanged.
    """
    n = len(items)
    if n == 0:
        return []
    n_chunks = -(-n // size)          # ceil
    base, rem = divmod(n, n_chunks)
    out: list[list[int]] = []
    i = 0
    for c in range(n_chunks):
        take = base + (1 if c < rem else 0)
        out.append(items[i:i + take])
        i += take
    return out


# The parameter `audio_head.weight` is tied to. Fixed by the checkpoint: its
# `tie_weights()` assigns `audio_head.weight = audio_embedding.weight`.
_TIED_SOURCE = "audio_embedding.weight"


def tolerate_list_tied_weights_keys() -> bool:
    """Let the loader accept the checkpoint's LIST form of `_tied_weights_keys`.
    Returns True if it installed the patch.

    The checkpoint's remote code declares `_tied_weights_keys =
    ["audio_head.weight"]`. Newer transformers treats that attribute as a
    `{target: source}` mapping and calls `.keys()` on it, so loading fails with
    `'list' object has no attribute 'keys'`; older transformers wants the list.

    Deliberately **self-detecting**: try the original first and only coerce after
    it raises AttributeError. An earlier version of this shim rewrote the class
    attribute to a dict up front, which fixed the new contract and would have
    broken the old one — patching for one version of a dependency while
    silently breaking another is not a fix. Trying first means the same code is
    correct on both.

    Also independent of class identity, which matters: dynamic remote code is
    re-copied and re-imported under a hashed module name, so the class object the
    loader ends up using need not be the one we could resolve ahead of time.
    """
    import transformers.modeling_utils as mu

    original = getattr(mu, "_get_tied_weight_keys", None)
    if original is None or getattr(original, "_patmir_list_tolerant", False):
        return False

    def _list_tolerant(module):
        try:
            return original(module)
        except AttributeError:
            for submodule in module.modules():
                keys = getattr(submodule, "_tied_weights_keys", None)
                if isinstance(keys, list):
                    submodule._tied_weights_keys = {k: _TIED_SOURCE for k in keys}
            return original(module)

    _list_tolerant._patmir_list_tolerant = True
    mu._get_tied_weight_keys = _list_tolerant
    return True


class HiggsV3Torch:
    id = "higgs_v3_torch"
    # IVC-style: clones from audio alone; ref_text only improves fidelity.
    ref_text_optional = True
    # Thread concurrency stays 1 — batching is the parallelism here, and the two
    # would fight over one CUDA stream. The adapter prefers `synthesize_batch`.
    tts_concurrency = 1
    # Lazily created per instance (see `probe`). A None class attribute rather
    # than an `__init__` assignment because the equivalence tests build this
    # class with `object.__new__` — and the decode loop reads `self.probe`.
    _probe: Probe | None = None

    @property
    def probe(self) -> Probe:
        """Intra-stage timing, armed only under `patmir bench` (the adapter
        calls `probe.start()` per job).

        The numbers that showed this decode running 4-5x off its
        memory-bandwidth bound came from instrumentation that was then
        reverted, leaving `tts/bench_probe.json` files on disk with no code
        left in the tree to regenerate them. This is that instrumentation, and
        it is disabled — and free — outside a benchmark.
        """
        pr = self._probe
        if pr is None:
            pr = self._probe = Probe()
        return pr

    def __init__(
        self,
        repo_id: str = REPO_DEFAULT,
        device: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int | None = None,
        seed: int = 0,
        max_retries: int = 2,
        max_batch: int = 8,
        cuda_graphs: bool = True,
        fixed_geometry: bool = False,
        decode_attention: str = "sdpa",
        _model: Any = None,
        _tokenizer: Any = None,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        import torch  # lazy (heavy)

        self.repo_id = repo_id
        self.device = _resolve_device(device)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k) if top_k else None
        self.base_seed = int(seed)
        self.max_retries = int(max_retries)
        self.tts_max_batch = max(1, int(max_batch))
        # Capture the decode step as a CUDA graph and replay it. The step is
        # bound by kernel-LAUNCH cost, not by the 4B weights: a 36-layer model
        # with negligible widths measured 42.9 ms/step on this box against the
        # real model's 45.9, so ~93% of the step is the CPU issuing ~36 layers
        # of kernels. A graph replays them from one launch.
        #
        # Bitwise-safe, which was NOT assumed going in and had to be measured:
        # graph replay is byte-identical to eager stepping, and StaticCache is
        # byte-identical to DynamicCache even with ragged left-padded prompts.
        # So this is not a quality trade — see `_capture_decode_graph`.
        self.cuda_graphs = bool(cuda_graphs)
        # Round the decode geometry onto a coarse grid so a job's chunks share
        # shapes. **Defaults OFF: measured 2026-09-07 as a net harm.** Bucketing
        # collapsed a real job from 8 geometries to 5 rather than to 1, while
        # padding raised the VRAM floor ~1 GB — so the address-space ceiling
        # arrived at run 2 instead of run 4. Kept because it is tested and is
        # the groundwork for a single process-wide geometry, which is what the
        # NVIDIA guidance ("reduce the number of graphs") actually asks for.
        # Applied whether or not graphs are on, and that is deliberate:
        # `cuda_graphs` lives in `ui.cache._NON_OUTPUT_KEYS` and carries a bench
        # env override, so anything gated on it can change a rendering under an
        # unchanged cache key — the exact trap the `max_batch` comment in
        # `build()` records. Padding costs the eager path a little width it does
        # not need; keeping one rendering per cache key is worth more.
        self.fixed_geometry = bool(fixed_geometry)
        # "gqa" swaps the single-token decode attention for a grouped matmul
        # that skips transformers' `repeat_kv` expansion; "sdpa" keeps
        # transformers' kernel everywhere.
        #
        # **Defaults to sdpa, and that is a quality decision, not caution.**
        # Unlike the graph, this changes the reduction order, and at
        # temperature 0.8 a one-ulp logit difference flips a multinomial draw:
        # measured, it re-rolls ALL 80 lines of the reference clip, moving
        # individual line durations by up to 1.24 s (total speech within
        # 1.4%). Expected quality is unchanged — it is a different draw from
        # the same distribution — but issue #19 records that this TTS's draws
        # vary by up to 26 dB on utterance-final level, so a re-roll is a
        # re-spin of a known lottery and needs an ear verdict before it becomes
        # the default.
        #
        # It is also why this knob belongs in `patmir.toml` when enabled rather
        # than defaulted on in code: `ui.cache.config_hash` hashes the
        # candidate entry, so setting it there correctly invalidates cached
        # dubs. Flipping it in code would leave one cache key able to hold
        # either rendering.
        #
        # Worth 27.15 -> 17.33 ms/step (agg RTF 0.309 -> 0.277) on top of the
        # graph. Enable with `decode_attention = "gqa"` on this candidate.
        self.decode_attention = str(decode_attention or "sdpa").lower()

        if _model is not None:
            # Injection seam for the equivalence test, which drives a tiny
            # randomly-initialized model on CPU. Nothing else uses it.
            self.model, self.tok = _model, _tokenizer
            self.dtype_name, self.attn_implementation = dtype, attn_implementation
        else:
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM, Qwen2TokenizerFast

            torch_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(dtype, torch.bfloat16)

            snap = snapshot_download(
                repo_id,
                allow_patterns=[
                    "*.safetensors", "*.safetensors.index.json", "config.json",
                    "tokenizer*.json", "chat_template.jinja", "*.py",
                ],
            )
            # Load the CONCRETE tokenizer class from the local snapshot, never
            # AutoTokenizer on the repo id — the trap the MLX candidate
            # documents: AutoTokenizer resolves the repo's custom *config* class
            # first, which prompts for trust_remote_code on a revision change
            # and, with funasr/modelscope imported (the emotion stage), lets
            # modelscope's hf_util patcher redirect the dynamic-module fetch to
            # the slow mirror (ADR 2026-05-26-prefer-fast-model-mirrors). The
            # tokenizer itself is a plain Qwen2 tokenizer with no auto_map.
            self.tok = Qwen2TokenizerFast.from_pretrained(snap)
            # Must happen BEFORE from_pretrained; see the function's docstring.
            tolerate_list_tied_weights_keys()
            # The model class *does* need trust_remote_code — it is the repo's
            # own `modeling_higgs_multimodal_qwen3`. Loading from the local
            # snapshot keeps that resolution off the network.
            model = AutoModelForCausalLM.from_pretrained(
                snap,
                dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
            self.model = model.to(self.device).eval()
            self.dtype_name = dtype
            self.attn_implementation = attn_implementation

        # The checkpoint's own module, reached through the loaded class rather
        # than re-imported: delay-pattern helpers, codec constants and the
        # delay/EOC sampler state machine all come from here so this file holds
        # no second copy of the model's protocol.
        self._up = sys.modules[type(self.model).__module__]
        self.n_codebooks = int(self.model.num_codebooks)

        # Point the BACKBONE at the grouped-attention kernel. Set on the text
        # config only, so the audio codec's own attention (if any) is untouched;
        # and the kernel itself delegates every q_len > 1 call back to
        # transformers, so prefill runs the same code as before.
        self.attn_impl_decode = self.attn_implementation
        if self.decode_attention == "gqa" and _register_gqa_attention():
            try:
                self.model.model.config._attn_implementation = _GQA_ATTN
                self.attn_impl_decode = _GQA_ATTN
            except Exception as exc:  # noqa: BLE001
                print(f"tts: could not select {_GQA_ATTN} "
                      f"({type(exc).__name__}: {exc}) — keeping "
                      f"{self.attn_implementation}")

        # Reference clip -> raw [T, N] codes. Upstream re-encodes the reference
        # on every call; with one pinned ref per speaker that is the same clip
        # re-encoded once per sentence.
        self._ref_cache: dict[str, Any] = {}
        # ONE memory pool shared by every capture, and ONE live capture reused
        # across same-shape chunks. Both exist because a fresh `CUDAGraph()`
        # per chunk reserves its own GPU virtual address range, and under WSL2
        # that address space is the binding resource long before VRAM is:
        # `dxgkio_reserve_gpu_va` starts failing with EOVERFLOW, cudaMalloc
        # then fails with gigabytes still free, and the decode falls back to
        # host-side paths that cost ~1.5 s/step against ~17 ms. See
        # `decisions/2026-09-07-tts-graph-pool-and-gpu-va-exhaustion.md`.
        self._graph_pool: Any = None
        self._graph_holder: Any = None
        self._graph_key: tuple | None = None
        self._chunk_steps = 0
        self._summary: dict[str, Any] = {
            "engine": f"higgs_v3_torch_{dtype}",
            "device": self.device,
            "max_batch": self.tts_max_batch,
            "cuda_graphs": self.cuda_graphs,
            # The kernel that RUNS, not the one asked for: when
            # `_register_gqa_attention` fails, `decode_attention` still reads
            # "gqa" while sdpa is what steps, and the summary named a renderer
            # that never ran.
            "decode_attention": ("gqa" if self.attn_impl_decode == _GQA_ATTN
                                 else self.attn_implementation),
            "sentences": 0,
            "overflows": 0,
            "batches": 0,
            # Present from construction so a reader cannot confuse "no bursts"
            # with "an older candidate that did not count them".
            "onset_bursts": 0,
            "onset_bursts_kept": 0,
            "graph_fallbacks": 0,
            # Captures actually taken vs. chunks served from the live capture.
            # A healthy long job holds `graph_captures` near the number of
            # DISTINCT (B, seq, max_cap) shapes, not near `batches`; a ratio
            # that tracks `batches` means reuse is missing and the GPU address
            # space is being churned again.
            "graph_captures": 0,
            "graph_reuses": 0,
            # Chunks whose measured cost per decode step exceeded
            # `_SLOW_STEP_MS`. Non-zero means the accelerator has degraded
            # under this process and the run is no longer comparable to a
            # baseline — the state that used to be visible only in dmesg.
            "slow_chunks": 0,
            "slowest_ms_per_step": 0.0,
        }

        self._check_audio_head_tied()

    def _check_audio_head_tied(self) -> None:
        """Fail loudly if `audio_head` is not tied to `audio_embedding`.

        The checkpoint deliberately omits `audio_head.weight` — upstream lists it
        in `_tied_weights_keys` and points it at `audio_embedding.weight` in
        `tie_weights()`. transformers therefore reports it on load as:

            audio_head.weight | MISSING | newly initialized

        which reads like a broken checkpoint and is not one; the tie overwrites
        the fresh init afterwards. Verified tied on the 3090 (shared storage,
        identical norms), 2026-08-23.

        The check exists because the failure mode is invisible. If a future
        transformers re-initialized that key *after* tying, the head producing
        every audio logit would be random — and nothing downstream would notice:
        the decode still runs, sentences still terminate, `overflows` still reads
        0, RTF still looks fine. The only symptom is that the output is noise. So
        assert the invariant rather than trusting a load report that actively
        suggests the opposite of the truth.
        """
        enc = getattr(self.model.config, "audio_encoder_config", None) or {}
        if not bool(enc.get("tie_word_embeddings", True)):
            return  # a checkpoint that genuinely ships its own head
        head = getattr(self.model.audio_head, "weight", None)
        embed = getattr(self.model.audio_embedding, "weight", None)
        if head is None or embed is None:
            return  # a stubbed head (tests) — nothing to verify
        if head.data_ptr() != embed.data_ptr():
            raise RuntimeError(
                "Higgs audio_head is NOT tied to audio_embedding, so it holds "
                "randomly-initialized weights and every synthesized line would be "
                "noise. The checkpoint omits `audio_head.weight` by design and "
                "relies on the model's `tie_weights()`; something (most likely a "
                "transformers version change re-initializing missing keys after "
                "tying) has broken that. Fix by calling `model.tie_weights()` "
                "after `from_pretrained`, or pin transformers."
            )

    # ---- reference encoding (cached) ---- #
    def _encode_ref(self, path: str):
        """Reference clip -> raw `[T, N]` codes, cached by path.

        Uses the model's public `get_audio_codec()` rather than its private
        `_encode_reference`, so this stays on the upstream contract.
        """
        cached = self._ref_cache.get(path)
        if cached is not None:
            return cached

        import torch  # lazy

        from patmir.eval.candidates.tts._audio_io import decode_mono

        codec = self.model.get_audio_codec()
        # ffmpeg, not torchaudio.load: on torch >= 2.11 that delegates to
        # torchcodec, whose native libs fail to dlopen on the WSL2 box.
        wav = torch.from_numpy(decode_mono(path, SR))
        # The codec needs at least a second of audio.
        if wav.shape[-1] < SR:
            wav = torch.nn.functional.pad(wav, (0, SR - wav.shape[-1]))
        while wav.ndim < 3:
            wav = wav.unsqueeze(0)
        with torch.no_grad():
            codes_BNT = codec.encode(wav.to(self.device, dtype=torch.float32)).audio_codes
        codes_TN = codes_BNT.squeeze(0).transpose(0, 1).to(torch.long).cpu()
        self._ref_cache[path] = codes_TN
        return codes_TN

    def _prepare(self, text: str, ref_audio: Any, ref_text: str | None,
                 emotion: str | None) -> _Request:
        tag = EMOTION_TAGS.get(emotion) if emotion else None
        prompt_text = f"<|emotion:{tag}|>{text}" if tag else text
        return _Request(
            prompt_text=prompt_text,
            ref_codes=self._encode_ref(str(ref_audio)) if ref_audio else None,
            ref_text=ref_text or None,
            cap=token_cap(prompt_text),
            seed=_request_seed(prompt_text, ref_text),
        )

    # ---- batched autoregressive decode ---- #
    def _acquire_decode_graph(self, *, embeds, mask_full, position_ids, seq,
                              B, N, device, probe):
        """Return `(holder, hidden0, next_pos)` for this chunk, reusing the
        live capture when the chunk has the same shape, else capturing one.

        Returns None on any failure, exactly as capture alone used to, so the
        caller's eager fallback is unchanged.

        **The reuse key is the full captured geometry** — `(B, seq,
        max_cache_len)`. Not a prefix of it, and not "close enough": the graph
        replays fixed addresses, and `_decode_chunk`'s own docstring records
        that the graph attends over `seq + max_cap` columns while the eager
        path attends over `seq + step + 1`, so a graph replayed at a geometry
        it was not captured at would not merely be wrong, it would be
        *silently* wrong — a rendered line with no error anywhere. A mismatch
        therefore drops the capture and takes a new one.

        On reuse the prefill still runs, into the SAME `StaticCache`: the
        capture froze the cache's addresses, so the new chunk's keys and values
        have to land in those buffers. `reset()` zeroes them first, which is
        what makes a reused capture bitwise identical to a fresh one rather
        than merely equivalent — `GraphReuseEquivalenceTest` pins that.
        """
        import torch

        if not self.cuda_graphs or device.type != "cuda":
            return None
        key = (B, seq, int(mask_full.shape[1]), N)
        holder = self._graph_holder
        if holder is not None and self._graph_key == key:
            try:
                with probe.section("prefill_s"):
                    cache = holder.cache
                    cache.reset()
                    for layer in cache.layers:
                        cl = getattr(layer, "cumulative_length", None)
                        if hasattr(cl, "zero_"):
                            cl.zero_()
                    # Adopt the CAPTURED mask buffer: it is the address the
                    # graph reads. Writing the caller's fresh tensor instead
                    # would leave the replay attending over the previous
                    # chunk's mask.
                    holder._mask.copy_(mask_full)
                    out = self.model.model(
                        inputs_embeds=embeds,
                        attention_mask=holder._mask[:, :seq],
                        position_ids=position_ids,
                        use_cache=True,
                        past_key_values=cache,
                    )
                    hidden0 = out.last_hidden_state[:, -1, :]
                    next_pos = position_ids[:, -1] + 1
                    holder._cum0 = [l.cumulative_length.clone()
                                    for l in cache.layers]
                    holder._pos0 = next_pos.unsqueeze(1).clone()
                    holder.rewind()
                self._summary["graph_reuses"] += 1
                return holder, hidden0, next_pos
            except Exception as exc:  # noqa: BLE001
                # A reuse that failed leaves the cache in an unknown state, so
                # the capture is dropped rather than retried against it.
                print(f"tts: decode-graph reuse failed "
                      f"({type(exc).__name__}: {exc}) — recapturing")
                self._release_decode_graph()

        # Shape changed (or nothing captured yet). Free the old capture BEFORE
        # taking a new one, so the shared pool holds one geometry's tensors
        # rather than accumulating every geometry the job has seen.
        self._release_decode_graph()
        captured = self._capture_decode_graph(
            embeds=embeds, mask_full=mask_full, position_ids=position_ids,
            seq=seq, B=B, N=N, device=device, probe=probe)
        if captured is not None:
            self._graph_holder = captured[0]
            self._graph_key = key
        return captured

    def _release_decode_graph(self) -> None:
        """Drop the live capture. The pool itself is kept — it is the arena we
        are deliberately reusing; only the graph and its tensors go."""
        self._graph_holder = None
        self._graph_key = None

    def _capture_decode_graph(self, *, embeds, mask_full, position_ids, seq,
                              B, N, device, probe):
        """Prefill into a `StaticCache` and capture one decode step.

        Returns `(holder, hidden0, next_pos)`, or None whenever graphs are
        unavailable or capture fails — in which case the caller does an
        ordinary prefill and steps eagerly.

        The fallback is EQUIVALENT, not bit-exact, and the difference used to
        be claimed away. Two reasons it diverges: `StaticCache` is
        `is_compileable`, so `create_causal_mask` always materializes a mask,
        which is what routes the step through `_gqa_decode`; the `DynamicCache`
        fallback gets `attention_mask=None` on an unpadded chunk and takes
        plain sdpa. And the graph attends over `seq + max_cap` columns with
        `-inf` padding against the fallback's `seq + step + 1`, so the softmax
        reduces a different width. Same distribution, different draw. A
        failure here therefore costs speed AND re-rolls that chunk, which is
        why the caller counts it in the summary (`graph_fallbacks`) rather
        than only printing it.

        Owning the prefill is deliberate: a `DynamicCache` prefill cannot be
        transplanted into a `StaticCache`, and doing both would pay for it
        twice.
        """
        import torch

        if not self.cuda_graphs or device.type != "cuda":
            return None
        try:
            from transformers import StaticCache
        except ImportError:
            return None

        model = self.model
        max_cache_len = mask_full.shape[1]
        try:
            # Two spans, not one nested in the caller's: `prefill_s` used to
            # wrap this whole call while `graph_capture_s` wrapped the same
            # prefill again, so the two overlapped and summing them
            # double-counted the prefill.
            with probe.section("prefill_s"):
                cache = StaticCache(
                    config=model.config.text_config, max_cache_len=max_cache_len)
                out = model.model(
                    inputs_embeds=embeds,
                    attention_mask=mask_full[:, :seq],
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=cache,
                )
                hidden0 = out.last_hidden_state[:, -1, :]
                next_pos = position_ids[:, -1] + 1
            with probe.section("graph_capture_s"):
                codes_buf = torch.zeros(B, N, dtype=torch.long, device=device)
                pos_buf = next_pos.unsqueeze(1).clone()
                cum0 = [l.cumulative_length.clone() for l in cache.layers]
                pos0 = pos_buf.clone()

                def _step():
                    step_embed = model.audio_embedding(codes_buf).unsqueeze(1).to(
                        embeds.dtype)
                    o = model.model(
                        inputs_embeds=step_embed,
                        attention_mask=mask_full,
                        position_ids=pos_buf,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    lg = model.audio_head(o.last_hidden_state[:, -1, :]).to(
                        torch.float32)
                    pos_buf.add_(1)
                    return lg

                holder = _DecodeGraph(
                    graph=None, logits=None, codes_buf=codes_buf,
                    pos_buf=pos_buf, cache=cache, cum0=cum0, pos0=pos0,
                    mask=mask_full, seq=seq)

                # PyTorch requires warmup on a side stream before capture.
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        _step()
                torch.cuda.current_stream().wait_stream(side)
                holder.rewind()

                graph = torch.cuda.CUDAGraph()
                # `pool=` is the whole point. Without it each capture takes a
                # PRIVATE memory pool, which is a fresh GPU virtual address
                # reservation that outlives the graph's usefulness; thousands
                # of those exhaust the WSL2 address space. One handle, created
                # once and reused, keeps every capture in the same arena. The
                # captured computation is unchanged, so output is unchanged.
                if self._graph_pool is None:
                    self._graph_pool = torch.cuda.graph_pool_handle()
                with torch.cuda.graph(graph, pool=self._graph_pool):
                    logits_static = _step()
                holder.graph = graph
                holder.logits = logits_static
                holder.rewind()
            self._summary["graph_captures"] += 1
            return holder, hidden0, next_pos
        except Exception as exc:  # noqa: BLE001
            print(f"tts: CUDA graph capture unavailable "
                  f"({type(exc).__name__}: {exc}) — eager decode")
            return None

    def _decode_chunk(self, requests: list[_Request],
                      attempt: int = 0) -> list[tuple[Any, bool]]:
        """Decode a chunk of requests together.

        Returns one `(waveform | None, overflowed)` per request, in order.
        `None` means the run produced fewer than `num_codebooks` rows, i.e. it
        collapsed before emitting anything de-deloadable.

        `attempt` shifts every row's RNG seed, so an overflow retry redraws
        instead of reproducing the run that just failed to terminate.
        """
        import torch

        up = self._up
        model = self.model
        N = self.n_codebooks
        device = next(model.parameters()).device
        B = len(requests)

        # ---- per-row prefill, then left-pad into one batch ---- #
        probe = self.probe
        per_row: list[Any] = []
        for req in requests:
            delayed = None
            if req.ref_codes is not None:
                delayed = up.apply_delay_pattern(req.ref_codes.to(torch.long))
            prompt_ids = model._build_prompt_ids(
                self.tok, req.prompt_text,
                num_ref_tokens=0 if delayed is None else delayed.shape[0],
                reference_text=req.ref_text,
            )
            per_row.append(model._prefill_embeds(prompt_ids, delayed))  # [1, S_b, D]

        hidden_dim = per_row[0].shape[2]
        # Fixed geometry: round the three dimensions that define a capture onto
        # a coarse grid so chunks share shapes instead of minting one each.
        # Only when graphs are on — with eager stepping the padding would be
        # compute spent for nothing.
        B_real = len(requests)
        if self.fixed_geometry:
            B = _bucket_up(B_real, _GEOM_B_GRID, hard_max=max(self.tts_max_batch,
                                                              B_real))
            seq = _bucket_up(max(e.shape[1] for e in per_row), _GEOM_SEQ_GRID)
            max_cap = _bucket_up(max(r.cap for r in requests), _GEOM_CAP_GRID,
                                 hard_max=1024)
        else:
            B = B_real
            seq = max(e.shape[1] for e in per_row)
            max_cap = max(r.cap for r in requests)
        embeds = torch.zeros(B, seq, hidden_dim, dtype=per_row[0].dtype, device=device)
        # Allocated once at its final width and filled one column at a time. The
        # loop used to `torch.cat` a column onto the mask every step, which
        # reallocated and copied a [B, seq+step] tensor ~1300 times per chunk;
        # a stable address is also a precondition for capturing the step.
        mask_full = torch.zeros(B, seq + max_cap, dtype=torch.long, device=device)
        for b, e in enumerate(per_row):
            s_b = e.shape[1]
            # LEFT padding: the last position is then a real token for every
            # row, so one `last_hidden_state[:, -1]` is correct for the batch.
            embeds[b, seq - s_b:] = e[0]
            mask_full[b, seq - s_b:seq] = 1
        for b in range(B_real, B):
            # A padding row is a COPY of a real one, not zeros: a row with an
            # all-zero mask makes the softmax reduce an all -inf window and
            # produce NaN, which propagates through the batched matmul and
            # would corrupt the real rows' logits. Its output is discarded and
            # `_PAD_ROW_CAP` retires it within a few steps.
            embeds[b] = embeds[0]
            mask_full[b, :seq] = mask_full[0, :seq]
        mask = mask_full[:, :seq]

        # Derived from the mask, not from cache_position: with left padding the
        # two disagree, and RoPE has to see each row's own token positions.
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)

        with torch.no_grad():
            # The graph path prefills into its own StaticCache, so only one
            # prefill runs either way.
            graph = None
            captured = self._acquire_decode_graph(
                embeds=embeds, mask_full=mask_full,
                position_ids=position_ids, seq=seq, B=B, N=N,
                device=device, probe=probe)
            if captured is not None:
                graph, hidden, next_pos = captured
                past = None
                # A REUSED capture reads its own mask buffer, not the one this
                # call just built. Rebinding here means the step loop's
                # `mask_full[:, seq + step] = 1` opens the column the graph
                # actually attends over; without it a reused graph would run
                # against a mask frozen at the previous chunk's.
                mask_full = graph._mask
            else:
                # A capture that failed AFTER its prefill records `prefill_s`
                # twice, which is right: two prefills ran.
                with probe.section("prefill_s"):
                    out = model.model(
                        inputs_embeds=embeds,
                        attention_mask=mask,
                        position_ids=position_ids,
                        use_cache=True,
                        cache_position=torch.arange(seq, device=device),
                    )
                    past = out.past_key_values
                    hidden = out.last_hidden_state[:, -1, :]      # [B, D]
                    next_pos = position_ids[:, -1] + 1             # [B]
            probe.set("cuda_graphs", graph is not None)
            if graph is None and self.cuda_graphs and device.type == "cuda":
                # This chunk wanted a graph and did not get one, so it renders
                # differently from its neighbours (see `_capture_decode_graph`).
                # A stderr line does not reach the manifest; this does.
                self._summary["graph_fallbacks"] += 1
                probe.bump("graph_fallback_chunks", 1)

            # One RNG stream per row, seeded from that request's own content, so
            # a sentence's audio cannot depend on its batch-mates or its slot.
            gens = []
            for req in requests:
                g = torch.Generator(device=device)
                g.manual_seed((self.base_seed + attempt + req.seed) & ((1 << 62) - 1))
                gens.append(g)
            for b in range(B_real, B):
                # A padding row is drawn for like any other (the sampler stacks
                # over the full batch), but from its OWN stream, so it cannot
                # consume draws that belong to a real sentence and change what
                # that sentence renders.
                g = torch.Generator(device=device)
                g.manual_seed((self.base_seed + attempt + 0x9E37_79B9 + b)
                              & ((1 << 62) - 1))
                gens.append(g)

            caps = torch.tensor([r.cap for r in requests]
                                + [_PAD_ROW_CAP] * (B - B_real),
                                dtype=torch.long, device=device)
            state = _DelayState(batch=B, num_codebooks=N, caps=caps,
                                boc_id=up.BOC_ID, eoc_id=up.EOC_ID, device=device)
            # `emitted[b, s]` is row b's s-th emitted frame. A row emits on every
            # step from 0 until it retires, so its s-th frame IS step s and no
            # index bookkeeping is needed. This replaces a `list[list[Tensor]]`
            # appended to with `codes_N.cpu()` — a device->host sync per row per
            # step, which is most of the gap between this loop's ~46 ms/step and
            # the few ms of kernel time it actually issues.
            emitted = torch.zeros(B, max_cap, N, dtype=torch.long, device=device)
            fed = torch.zeros(B, N, dtype=torch.long, device=device)
            cache_pos = torch.arange(seq, seq + max_cap, device=device)

            steps_run = 0
            with probe.section("decode_s"):
                for step in range(max_cap):
                    steps_run = step + 1
                    # `sample_s` / `forward_s` split only under
                    # `--tts-probe step`: it needs a device sync per step, which
                    # inflates the very total it explains. `decode_s` is the
                    # number to quote.
                    with probe.step_section("sample_s"):
                        if graph is None or step == 0:
                            # Step 0's logits come from the prefill's hidden
                            # state, which is outside the captured region.
                            logits = model.audio_head(hidden).to(torch.float32)
                        else:
                            # Produced by the previous iteration's replay.
                            logits = graph.logits                             # [B, N, V]
                        if self.temperature <= 1e-5:
                            # Upstream's argmax path consumes no randomness, so
                            # neither does this one — which is what lets the
                            # equivalence test compare batched against
                            # standalone rows at temperature 0.
                            codes = logits.argmax(dim=-1).to(torch.long)
                        else:
                            probs = _filter_probs(
                                logits, temperature=self.temperature,
                                top_p=self.top_p, top_k=self.top_k)
                            # Filtering is batched; the DRAW stays per row, from
                            # that row's own generator, in slot order. Retired
                            # rows are drawn for as well: their generators are
                            # never read again, and drawing unconditionally
                            # keeps this one `stack` rather than a branch per row.
                            codes = torch.stack([
                                probs[b].multinomial(num_samples=1, generator=gens[b])
                                .squeeze(-1)
                                for b in range(B)
                            ]).to(torch.long)                                 # [B, N]

                        emitting = state.step(codes, step)
                        emitted[:, step] = codes
                        # A retired row keeps being fed its last emitted frame so
                        # the batch shape holds. Attention is per-sequence, so
                        # this cannot affect the other rows.
                        fed = torch.where(emitting.unsqueeze(1), codes, fed)

                    if step + 1 >= max_cap:
                        break
                    if (step + 1) % _SYNC_EVERY == 0 and state.all_done():
                        break

                    with probe.step_section("forward_s"):
                        mask_full[:, seq + step] = 1
                        if graph is not None:
                            # One launch for the whole 36-layer step. The codes
                            # buffer is a captured input, so feeding it is a
                            # copy into a fixed address.
                            graph.codes_buf.copy_(fed)
                            graph.replay()
                        else:
                            step_embed = model.audio_embedding(fed).unsqueeze(1).to(
                                embeds.dtype)
                            out = model.model(
                                inputs_embeds=step_embed,
                                attention_mask=mask_full[:, :seq + step + 1],
                                position_ids=next_pos.unsqueeze(1),
                                past_key_values=past,
                                use_cache=True,
                                cache_position=cache_pos[step:step + 1],
                            )
                            past = out.past_key_values
                            hidden = out.last_hidden_state[:, -1, :]
                            next_pos = next_pos + 1

            # The one device->host read of the whole chunk.
            lengths = state.length.tolist()
            overflow_flags = state.overflowed.tolist()
            emitted_host = emitted.cpu()

            results: list[tuple[Any, bool]] = []
            with probe.section("codec_s"):
                # `B_real`, not `B`: padding rows exist only to hold the
                # captured geometry and must never reach the caller, which maps
                # results positionally onto its request slots.
                for b in range(B_real):
                    if lengths[b] < N:
                        results.append((None, bool(overflow_flags[b])))
                        continue
                    delayed_LN = emitted_host[b, :lengths[b]]
                    codes_TN = up.reverse_delay_pattern(delayed_LN)
                    # `_decode_codes` moves its input to the device itself, and
                    # the pre-2026-09-02 path handed it a CPU tensor. Unchanged.
                    results.append((model._decode_codes(codes_TN),
                                    bool(overflow_flags[b])))
            # `live_row_steps` vs `total_row_steps` answers whether compacting
            # retired rows out of the batch mid-flight would be worth it: such a
            # row still occupies its slot and still pays its KV read until the
            # whole chunk drains.
            # Tracked on the instance as well as the probe: the probe is off in
            # production, and the degradation this feeds is exactly the thing
            # that only ever shows up in production.
            self._chunk_steps += steps_run
            probe.bump("steps", steps_run)
            probe.bump("chunks", 1)
            probe.bump("live_row_steps", sum(lengths[:B_real]))
            probe.bump("total_row_steps", B * steps_run)
            probe.detail({
                "B": B, "B_real": B_real, "seq": seq, "steps": steps_run,
                "caps": [r.cap for r in requests],
                "lengths": lengths[:B_real],
                "attempt": attempt,
            })
            return results

    def _decode_chunk_traced(self, requests: list[_Request],
                             attempt: int = 0) -> list[tuple[Any, bool]]:
        """`_decode_chunk` under `torch.profiler`, written as a chrome trace.

        Separate method so the hot path carries no profiler branch, and so a
        profiler that fails to initialize degrades to an ordinary decode rather
        than failing the stage.
        """
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile
        except Exception:  # noqa: BLE001
            return self._decode_chunk(requests, attempt)
        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        path = self.probe.trace_path
        try:
            with profile(activities=acts) as prof:
                out = self._decode_chunk(requests, attempt)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(path))
                self.probe.set("trace", str(path))
        except Exception as exc:  # noqa: BLE001
            print(f"tts: profiler trace failed ({type(exc).__name__}: {exc})")
            return self._decode_chunk(requests, attempt)
        return out

    def _probe_finalize(self) -> None:
        """Derive the per-step rates and stamp the run's identity.

        `ms_per_step_decode` is the headline: the whole loop's wall over the
        steps it actually ran, with no per-step syncs in it. The forward/sample
        split beside it exists only at `--tts-probe step` and is inflated by
        its own instrumentation — attribute with it, quote the other.
        """
        pr = self.probe
        if not pr.enabled:
            return
        steps = pr.counters.get("steps", 0)
        if steps:
            for span, name in (("decode_s", "ms_per_step_decode"),
                               ("forward_s", "ms_per_step_forward"),
                               ("sample_s", "ms_per_step_sample")):
                if span in pr.spans:
                    pr.set(name, round(1000 * pr.spans[span] / steps, 2))
        total_rows = pr.counters.get("total_row_steps", 0)
        if total_rows:
            pr.set("live_row_fraction",
                   round(pr.counters.get("live_row_steps", 0) / total_rows, 3))
        pr.set("attn_impl", self.attn_implementation)
        pr.set("attn_impl_decode", getattr(self, "attn_impl_decode",
                                           self.attn_implementation))
        pr.set("max_batch", self.tts_max_batch)
        pr.set("dtype", self.dtype_name)
        pr.set("device", self.device)

    def _note_chunk_cost(self, wall_s: float) -> None:
        """Flag a chunk that cost far more per decode step than any healthy
        path on this hardware.

        The three real renderers measure 17.33 ms/step (graph + gqa), 27.15
        (graph) and 45.55 (eager), so `_SLOW_STEP_MS` sits above the slowest of
        them by 4x and cannot fire on an ordinary fallback. What it does catch
        is the accelerator degrading underneath a long-lived process — measured
        at ~1500 ms/step, i.e. 33x the eager path — which previously appeared
        nowhere: not in the manifest, not in the job record, not in the stage
        log. Only `dmesg` knew, and only if you thought to look.

        Reported, never acted on. Aborting the job would turn a slow dub into
        no dub, and the operator is the one who decides whether to restart.
        """
        steps = getattr(self, "_chunk_steps", 0)
        if steps <= 0 or wall_s <= 0:
            return
        ms = 1000.0 * wall_s / steps
        # `.get`/`setdefault` rather than direct indexing: a candidate built
        # for orchestration tests carries a minimal summary, and a watchdog
        # that raised there would fail the run it exists to report on.
        if ms > self._summary.get("slowest_ms_per_step", 0.0):
            self._summary["slowest_ms_per_step"] = round(ms, 1)
        if ms < _SLOW_STEP_MS:
            return
        self._summary["slow_chunks"] = self._summary.get("slow_chunks", 0) + 1
        if self._summary["slow_chunks"] == 1:
            # Once per job, with the diagnosis attached: a second reader should
            # not have to rediscover what this means.
            print(
                f"tts: decode is {ms:.0f} ms/step against ~{_HEALTHY_STEP_MS} ms "
                f"expected ({steps} steps in {wall_s:.1f} s). The accelerator "
                f"has degraded under this process — on WSL2 check "
                f"`dmesg | grep reserve_gpu_va` for EOVERFLOW, which means the "
                f"GPU address space is exhausted and only restarting the "
                f"server clears it.",
                file=sys.stderr,
            )

    # ---- public API (driven by SentenceTTS adapter) ---- #
    def _synthesize_chunk(self, chunk_i: int, chunk: list[int],
                          prepared: list[Any], audio: list[Any]) -> None:
        """Decode one chunk into `audio`, redrawing rows that overflow or open
        on a burst. A method rather than a loop body so the test that pins the
        retry contract runs THIS code and not a hand-copied replica of it.

        Overflow and onset-burst redraws share `max_retries`: `attempt` is the
        only reseed, so a row that overflowed twice and then bursts ships
        bursting with no redraw for the burst. Stated rather than hidden; a
        separate budget is a policy call, not a bug fix.
        """
        pending = list(chunk)
        self._chunk_steps = 0
        # Slots already counted as bursting in this chunk, so the count is per
        # LINE. Counting attempts inflated the "2 of 77" rate the ADR quotes
        # whenever one row burst twice. `overflows` keeps counting attempts —
        # it is a pre-existing series and older runs are compared against it.
        burst_seen: set[int] = set()
        for attempt in range(self.max_retries + 1):
            if not pending:
                break
            self._summary["batches"] += 1
            # A profiler trace covers the FIRST chunk only: tracing a whole
            # job writes hundreds of MB, and every chunk runs the same loop.
            if chunk_i == 0 and attempt == 0 and self.probe.trace_path is not None:
                outs = self._decode_chunk_traced(
                    [prepared[i] for i in pending], attempt)
            else:
                outs = self._decode_chunk([prepared[i] for i in pending], attempt)
            retry: list[int] = []
            last_attempt = attempt >= self.max_retries
            for slot, (wav, over) in zip(pending, outs):
                burst = wav is not None and not over and _has_onset_burst(wav)
                if wav is not None and not over and not burst:
                    audio[slot] = wav
                    continue
                if over:
                    self._summary["overflows"] += 1
                if burst:
                    if slot not in burst_seen:
                        burst_seen.add(slot)
                        self._summary["onset_bursts"] += 1
                    # Keep the audio: on the final attempt a redrawn-but-
                    # still-bursting line is far better than the silence
                    # the `None` path would leave, and the mux trims the
                    # head as a second line of defence.
                    audio[slot] = wav
                    if last_attempt:
                        self._summary["onset_bursts_kept"] += 1
                        continue
                retry.append(slot)
            pending = retry

    def synthesize_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        on_chunk: Any = None,
    ) -> list[bytes]:
        """Synthesize many sentences, batching the decode. Order is preserved.

        Each request is `{"text": str, "ref_audio": ..., "ref_text": ...,
        "emotion": ...}`; every key but `text` is optional.

        `on_chunk(indices, wall_s)` is called after each decode chunk with the
        request indices it covered and its wall time — the adapter uses it for
        progress and for per-chunk bench timing, since the whole list has to
        arrive in one call for the length sort to be global.
        """
        import time

        import torch

        prepared = [
            self._prepare(
                r["text"], r.get("ref_audio"), r.get("ref_text"), r.get("emotion")
            )
            for r in requests
        ]
        audio: list[Any] = [None] * len(prepared)

        # Sort by cap before chunking: a chunk costs max(steps), not mean, so
        # grouping similar lengths is what makes batching pay.
        order = sorted(range(len(prepared)), key=lambda i: prepared[i].cap)
        for chunk_i, chunk in enumerate(_balanced_chunks(order, self.tts_max_batch)):
            started = time.perf_counter()
            self._synthesize_chunk(chunk_i, chunk, prepared, audio)
            self._note_chunk_cost(time.perf_counter() - started)
            if on_chunk is not None:
                # Retries are folded into the chunk's wall deliberately: a
                # reseed is real time spent on these sentences.
                on_chunk(list(chunk), time.perf_counter() - started)

        self._summary["sentences"] += len(prepared)
        self._probe_finalize()
        out: list[bytes] = []
        for wav in audio:
            if wav is None:
                # Short silence so the pipeline keeps going rather than aborting
                # the job; the mux places a gap. NOTE a deliberate divergence
                # from the MLX candidate: when the final attempt overflows, that
                # one keeps and decodes the overflowed codes, while this returns
                # silence. An overflowed run is a long garbage line, and under
                # `voice_no_overlap` the mux would ripple every later line
                # behind it — one gap is less damage than a cascade.
                out.append(_wav_bytes(np.zeros(int(SR * 0.2), dtype=np.float32), SR))
            else:
                out.append(_wav_bytes(np.asarray(wav, dtype=np.float32), SR))
        return out

    def synthesize(self, text: str, *, ref_audio: Any = None, ref_text: str | None = None,
                   emotion: str | None = None) -> bytes:
        """Single-sentence entry point, for adapters/candidates that don't batch.

        Deliberately a batch of one rather than a second decode implementation —
        two loops would be two things to keep correct.
        """
        return self.synthesize_batch([{
            "text": text, "ref_audio": ref_audio, "ref_text": ref_text, "emotion": emotion,
        }])[0]

    def summarize(self) -> dict[str, Any]:
        return dict(self._summary)

    def reset_summary(self) -> None:
        """Zero the per-job counters. Called by the TTS adapter at the start of
        each job so a warm-cached instance (watch-along stage_cache) reports
        per-section numbers instead of accumulating across sections."""
        # Every per-job counter, not a subset: `onset_bursts` was added later
        # and missed here, so a warm watch-along instance reported section 5's
        # sentence count against the whole film's burst count.
        self._summary.update({
            "sentences": 0, "overflows": 0, "batches": 0,
            "onset_bursts": 0, "onset_bursts_kept": 0, "graph_fallbacks": 0,
            # The graph and watchdog counters, added 2026-09-07 — and missed
            # here on the first pass, which is the very mistake the paragraph
            # above records. `GraphCounterResetTest` now pins the whole set, so
            # the next counter cannot repeat it silently.
            "graph_captures": 0, "graph_reuses": 0,
            "slow_chunks": 0, "slowest_ms_per_step": 0.0,
        })



# --- onset-burst guard ---------------------------------------------------
#
# The decoder sometimes opens an utterance with a loud pitch-free impulse
# instead of speech — heard by the author as "a muffled gun shot" before the
# first word, and audible mainly on a job's first line because that line is
# the one with no lead-in to mask it.
#
# Root cause, isolated 2026-09-03 by replaying one job's 77 requests through
# `synthesize_batch` directly: with `ref_text` passed (what the adapter does)
# 2 of 77 lines burst, including the first; with `ref_text` empty and
# everything else identical, 0 of 77. It is the reference transcript in the
# cloning prompt — a Russian sentence conditioning an Armenian target — not
# the reference audio, not the batch composition, and not the decode path
# (single-row, batch-of-12 and all-77 all reproduce or clear it purely on
# whether `ref_text` is present).
#
# Rather than drop `ref_text` — which is real cloning conditioning whose
# removal is a quality question, not a bug fix — a bursting row is treated
# exactly like an overflow and **redrawn**: `attempt + 1` reseeds that row's
# generator, which is already how non-termination is handled. Conditioning is
# untouched, so voice similarity cannot regress, and at ~2.6% of lines the
# retry cost is negligible.
#
# The test is deliberately narrow, because a word-initial voiceless fricative
# (/s/, /ʃ/, /h/) is pitch-free too and must survive: the first 5 ms has to
# reach 40% of the clip's own peak, which speech attacks do not — the median
# first-frame amplitude over 3040 measured non-first lines is 0.005 of peak.
_ONSET_FRAME_SECONDS = 0.005
_ONSET_PEAK_FRACTION = 0.4
_ONSET_TONAL_RATIO = 0.05


def _has_onset_burst(wav: Any, sr: int = SR) -> bool:
    """True if a waveform opens with a loud impulse carrying no pitch."""
    import numpy as np

    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = max(8, int(_ONSET_FRAME_SECONDS * sr))
    if a.size < 4 * n:
        return False
    peak = float(np.abs(a).max())
    if peak <= 0.0:
        return False
    head = a[:n]
    if float(np.abs(head).max()) < _ONSET_PEAK_FRACTION * peak:
        return False
    spec = np.abs(np.fft.rfft(head * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    high = spec[freqs >= 300.0].sum()
    if high <= 0.0:
        return False
    low = spec[(freqs > 20.0) & (freqs < 300.0)].sum()
    return bool(low / high < _ONSET_TONAL_RATIO)


def _wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean override from the environment.

    Present so a perf A/B can flip one lever between two runs without editing
    `patmir.toml` — which would otherwise put a config change in the middle of
    a measurement series.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def build(config: dict[str, Any]) -> HiggsV3Torch:
    return HiggsV3Torch(
        repo_id=config.get("repo_id", REPO_DEFAULT),
        device=str(config.get("device", "auto")),
        dtype=str(config.get("dtype", "bfloat16")),
        attn_implementation=str(config.get("attn_implementation", "sdpa")),
        temperature=float(config.get("temperature", 0.8)),
        top_p=float(config.get("top_p", 0.95)),
        top_k=config.get("top_k"),
        seed=int(config.get("seed", 0)),
        max_retries=int(config.get("max_retries", 2)),
        # `max_batch` and `decode_attention` change the audio (batching is not
        # bitwise-invariant across shapes; the gqa kernel reduces in a different
        # order), and `ui.cache.config_hash` reads only the TOML entry — so
        # they come from the entry and nowhere else. The env overrides they
        # once had let a bench run write a differently rendered dub under an
        # unchanged cache key. Sweep them by editing `patmir.toml`.
        max_batch=int(config.get("max_batch", 8)),
        # From the entry and nowhere else, same rule as `max_batch` above.
        # This had a bench-only env override (`PATMIR_TTS_CUDA_GRAPHS`), taken
        # to be safe because `cuda_graphs` sat in `_NON_OUTPUT_KEYS`. Measured
        # 2026-09-07: flipping it re-renders **24 of 24** lines, with same-flag
        # repeats byte-identical as controls. So the override let a bench run
        # write a differently rendered dub under an unchanged cache key —
        # precisely the failure the `max_batch` note above records. The key now
        # contains `cuda_graphs`, and the env override is gone so nothing can
        # route around it.
        cuda_graphs=bool(config.get("cuda_graphs", True)),
        # Output-affecting (padding changes the batch shape, and batching is
        # not bitwise-invariant across shapes), so it comes from the entry with
        # NO env override — same rule as `max_batch` above.
        fixed_geometry=bool(config.get("fixed_geometry", False)),
        decode_attention=str(config.get("decode_attention", "sdpa")),
    )
