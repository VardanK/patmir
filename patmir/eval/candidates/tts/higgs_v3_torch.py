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
* **Sampling is not batch-invariant.** Rows share one global RNG stream, so a
  sentence's output depends on the batch it landed in. Expected for sampled
  decoding, and the reason the equivalence test pins temperature to 0.

Both Higgs candidates stay registered; `[pipeline].tts` (or a machine profile)
picks one. ADRs: `2026-06-20-higgs-v3-tts-mlx-q8` for the model,
`2026-08-23-cuda-port-machine-profiles` for this runtime.
"""
from __future__ import annotations

import io
import os
import sys
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

REPO_DEFAULT = "multimodalart/higgs-audio-v3-tts-4b-transformers"

# Output sample rate, fixed by the checkpoint's codec.
SR = 24000

# Pipeline emotion bins -> Higgs inline emotion tags. Kept identical to
# `higgs_v3_mlx.EMOTION_TAGS` so a runtime swap can't change delivery
# (asserted in test_config_profiles.RuntimeParityTest).
EMOTION_TAGS = {
    "happy": "elation",
    "sad": "sadness",
    "angry": "anger",
    "fearful": "fear",
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


def _chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


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

        # Reference clip -> raw [T, N] codes. Upstream re-encodes the reference
        # on every call; with one pinned ref per speaker that is the same clip
        # re-encoded once per sentence.
        self._ref_cache: dict[str, Any] = {}
        self._summary: dict[str, Any] = {
            "engine": f"higgs_v3_torch_{dtype}",
            "device": self.device,
            "max_batch": self.tts_max_batch,
            "sentences": 0,
            "overflows": 0,
            "batches": 0,
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
        )

    # ---- batched autoregressive decode ---- #
    def _decode_chunk(self, requests: list[_Request]) -> list[tuple[Any, bool]]:
        """Decode a chunk of requests together.

        Returns one `(waveform | None, overflowed)` per request, in order.
        `None` means the run produced fewer than `num_codebooks` rows, i.e. it
        collapsed before emitting anything de-deloadable.
        """
        import torch

        up = self._up
        model = self.model
        N = self.n_codebooks
        device = next(model.parameters()).device
        B = len(requests)

        # ---- per-row prefill, then left-pad into one batch ---- #
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

        seq = max(e.shape[1] for e in per_row)
        hidden_dim = per_row[0].shape[2]
        embeds = torch.zeros(B, seq, hidden_dim, dtype=per_row[0].dtype, device=device)
        mask = torch.zeros(B, seq, dtype=torch.long, device=device)
        for b, e in enumerate(per_row):
            s = e.shape[1]
            # LEFT padding: the last position is then a real token for every
            # row, so one `last_hidden_state[:, -1]` is correct for the batch.
            embeds[b, seq - s:] = e[0]
            mask[b, seq - s:] = 1

        # Derived from the mask, not from cache_position: with left padding the
        # two disagree, and RoPE has to see each row's own token positions.
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)

        with torch.no_grad():
            out = model.model(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=True,
                cache_position=torch.arange(seq, device=device),
            )
            past = out.past_key_values
            hidden = out.last_hidden_state[:, -1, :]          # [B, D]
            next_pos = position_ids[:, -1] + 1                 # [B]

            states = [up._SamplerState(num_codebooks=N) for _ in range(B)]
            rows: list[list[Any]] = [[] for _ in range(B)]
            done = [False] * B
            overflowed = [False] * B
            codes = torch.zeros(B, N, dtype=torch.long, device=device)

            for step in range(max(r.cap for r in requests)):
                logits = model.audio_head(hidden).to(torch.float32)   # [B, N, V]
                for b in range(B):
                    if done[b]:
                        # Keep feeding this row's last codes so the batch shape
                        # holds; its outputs are no longer read. Attention is
                        # per-sequence, so this cannot affect the other rows.
                        continue
                    codes_N = up._sampler_step(
                        logits[b], states[b],
                        temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
                    )
                    if states[b].generation_done:
                        done[b] = True
                        continue
                    codes[b] = codes_N
                    rows[b].append(codes_N.cpu())
                    if len(rows[b]) >= requests[b].cap:
                        # Ran to its own cap without emitting a stop. Owning the
                        # loop means this is an exact signal, not an inference
                        # from the output's length.
                        done[b] = True
                        overflowed[b] = True
                if all(done):
                    break

                step_embed = model.audio_embedding(codes).unsqueeze(1).to(embeds.dtype)
                mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=device)], dim=1)
                out = model.model(
                    inputs_embeds=step_embed,
                    attention_mask=mask,
                    position_ids=next_pos.unsqueeze(1),
                    past_key_values=past,
                    use_cache=True,
                    cache_position=torch.tensor([seq + step], device=device),
                )
                past = out.past_key_values
                hidden = out.last_hidden_state[:, -1, :]
                next_pos = next_pos + 1

            results: list[tuple[Any, bool]] = []
            for b in range(B):
                if len(rows[b]) < N:
                    results.append((None, overflowed[b]))
                    continue
                delayed_LN = torch.stack(rows[b], dim=0)
                codes_TN = up.reverse_delay_pattern(delayed_LN)
                results.append((model._decode_codes(codes_TN), overflowed[b]))
            return results

    # ---- public API (driven by SentenceTTS adapter) ---- #
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
        for chunk in _chunks(order, self.tts_max_batch):
            started = time.perf_counter()
            pending = list(chunk)
            for attempt in range(self.max_retries + 1):
                if not pending:
                    break
                torch.manual_seed(self.base_seed + attempt)
                self._summary["batches"] += 1
                outs = self._decode_chunk([prepared[i] for i in pending])
                retry: list[int] = []
                for slot, (wav, over) in zip(pending, outs):
                    if wav is not None and not over:
                        audio[slot] = wav
                        continue
                    if over:
                        self._summary["overflows"] += 1
                    retry.append(slot)
                pending = retry
            if on_chunk is not None:
                # Retries are folded into the chunk's wall deliberately: a
                # reseed is real time spent on these sentences.
                on_chunk(list(chunk), time.perf_counter() - started)

        self._summary["sentences"] += len(prepared)
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
        self._summary.update({"sentences": 0, "overflows": 0, "batches": 0})


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
        max_batch=int(config.get("max_batch", 8)),
    )
