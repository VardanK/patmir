"""Higgs Audio v3 TTS (4B) on PyTorch — the CUDA path.

Sibling of `higgs_v3_mlx.py`. Same model, same voice cloning, same inline
emotion tags; the difference is who runs the autoregressive decode.

On Apple Silicon we had to hand-port the AR graph to MLX because torch-MPS
cannot do fast quantized matmul, and the decode is memory-bandwidth-bound on
the 4B weights (~69 ms/step at torch-MPS bf16 vs ~28 ms/step at MLX q8). On a
discrete NVIDIA GPU that whole argument dissolves: bf16 is native, bandwidth is
~4.7x the M1 Pro's, and the upstream repo already ships a complete transformers
implementation (`HiggsMultimodalQwen3ForConditionalGeneration.generate_speech`).
So this candidate is a thin wrapper over upstream rather than a second port —
no hand-maintained AR loop, no weight remap, no delay-pattern sampler of our
own.

What this adds on top of `generate_speech`:

  * reference-code caching, so cloning a speaker encodes their ref clip once per
    job instead of once per sentence (upstream re-encodes every call);
  * the emotion-bin -> inline `<|emotion:..|>` tag mapping, shared with the MLX
    candidate;
  * the non-termination guardrail — a length-proportional token cap plus
    reseed-and-retry (see `_looks_overflowed`).

Both candidates are registered; `[pipeline].tts` (or a machine profile) picks
one. ADR: docs/architecture/decisions/2026-06-20-higgs-v3-tts-mlx-q8.md for the
model choice, and the CUDA-port ADR for this runtime.
"""
from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_DEFAULT = "multimodalart/higgs-audio-v3-tts-4b-transformers"

# Fixed by the checkpoint (mirrored from the upstream modeling module so we
# don't import private names): 8 codebooks, 24 kHz output.
N_CB, SR = 8, 24000

# Pipeline emotion bins -> Higgs inline emotion tags. Kept identical to
# `higgs_v3_mlx.EMOTION_TAGS` so a runtime swap can't change delivery.
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


class HiggsV3Torch:
    id = "higgs_v3_torch"
    # IVC-style: clones from audio alone; ref_text only improves fidelity.
    ref_text_optional = True
    # Batch-1 decode on one CUDA stream. Left at 1 until measured: torch ops
    # issued from several Python threads still serialize on the default stream
    # and contend for the GIL, so concurrency here is a measurement, not a
    # given. Raise via `tts_concurrency` in patmir.toml once benchmarked.
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
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        import torch  # lazy (heavy)
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, Qwen2TokenizerFast

        self.repo_id = repo_id
        self.device = _resolve_device(device)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k) if top_k else None
        self.base_seed = int(seed)
        self.max_retries = int(max_retries)

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
        # AutoTokenizer on the repo id — the same trap the MLX candidate
        # documents: AutoTokenizer resolves the repo's custom *config* class
        # first, which prompts for trust_remote_code on a revision change and,
        # with funasr/modelscope imported (the emotion stage), lets modelscope's
        # hf_util patcher redirect the dynamic-module fetch to the slow mirror
        # (ADR 2026-05-26-prefer-fast-model-mirrors). The tokenizer itself is a
        # plain Qwen2 tokenizer with no auto_map.
        self.tok = Qwen2TokenizerFast.from_pretrained(snap)

        # The model class *does* need trust_remote_code — it is the repo's own
        # `modeling_higgs_multimodal_qwen3`. Loading from the local snapshot
        # path keeps that resolution off the network.
        model = AutoModelForCausalLM.from_pretrained(
            snap,
            dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        self.model = model.to(self.device).eval()
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation

        # Reference clip -> raw [T, N] codes. Upstream's `generate_speech`
        # re-encodes the reference on every call; with one pinned ref per
        # speaker that is the same clip re-encoded once per sentence.
        self._ref_cache: dict[str, Any] = {}
        # Codec frames per output sample, calibrated off the first reference
        # encode (see `_looks_overflowed`).
        self._samples_per_frame: float | None = None
        self._summary: dict[str, Any] = {
            "engine": f"higgs_v3_torch_{dtype}",
            "device": self.device,
            "sentences": 0,
            "overflows": 0,
        }

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
        import torchaudio  # lazy

        codec = self.model.get_audio_codec()
        wav, sr = torchaudio.load(path)
        wav = wav.float()
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        # The codec needs at least a second of audio.
        if wav.shape[-1] < SR:
            wav = torch.nn.functional.pad(wav, (0, SR - wav.shape[-1]))
        n_samples = int(wav.shape[-1])
        while wav.ndim < 3:
            wav = wav.unsqueeze(0)
        with torch.no_grad():
            codes_BNT = codec.encode(wav.to(self.device, dtype=torch.float32)).audio_codes
        codes_TN = codes_BNT.squeeze(0).transpose(0, 1).to(torch.long).cpu()

        frames = int(codes_TN.shape[0])
        if frames > 0 and self._samples_per_frame is None:
            self._samples_per_frame = n_samples / frames
        self._ref_cache[path] = codes_TN
        return codes_TN

    # ---- non-termination guardrail ---- #
    def _looks_overflowed(self, n_samples: int, max_new: int) -> bool:
        """Did the decode run to the token cap instead of emitting a stop?

        Upstream's `generate_speech` returns only the waveform, so overflow has
        to be inferred from its length: a run that never terminates produces
        exactly `max_new` rows, i.e. `max_new - (N - 1)` frames of audio. The
        frames-to-samples ratio is calibrated from the first reference encode
        (exact for this codec, no magic constant) — with no reference clip in
        the job there is nothing to calibrate against, so detection is skipped
        and the cap alone bounds the damage.

        This is an inference, not a signal from the model. It is a safety net:
        at the default temperature 0.8 the measured overflow rate on the MLX
        candidate was 0 of 25 (Reacher), 0/37 (masha), 0/60 (Fallout).
        """
        spf = self._samples_per_frame
        if not spf or n_samples <= 0:
            return False
        expected_max = spf * max(1, max_new - (N_CB - 1))
        return n_samples >= 0.98 * expected_max

    # ---- public API (driven by SentenceTTS adapter) ---- #
    def synthesize(self, text: str, *, ref_audio: Any = None, ref_text: str | None = None,
                   emotion: str | None = None) -> bytes:
        import torch  # lazy

        tag = EMOTION_TAGS.get(emotion) if emotion else None
        prompt_text = f"<|emotion:{tag}|>{text}" if tag else text
        ref_codes = self._encode_ref(str(ref_audio)) if ref_audio else None
        # Length-proportional cap so a non-terminating run is caught in ~1 s.
        # Legit lines use ~2.3 steps/char; cap at 4x/char, floor 200, ceil 1024.
        # Same formula as the MLX candidate.
        max_new = min(1024, max(200, int(len(prompt_text) * 4)))

        audio: np.ndarray | None = None
        for attempt in range(self.max_retries + 1):
            torch.manual_seed(self.base_seed + attempt)
            wav = self.model.generate_speech(
                prompt_text,
                self.tok,
                reference_codes=ref_codes,
                reference_text=ref_text or None,
                max_new_tokens=max_new,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            samples = np.asarray(wav.detach().cpu().float().numpy(), dtype=np.float32).squeeze()
            n = int(samples.size)
            if self._looks_overflowed(n, max_new):
                self._summary["overflows"] += 1
                continue
            if n > 0:
                audio = samples
                break

        self._summary["sentences"] += 1
        if audio is None:
            # Short silence so the pipeline keeps going rather than aborting the
            # job; the mux places a gap. NOTE a deliberate divergence from the
            # MLX candidate: when the FINAL attempt overflows, that one keeps
            # and decodes the overflowed codes, while this returns silence. An
            # overflowed run is a long garbage line, and under
            # `voice_no_overlap` the mux would ripple every later line behind
            # it — one gap is less damage than a cascade.
            audio = np.zeros(int(SR * 0.2), dtype=np.float32)
        return _wav_bytes(audio, SR)

    def summarize(self) -> dict[str, Any]:
        return dict(self._summary)

    def reset_summary(self) -> None:
        """Zero the per-job counters. Called by the TTS adapter at the start of
        each job so a warm-cached instance (watch-along stage_cache) reports
        per-section numbers instead of accumulating across sections."""
        self._summary.update({"sentences": 0, "overflows": 0})


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
    )
