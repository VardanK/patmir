"""Higgs Audio v3 TTS (4B) via a native MLX port of the AR backbone.

Why MLX: the decode is memory-bandwidth-bound on the 4B weights (~69 ms/step
in torch-MPS bf16). MLX does fast *quantized* matmul on Apple Silicon, which
torch-MPS can't; q8 cuts that to ~28 ms/step. We port only the autoregressive
graph — the Qwen3-4B backbone (via `mlx_lm`) plus Higgs' fused multi-codebook
embedding/head and delay-pattern sampler — and run it quantized. The 0.8 GB
audio codec (`bosonai/higgs-audio-v2-tokenizer`, ref-encode + waveform-decode)
stays in PyTorch on MPS; it runs once per utterance, not per step.

Voice cloning is zero-shot from a reference clip (+ optional transcript, which
improves fidelity but isn't required → `ref_text_optional = True`). Emotion is
driven by Higgs' inline control tags (`<|emotion:anger|>` at the sentence
start), mapped from the pipeline's 5 emotion bins — this is real emotion
control, unlike OmniVoice's guidance_scale proxy.

Non-termination guardrail: the AR decoder occasionally fails to emit its stop
token and runs to the token cap (a known Higgs failure mode; empirically tied
to LOW temperature / repetitive refs, not high temperature). We cap
`max_new_tokens` proportional to text length so an overflow is caught in ~1 s,
and reseed-and-retry on overflow. Default temperature is 0.8 (the stable zone
measured across short/long lines; lower temps caused *more* overflows).

ADR: docs/architecture/decisions/2026-06-20-higgs-v3-tts-mlx-q8.md
"""
from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_DEFAULT = "multimodalart/higgs-audio-v3-tts-4b-transformers"
CODEC_DEFAULT = "bosonai/higgs-audio-v2-tokenizer"

# Higgs codec/delay constants (fixed by the checkpoint).
BOC_ID, EOC_ID, AUDIO_PH = 1024, 1025, -100
N_CB, V_CB, SR = 8, 1026, 24000
_REQUIRED_SPECIALS = ("<|tts|>", "<|ref_audio|>", "<|text|>", "<|audio|>")

# Pipeline emotion bins -> Higgs inline emotion tags (sentence-level). "neutral"
# emits no tag. Tunable by ear — Higgs exposes 21 emotions; these are the
# closest fit for cartoon/drama dialogue.
EMOTION_TAGS = {
    "happy": "elation",
    "sad": "sadness",
    "angry": "anger",
    "fearful": "fear",
    "neutral": None,
}


class HiggsV3MLX:
    id = "higgs_v3_mlx"
    # IVC-style: clones from audio alone; ref_text only improves fidelity.
    ref_text_optional = True
    # Single MLX GPU stream (like OmniVoice) — no concurrency benefit.
    tts_concurrency = 1

    def __init__(
        self,
        repo_id: str = REPO_DEFAULT,
        codec_id: str = CODEC_DEFAULT,
        quant: str = "q8",
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 0,
        max_retries: int = 2,
        codec_device: str = "mps",
        group_size: int = 64,
    ) -> None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        import mlx.core as mx  # lazy
        import mlx.nn as nn
        from mlx_lm.models import qwen3
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import AutoTokenizer
        import json

        self._mx = mx
        self.repo_id = repo_id
        self.codec_id = codec_id
        self.quant = quant
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.base_seed = int(seed)
        self.max_retries = int(max_retries)
        self.codec_device = codec_device

        cfg = json.load(open(hf_hub_download(repo_id, "config.json")))
        tc = cfg["text_config"]
        args = qwen3.ModelArgs(
            model_type="qwen3", hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"], intermediate_size=tc["intermediate_size"],
            num_attention_heads=tc["num_attention_heads"], rms_norm_eps=tc["rms_norm_eps"],
            vocab_size=tc["vocab_size"], num_key_value_heads=tc["num_key_value_heads"],
            max_position_embeddings=tc["max_position_embeddings"],
            rope_theta=tc["rope_parameters"]["rope_theta"], head_dim=tc["head_dim"],
            tie_word_embeddings=True,
        )
        model = qwen3.Qwen3Model(args)
        snap = snapshot_download(repo_id, allow_patterns=["*.safetensors", "config.json"])
        all_w = mx.load(os.path.join(snap, "model.safetensors"))
        qw, w_audio = self._remap(all_w)
        model.load_weights(list(qw.items()), strict=True)
        model.eval()
        if quant in ("q8", "q4"):
            nn.quantize(model, group_size=group_size, bits=8 if quant == "q8" else 4)
        mx.eval(model.parameters())
        self.model = model
        self.w_audio = w_audio.astype(mx.bfloat16)

        self.tok = AutoTokenizer.from_pretrained(repo_id)
        self._codec = None
        self._ref_cache: dict[str, Any] = {}
        self._summary: dict[str, Any] = {"engine": f"higgs_v3_{quant}", "sentences": 0, "overflows": 0}

    # ---- weight remap (Higgs ckpt -> mlx_lm Qwen3Model) ---- #
    @staticmethod
    def _remap(weights):
        qw, w_audio = {}, None
        for k, v in weights.items():
            if k == "tied.embedding.modality_embeddings.0.embedding.weight":
                w_audio = v
            elif k == "tied.embedding.text_embedding.weight":
                qw["embed_tokens.weight"] = v
            elif k.startswith("body.layers."):
                qw[k[len("body."):]] = v
            elif k == "body.norm.weight":
                qw["norm.weight"] = v
            # skip codec (tied.embedding.modality_embeddings.0.model.*) + tied.head.*
        return qw, w_audio

    # ---- fused multi-codebook embed / head ---- #
    def _fused_embed(self, codes_LN):
        mx = self._mx
        offsets = mx.arange(N_CB) * V_CB
        return self.w_audio[codes_LN + offsets].sum(axis=-2)

    def _audio_logits(self, hidden_D):
        mx = self._mx
        return (hidden_D @ self.w_audio.T).reshape(N_CB, V_CB).astype(mx.float32)

    # ---- sampler (delay pattern + EOC state machine) ---- #
    def _sample(self, logits_NV):
        mx = self._mx
        t = self.temperature
        if t <= 1e-5:
            return mx.argmax(logits_NV, axis=-1)
        logits = logits_NV / t
        tp = self.top_p
        if tp is not None and tp < 1.0:
            order = mx.argsort(-logits, axis=-1)
            s = mx.take_along_axis(logits, order, axis=-1)
            probs = mx.softmax(s, axis=-1)
            cum = mx.cumsum(probs, axis=-1)
            keep = cum - probs < tp
            s = mx.where(keep, s, mx.full(s.shape, -1e9))
            logits = mx.take_along_axis(s, mx.argsort(order, axis=-1), axis=-1)
        return mx.random.categorical(logits, axis=-1)

    # ---- codec (torch, on MPS, lazy) ---- #
    def _get_codec(self):
        if self._codec is None:
            import torch
            from transformers import AutoModel
            c = AutoModel.from_pretrained(self.codec_id, dtype=torch.float32, trust_remote_code=True)
            self._codec = c.to(self.codec_device).eval()
        return self._codec

    def _encode_ref(self, path: str):
        if path in self._ref_cache:
            return self._ref_cache[path]
        import torch
        import torchaudio
        codec = self._get_codec()
        wav, sr = torchaudio.load(path)
        wav = wav.float()
        while wav.ndim < 3:
            wav = wav.unsqueeze(0)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.shape[-1] < SR:
            wav = torch.nn.functional.pad(wav, (0, SR - wav.shape[-1]))
        codes_BNT = codec.encode(wav.to(self.codec_device)).audio_codes
        codes_TN = codes_BNT.squeeze(0).transpose(0, 1).to(torch.long).cpu()
        out = self._mx.array(self._torch_apply_delay(codes_TN).numpy())
        self._ref_cache[path] = out
        return out

    @staticmethod
    def _torch_apply_delay(codes_TN):
        import torch
        T, N = codes_TN.shape
        out = torch.full((T + N - 1, N), EOC_ID, dtype=torch.long)
        tix = torch.arange(T + N - 1)
        for c in range(N):
            out[tix < c, c] = BOC_ID
            out[c:c + T, c] = codes_TN[:, c]
        return out

    def _decode_codes(self, codes_TN_mx):
        import torch
        codec = self._get_codec()
        codes_TN = torch.tensor(codes_TN_mx.tolist(), dtype=torch.long)
        cv = V_CB - 2
        codes_TN = torch.where(codes_TN >= cv, torch.zeros_like(codes_TN), codes_TN)
        codes_BNT = codes_TN.transpose(0, 1).unsqueeze(0).to(self.codec_device)
        audio = codec.decode(codes_BNT).audio_values
        return audio.squeeze(0).squeeze(0).detach().cpu().float().numpy()

    # ---- prompt assembly ---- #
    def _specials(self):
        vocab = dict(self.tok.get_added_vocab())
        ids = {t: vocab[t] for t in _REQUIRED_SPECIALS}
        ids["<|ref_text|>"] = vocab.get("<|ref_text|>")
        return ids

    def _build_prompt(self, text: str, num_ref: int, ref_text: str | None):
        sp = self._specials()
        ids = [sp["<|tts|>"]]
        if ref_text and num_ref > 0 and sp["<|ref_text|>"] is not None:
            ids.append(sp["<|ref_text|>"])
            ids += self.tok.encode(ref_text, add_special_tokens=False)
        if num_ref > 0:
            ids.append(sp["<|ref_audio|>"])
            ids += [AUDIO_PH] * num_ref
        ids.append(sp["<|text|>"])
        ids += self.tok.encode(text, add_special_tokens=False)
        ids.append(sp["<|audio|>"])
        return ids

    # ---- one generation pass; returns (codes_TN | None) ---- #
    def _generate_once(self, text, delayed_ref, ref_text, max_new, seed):
        mx = self._mx
        from mlx_lm.models.cache import make_prompt_cache
        mx.random.seed(seed)
        num_ref = 0 if delayed_ref is None else int(delayed_ref.shape[0])
        ids = self._build_prompt(text, num_ref, ref_text)
        ids_mx = mx.array(ids)
        mask_audio = ids_mx == AUDIO_PH
        safe = mx.where(mask_audio, mx.zeros_like(ids_mx), ids_mx)
        embeds = self.model.embed_tokens(safe)
        if delayed_ref is not None:
            aud = self._fused_embed(delayed_ref[:num_ref]).astype(embeds.dtype)
            pos = [i for i, t in enumerate(ids) if t == AUDIO_PH]
            embeds[mx.array(pos)] = aud
        embeds = embeds[None]

        cache = make_prompt_cache(self.model)
        h = self.model(mx.zeros((1, embeds.shape[1]), dtype=mx.int32), cache=cache, input_embeddings=embeds)
        h_last = h[0, -1]

        delay = 0
        eoc = None
        rows = []
        for _ in range(max_new):
            logits = self._audio_logits(h_last)
            codes = self._sample(logits).astype(mx.int32)
            done = False
            if delay < N_CB:
                nxt = delay + 1
                if nxt < N_CB:
                    codes[nxt:] = BOC_ID
                delay += 1
            elif eoc is not None:
                eoc -= 1
                if eoc <= 0:
                    done = True
            elif int(codes[0].item()) == EOC_ID:
                eoc = N_CB - 2
            if done:
                break
            rows.append(codes)
            step = self._fused_embed(codes).astype(embeds.dtype)[None, None]
            h = self.model(mx.zeros((1, 1), dtype=mx.int32), cache=cache, input_embeddings=step)
            h_last = h[0, -1]
            mx.eval(h_last, codes)

        overflowed = len(rows) >= max_new
        if len(rows) < N_CB:
            return None, overflowed
        delayed_LN = mx.stack(rows, axis=0)
        T = delayed_LN.shape[0] - (N_CB - 1)
        codes_TN = mx.stack([delayed_LN[c:c + T, c] for c in range(N_CB)], axis=1)
        return codes_TN, overflowed

    # ---- public API (driven by SentenceTTS adapter) ---- #
    def synthesize(self, text: str, *, ref_audio: Any = None, ref_text: str | None = None,
                   emotion: str | None = None) -> bytes:
        tag = EMOTION_TAGS.get(emotion) if emotion else None
        prompt_text = f"<|emotion:{tag}|>{text}" if tag else text
        delayed_ref = self._encode_ref(str(ref_audio)) if ref_audio else None
        # Length-proportional cap so a non-terminating run is caught fast
        # (legit lines use ~2.3 steps/char; cap at 4×/char, floor 200, ceil 1024).
        max_new = min(1024, max(200, int(len(prompt_text) * 4)))

        codes_TN = None
        for attempt in range(self.max_retries + 1):
            codes_TN, overflowed = self._generate_once(
                prompt_text, delayed_ref, ref_text, max_new, self.base_seed + attempt
            )
            if codes_TN is not None and not overflowed:
                break
            if overflowed:
                self._summary["overflows"] += 1
        self._summary["sentences"] += 1

        if codes_TN is None:
            # total failure → emit short silence so the pipeline keeps going
            audio = np.zeros(int(SR * 0.2), dtype=np.float32)
        else:
            audio = self._decode_codes(codes_TN)
        return _wav_bytes(audio, SR)

    def summarize(self) -> dict[str, Any]:
        return dict(self._summary)


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


def build(config: dict[str, Any]) -> HiggsV3MLX:
    return HiggsV3MLX(
        repo_id=config.get("repo_id", REPO_DEFAULT),
        codec_id=config.get("codec_id", CODEC_DEFAULT),
        quant=str(config.get("quant", "q8")),
        temperature=float(config.get("temperature", 0.8)),
        top_p=float(config.get("top_p", 0.95)),
        seed=int(config.get("seed", 0)),
        max_retries=int(config.get("max_retries", 2)),
        codec_device=str(config.get("codec_device", "mps")),
        group_size=int(config.get("group_size", 64)),
    )
