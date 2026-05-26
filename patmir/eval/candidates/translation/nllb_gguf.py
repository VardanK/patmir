"""NLLB-200 translator via llama-cpp-python (GGUF on Metal).

NLLB is a Seq2Seq encoder-decoder model. llama.cpp added encoder-decoder
support in late 2024 (initially for T5; NLLB followed). The model's GGUF
ships the FLORES-200 language tokens (`eng_Latn`, `hye_Armn`, etc.) as
part of its tokenizer. We:

  1. Build a prompt that prefixes the source-language tag to the input.
  2. Let llama.cpp run the encoder over the prompt and start the decoder.
  3. Pre-prime the decoder by adding the target-language tag at the end of
     the prompt so the first generated token continues in `hye_Armn`.

This is the de-facto format used by community NLLB GGUF ports
(`acceldium/nllb-200-3.3B-GGUF` has no usage README, so format is
inferred from M2M-100/NLLB tokenizer conventions).

Install once: `CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python`.
"""
from __future__ import annotations

from typing import Any


# NLLB uses FLORES-200 BCP-47-style codes.
NLLB_LANG = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "hy": "hye_Armn",
}


class NLLBGGUFTranslator:
    id = "nllb_gguf"

    def __init__(
        self,
        repo_id: str,
        filename: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        n_ctx: int = 1024,
        n_gpu_layers: int = -1,
    ) -> None:
        from llama_cpp import Llama  # lazy

        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.target_lang = NLLB_LANG["hy"]

    def translate(self, text: str, source_lang: str) -> str:
        src = NLLB_LANG.get(source_lang, source_lang)
        # NLLB convention: source tag + text + target tag primes the decoder.
        prompt = f"{src} {text} {self.target_lang} "
        out = self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["</s>", f"{self.target_lang} "],
        )
        text_out = out["choices"][0]["text"] or ""
        # Strip any trailing language tag echo.
        for tag in NLLB_LANG.values():
            text_out = text_out.replace(tag, "")
        return text_out.strip()


def build(config: dict[str, Any]) -> NLLBGGUFTranslator:
    return NLLBGGUFTranslator(
        repo_id=config["repo_id"],
        filename=config["filename"],
        max_tokens=int(config.get("max_tokens", 512)),
        temperature=float(config.get("temperature", 0.0)),
        top_p=float(config.get("top_p", 1.0)),
        n_ctx=int(config.get("n_ctx", 1024)),
        n_gpu_layers=int(config.get("n_gpu_layers", -1)),
    )
