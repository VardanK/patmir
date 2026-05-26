"""Qwen3 translator via llama-cpp-python (GGUF on Metal).

Mirrors `qwen3_instruct_mlx.py` but runs a GGUF quantization through
llama.cpp's Metal backend instead of MLX. Lets us reach Qwen3-14B Q4_K_M
(~9 GB) on a 32 GB Mac with headroom for the rest of the pipeline.

The GGUF file embeds Qwen3's Jinja chat template; llama-cpp renders it on
each `create_chat_completion` call. To suppress thinking mode we append
`/no_think` to the user message (the soft directive Qwen3's chat template
honors) and strip any `<think>...</think>` envelope that might still leak.

Install once: `CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python`.
"""
from __future__ import annotations

from typing import Any


_LANG_NAME = {"en": "English", "ru": "Russian", "hy": "Armenian"}


_SYSTEM = (
    "You are a professional translator specializing in Eastern Armenian (hy_AM). "
    "You translate text from English or Russian into natural, idiomatic Eastern "
    "Armenian as spoken in Armenia. You preserve tone, register, and emotion. "
    "You never produce Western Armenian (hyw). You reply with ONLY the Armenian "
    "translation — no quotes, no explanations, no source-language echoing."
)


class Qwen3GGUFTranslator:
    id = "qwen3_gguf"

    def __init__(
        self,
        repo_id: str,
        filename: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        no_think: bool = True,
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
        self.repetition_penalty = repetition_penalty
        self.no_think = no_think

    def translate(self, text: str, source_lang: str) -> str:
        src_name = _LANG_NAME.get(source_lang, source_lang)
        user_msg = (
            f"Translate this {src_name} sentence into Eastern Armenian:\n\n"
            f"{text}"
        )
        if self.no_think:
            user_msg += " /no_think"

        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repetition_penalty,
        )
        content = out["choices"][0]["message"]["content"] or ""
        return _strip_thinking(content).strip()


def _strip_thinking(text: str) -> str:
    """Strip `<think>...</think>` envelopes that occasionally appear even
    with `/no_think`."""
    if "<think>" in text and "</think>" in text:
        end = text.index("</think>") + len("</think>")
        return text[end:].strip()
    return text


def build(config: dict[str, Any]) -> Qwen3GGUFTranslator:
    return Qwen3GGUFTranslator(
        repo_id=config["repo_id"],
        filename=config["filename"],
        max_tokens=int(config.get("max_tokens", 256)),
        temperature=float(config.get("temperature", 0.1)),
        top_p=float(config.get("top_p", 0.9)),
        repetition_penalty=float(config.get("repetition_penalty", 1.15)),
        n_ctx=int(config.get("n_ctx", 4096)),
        n_gpu_layers=int(config.get("n_gpu_layers", -1)),
        no_think=bool(config.get("no_think", True)),
    )
