"""Translator via TranslateGemma on MLX (Apple Silicon).

Google's TranslateGemma family — Gemma-3 base, fine-tuned for 55+ language
pairs. We use the 27B 4-bit MLX build by default; 12B / 4B variants exist
and are config-swappable.

The model has **no system-prompt surface and no register/style control**
(unlike Gemini Flash via OpenRouter). It's a pure translator: source lang +
target lang + text → target. The casual-Yerevan register prompt and per-
sentence emotion hint we lean on for Gemini are simply not honored here.
Output will tend to be more literal / register-neutral.

mlx-lm doesn't (as of 0.31.x) apply TranslateGemma's structured chat
template — see ml-explore/mlx-lm#783. We bypass the template and emit the
exact prompt string the model was trained on, mirroring the Jinja in
`mlx-community/translategemma-*-it-*/chat_template.jinja`:

    <bos><start_of_turn>user
    You are a professional <SrcLang> (<src>) to <TgtLang> (<tgt>) translator.
    Your goal is to accurately convey the meaning and nuances of the original
    <SrcLang> text while adhering to <TgtLang> grammar, vocabulary, and
    cultural sensitivities.
    Produce only the <TgtLang> translation, without any additional
    explanations or commentary. Please translate the following <SrcLang>
    text into <TgtLang>:


    <source text>
    <end_of_turn>
    <start_of_turn>model

Language codes are ISO 639-1 alpha-2 (`en`, `ru`, `hy`). Confirmed `"hy":
"Armenian"` in the model's language map. Whether `hy` is in the 55
production-tier or the experimental tail is unknown without the technical
report — only the ear-test answers that.

Same `translate(text, source_lang, emotion=None) -> str` contract as the
other candidates; `emotion` is accepted for adapter uniformity and silently
ignored (the model has no surface for it).
"""
from __future__ import annotations

from typing import Any

# Language-name table — minimal, covers the codes the pipeline actually
# emits today. Mirrors the keys present in the TranslateGemma Jinja
# template. The full template lists 160+ codes; we don't need them all
# at runtime since we only ever ask for `<source> → hy`.
_LANGUAGES: dict[str, str] = {
    "en": "English",
    "ru": "Russian",
    "hy": "Armenian",
    "hy-AM": "Armenian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "pt": "Portuguese",
    "tr": "Turkish",
    "ar": "Arabic",
}

# Default repo. Override via `model_id=` in the toml candidate entry.
_DEFAULT_REPO = "mlx-community/translategemma-27b-it-4bit"


def _normalize_lang(code: str) -> str:
    """ISO 639-1 with optional region subtag. Lowercase + hyphen-normalize."""
    if not code:
        return "en"
    norm = code.replace("_", "-")
    # Drop subtag if the table only has the bare language form.
    if norm not in _LANGUAGES and "-" in norm:
        bare = norm.split("-", 1)[0]
        if bare in _LANGUAGES:
            return bare
    return norm


def _build_prompt(src_code: str, tgt_code: str, text: str) -> str:
    """Reproduce the chat_template.jinja output for a single text translation."""
    src_lang = _LANGUAGES.get(src_code, src_code)
    tgt_lang = _LANGUAGES.get(tgt_code, tgt_code)
    return (
        f"<bos><start_of_turn>user\n"
        f"You are a professional {src_lang} ({src_code}) to "
        f"{tgt_lang} ({tgt_code}) translator. Your goal is to accurately "
        f"convey the meaning and nuances of the original {src_lang} text "
        f"while adhering to {tgt_lang} grammar, vocabulary, and cultural "
        f"sensitivities.\n"
        f"Produce only the {tgt_lang} translation, without any additional "
        f"explanations or commentary. Please translate the following "
        f"{src_lang} text into {tgt_lang}:\n\n\n"
        f"{text.strip()}"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


class TranslateGemmaMLX:
    id = "translategemma_mlx"

    def __init__(
        self,
        model_id: str = _DEFAULT_REPO,
        target_lang: str = "hy",
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        import mlx_lm  # lazy — heavy import
        self._mlx_lm = mlx_lm
        self.model_id = model_id
        self.target_lang = target_lang
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        # Lazy load on first call so building the candidate is cheap (the
        # 27B 4-bit weight set is ~15 GB; we don't want to pay that until
        # we actually translate).
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._model, self._tokenizer = self._mlx_lm.load(self.model_id)

    def translate(self, text: str, source_lang: str, emotion: str | None = None) -> str:
        del emotion  # TranslateGemma has no register/emotion surface.
        if not text.strip():
            return ""
        self._ensure_loaded()
        src = _normalize_lang(source_lang)
        tgt = _normalize_lang(self.target_lang)
        prompt = _build_prompt(src, tgt, text)
        # mlx_lm.generate signature is stable across 0.20-0.31; pass kwargs
        # the version installed actually accepts. Temperature/top_p land via
        # `sampler` on recent mlx_lm; on older versions they're top-level.
        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
            out = self._mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
                verbose=False,
            )
        except ImportError:
            # Older mlx-lm: positional sampler args.
            out = self._mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temp=self.temperature,
                top_p=self.top_p,
                verbose=False,
            )
        return _strip_stop_tokens(out).strip()


def _strip_stop_tokens(text: str) -> str:
    """Remove TranslateGemma's turn-end tokens if they leak into the output."""
    for marker in ("<end_of_turn>", "<eos>", "<start_of_turn>"):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
    return text


def build(config: dict[str, Any]) -> TranslateGemmaMLX:
    return TranslateGemmaMLX(
        model_id=config.get("model_id", _DEFAULT_REPO),
        target_lang=config.get("target_lang", "hy"),
        max_tokens=int(config.get("max_tokens", 512)),
        temperature=float(config.get("temperature", 0.0)),
        top_p=float(config.get("top_p", 1.0)),
    )
