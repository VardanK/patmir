"""Language-tag normalization shared by the Qwen3-ASR candidates.

Qwen3-ASR reports the language as a full English name (`language English`,
`language Russian`); downstream stages want ISO 639-1. Both the torch and the
MLX candidate used to carry their own copy of this function, and both copies
had the same hole: anything they did not recognise was returned VERBATIM
(`key.split()[0]`), so a truncated probe such as `"Engl"` became the language
`"engl"`, was forced onto every chunk, and the "no tag → run the full probe"
fallback never fired because the value was not None. One copy here, and it
validates: a tag is either a known name, an ISO code, or nothing.
"""
from __future__ import annotations

# Full names Qwen3-ASR emits, per its model card's language list, plus the
# aliases that show up in practice. Lower-cased keys.
FULL_TO_ISO: dict[str, str] = {
    "english": "en",
    "russian": "ru",
    "armenian": "hy",
    "chinese": "zh",
    "mandarin": "zh",
    "chinese mandarin": "zh",
    "cantonese": "yue",
    "arabic": "ar",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "portuguese": "pt",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "turkish": "tr",
    "ukrainian": "uk",
    "polish": "pl",
    "dutch": "nl",
    "hindi": "hi",
    "indonesian": "id",
    "vietnamese": "vi",
    "thai": "th",
    "malay": "ms",
    "swedish": "sv",
    "danish": "da",
    "finnish": "fi",
    "norwegian": "no",
    "czech": "cs",
    "greek": "el",
    "hungarian": "hu",
    "romanian": "ro",
    "bulgarian": "bg",
    "persian": "fa",
    "farsi": "fa",
    "hebrew": "he",
    "filipino": "tl",
    "tagalog": "tl",
    "macedonian": "mk",
    "croatian": "hr",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "lithuanian": "lt",
    "latvian": "lv",
    "estonian": "et",
    "georgian": "ka",
    "azerbaijani": "az",
    "kazakh": "kk",
    "uzbek": "uz",
    "bengali": "bn",
    "urdu": "ur",
    "tamil": "ta",
    "telugu": "te",
    "swahili": "sw",
    "catalan": "ca",
    "galician": "gl",
    "basque": "eu",
    "welsh": "cy",
    "irish": "ga",
    "icelandic": "is",
    "afrikaans": "af",
    "belarusian": "be",
    "bosnian": "bs",
    "albanian": "sq",
    "mongolian": "mn",
    "nepali": "ne",
    "sinhala": "si",
    "khmer": "km",
    "lao": "lo",
    "burmese": "my",
    "amharic": "am",
}


def to_iso(language: str | None) -> str | None:
    """ISO 639-1 code for a Qwen3-ASR language tag, or None when the tag is
    not one we recognise.

    None is the load-bearing return: the callers treat it as "no usable tag"
    and pay for a full-length probe rather than guessing. Returning the
    unknown string instead — which is what this did before — made a
    truncated 24-token probe output look like a detected language.
    """
    if not language:
        return None
    key = " ".join(language.strip().lower().split())
    if not key:
        return None
    if key in FULL_TO_ISO:
        return FULL_TO_ISO[key]
    if len(key) == 2 and key.isalpha() and key.isascii():
        return key
    # A tag with trailing text ("English asr", "Russian.") still has a name
    # in front; a name fragment ("Engl") does not.
    head = key.split()[0].strip(".,:;")
    if head in FULL_TO_ISO:
        return FULL_TO_ISO[head]
    return None
