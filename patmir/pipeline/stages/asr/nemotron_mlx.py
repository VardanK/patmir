"""ASR stage — NVIDIA Nemotron 3.5 streaming ASR (FastConformer-RNNT) on MLX.

Runs through `mlx-audio` (`mlx_audio.stt.load`). `model.generate(path)` returns
an `AlignedResult` with `.text` and `.sentences`; each `AlignedSentence` carries
`.start/.end` and a list of `AlignedToken` (`.text/.start/.end/.duration`).
Tokens are sentencepiece-style subwords with a leading-space word-boundary
convention (e.g. `' '`,`'R'`,`'ight'`,`' '`,`'f'`,`'u'`,`'ck'`,`' up'`), so we
merge them back into words on leading whitespace.

Language: the result does NOT report the detected language, so when the source
language is unknown we auto-detect (omit the kwarg) and infer ISO 639-1 from the
script of the transcript (Cyrillic→ru, Armenian→hy, CJK→zh, else en) for
translation routing. A known source can be pinned via the `language` config key
(a `prompt_dictionary` locale key like `en-US` / `ru`); an invalid key falls
back to auto-detect rather than erroring. Forcing the *wrong* locale produces
phonetic gibberish, so auto-detect is the safe default.

Install once: `uv pip install "git+https://github.com/Blaizzy/mlx-audio.git"`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Sentence, Transcript, Word
from patmir.pipeline.stages.asr._sentence import (
    filter_words_by_vad,
    is_non_lexical,
    write_transcript,
)


def _detect_lang(text: str) -> str:
    """Infer ISO 639-1 from the dominant script of the transcript. Patmir's
    sources are overwhelmingly English or Russian; this only needs to route the
    translator, not be a full LID. Returns 'en' when no strong signal."""
    cyr = arm = cjk = lat = 0
    for ch in text:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            cyr += 1
        elif 0x0530 <= o <= 0x058F:
            arm += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF:
            cjk += 1
        elif (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A):
            lat += 1
    counts = {"ru": cyr, "hy": arm, "zh": cjk, "en": lat}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "en"


def _tokens_to_words(tokens: list[Any]) -> list[Word]:
    """Merge subword `AlignedToken`s into whitespace-delimited words, carrying
    timing from the first/last subword of each word."""
    words: list[Word] = []
    buf = ""
    start: float | None = None
    end: float | None = None

    def flush() -> None:
        nonlocal buf, start, end
        if buf.strip() and start is not None and end is not None:
            words.append(Word(text=buf.strip(), start=start, end=end))
        buf = ""
        start = None
        end = None

    for tok in tokens or []:
        t = getattr(tok, "text", "") or ""
        if t.strip() == "":
            flush()  # pure-whitespace token is a hard word boundary
            continue
        if (t[:1].isspace() or t.startswith("▁")) and buf.strip():
            flush()
        if start is None:
            start = float(getattr(tok, "start", 0.0))
        buf += t
        end = float(getattr(tok, "end", getattr(tok, "start", 0.0)))
    flush()
    return words


def _segment_words(
    words: list[Word],
    *,
    max_gap_s: float,
    max_seconds: float,
    max_words: int,
) -> list[Sentence]:
    """Group timed words into sentences. Splits on terminal punctuation, a word
    gap > `max_gap_s`, OR hard caps on duration / word count.

    The hard caps are load-bearing for Nemotron: the model emits **no sentence
    punctuation**, so continuous narration would otherwise collapse into one
    giant run-on per audio chunk — which the translator then drops (returns
    empty) and the TTS can't place. The caps keep chunks TTS-sized even with
    zero punctuation and few pauses.
    """
    sentences: list[Sentence] = []
    bucket: list[Word] = []

    def flush() -> None:
        if not bucket:
            return
        text = " ".join(w.text for w in bucket).strip()
        if text:
            sentences.append(Sentence(
                idx=len(sentences),
                text=text,
                start=bucket[0].start,
                end=bucket[-1].end,
                words=list(bucket),
            ))
        bucket.clear()

    for i, w in enumerate(words):
        bucket.append(w)
        ends_punct = w.text.endswith((".", "!", "?", "。", "！", "？", "…"))
        gap_next = words[i + 1].start - w.end if i + 1 < len(words) else 0.0
        dur = bucket[-1].end - bucket[0].start
        if ends_punct or gap_next > max_gap_s or dur >= max_seconds or len(bucket) >= max_words:
            flush()
    flush()
    return sentences


class NemotronMLX:
    id = "nemotron_mlx"

    def __init__(
        self,
        repo_id: str,
        language: str | None = None,
        vad_filter: bool = True,
        max_gap_s: float = 0.6,
        max_sentence_seconds: float = 10.0,
        max_sentence_words: int = 40,
    ) -> None:
        from mlx_audio.stt import load  # lazy import (heavy)
        self._model = load(repo_id)
        self.repo_id = repo_id
        self.language = language or None
        self.vad_filter = vad_filter
        self.max_gap_s = max_gap_s
        self.max_sentence_seconds = max_sentence_seconds
        self.max_sentence_words = max_sentence_words

    def _valid_lang(self, lang: str | None) -> str | None:
        """Return `lang` only if the model accepts it as a prompt key; else None
        (→ auto-detect). Guards both the configured override and gap-filler
        re-runs that pass an inferred ISO code like 'ru'."""
        if not lang:
            return None
        pd = getattr(self._model, "prompt_dictionary", None)
        if pd is not None and hasattr(pd, "__contains__") and lang not in pd:
            return None
        return lang

    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript:
        ctx.progress("asr", 0.05, f"nemotron-mlx ({self.repo_id})")
        forced = self._valid_lang(self.language or ctx.source_lang)
        result = (
            self._model.generate(str(audio.path), language=forced)
            if forced
            else self._model.generate(str(audio.path))
        )

        vad_intervals = ctx.extra.get("vad_intervals")
        # Collect all timed words across Nemotron's sentences, then RE-SEGMENT
        # ourselves — the model emits no punctuation, so its own sentence
        # boundaries collapse continuous speech into one giant run-on. Build
        # words from the subword tokens; fall back to a single coarse word per
        # AlignedSentence only if a sentence carries text but no tokens.
        all_words: list[Word] = []
        for asent in getattr(result, "sentences", []) or []:
            seg_words = _tokens_to_words(getattr(asent, "tokens", []) or [])
            if not seg_words:
                seg_text = (getattr(asent, "text", "") or "").strip()
                if seg_text:
                    seg_words = [Word(
                        text=seg_text,
                        start=float(getattr(asent, "start", 0.0)),
                        end=float(getattr(asent, "end", getattr(asent, "start", 0.0))),
                    )]
            all_words.extend(seg_words)
        if self.vad_filter:
            all_words = filter_words_by_vad(all_words, vad_intervals)

        # Drop onomatopoeia/laughter sentences; renumber the survivors.
        sentences: list[Sentence] = []
        words: list[Word] = []
        for s in _segment_words(
            all_words,
            max_gap_s=self.max_gap_s,
            max_seconds=self.max_sentence_seconds,
            max_words=self.max_sentence_words,
        ):
            if is_non_lexical(s.text):
                continue
            s.idx = len(sentences)
            sentences.append(s)
            words.extend(s.words)

        full_text = (getattr(result, "text", "") or "").strip() or " ".join(s.text for s in sentences)
        language = _detect_lang(full_text)
        ctx.source_lang = language
        ctx.progress("asr", 0.9, f"detected language={language}, {len(sentences)} sentences")

        transcript = Transcript(language=language, words=words, sentences=sentences)
        write_transcript(ctx.job_dir, transcript)
        ctx.progress("asr", 1.0, "done")
        return transcript

    def transcribe_slice(self, audio_path: Path, language: str | None) -> list[Word]:
        """Re-transcribe a single short clip → words (no file I/O, no
        segmentation). Used by the runner's diarization-driven gap-filler."""
        forced = self._valid_lang(language)
        result = (
            self._model.generate(str(audio_path), language=forced)
            if forced
            else self._model.generate(str(audio_path))
        )
        words: list[Word] = []
        for asent in getattr(result, "sentences", []) or []:
            words.extend(_tokens_to_words(getattr(asent, "tokens", []) or []))
        return words


def build(config: dict[str, Any]) -> NemotronMLX:
    lang = config.get("language")
    if isinstance(lang, str) and lang.strip().lower() in ("", "none", "auto"):
        lang = None
    return NemotronMLX(
        repo_id=config["repo_id"],
        language=lang,
        vad_filter=bool(config.get("vad_filter", True)),
        max_gap_s=float(config.get("max_gap_s", 0.6)),
        max_sentence_seconds=float(config.get("max_sentence_seconds", 10.0)),
        max_sentence_words=int(config.get("max_sentence_words", 40)),
    )
