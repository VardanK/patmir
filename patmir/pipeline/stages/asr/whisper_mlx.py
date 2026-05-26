"""ASR stage — mlx-whisper (large-v3 family on Apple Metal).

Wraps `mlx_whisper.transcribe` which runs OpenAI Whisper natively on MLX.
Word-level timestamps come straight from Whisper's own alignment (no
separate forced-aligner needed). Sentence segmentation uses the model's
own segments (it splits on prosodic / punctuation cues during decoding),
with `_sentence.group_into_sentences` as a fallback when segments are
absent.

Multilingual: 99 languages including English and Russian. We pass through
`ctx.source_lang` as a hint; if unset, Whisper auto-detects.

Install once: `uv pip install mlx-whisper`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Sentence, Transcript, Word
from patmir.pipeline.stages.asr._sentence import is_non_lexical, write_transcript


class WhisperMLX:
    id = "whisper_mlx"

    def __init__(
        self,
        repo_id: str,
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
        no_speech_threshold: float = 0.6,
    ) -> None:
        import mlx_whisper  # lazy import
        self._mlx_whisper = mlx_whisper
        self.repo_id = repo_id
        self.temperature = temperature
        self.condition_on_previous_text = condition_on_previous_text
        self.no_speech_threshold = no_speech_threshold

    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript:
        ctx.progress("asr", 0.05, f"mlx-whisper ({self.repo_id})")
        result = self._mlx_whisper.transcribe(
            str(audio.path),
            path_or_hf_repo=self.repo_id,
            language=ctx.source_lang or None,
            word_timestamps=True,
            temperature=self.temperature,
            condition_on_previous_text=self.condition_on_previous_text,
            no_speech_threshold=self.no_speech_threshold,
            verbose=False,
        )
        language = result.get("language") or ctx.source_lang or "en"
        ctx.source_lang = language
        ctx.progress("asr", 0.5, f"detected language={language}")

        words: list[Word] = []
        sentences: list[Sentence] = []
        for seg in result.get("segments", []):
            seg_text = (seg.get("text") or "").strip()
            seg_words: list[Word] = []
            for w in seg.get("words", []) or []:
                tok = (w.get("word") or "").strip()
                if not tok:
                    continue
                wobj = Word(
                    text=tok,
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", w.get("start", 0.0))),
                )
                words.append(wobj)
                seg_words.append(wobj)
            if not seg_text:
                continue
            # Drop onomatopoeic repetition (laughter, "yum-yum", etc.) so
            # they don't translate to garbage or get TTS'd as artificial
            # speech. The original sound survives in the htdemucs vocals
            # stem that the mux mixes back outside ducked windows.
            if is_non_lexical(seg_text):
                continue
            if seg_words:
                s_start = min(w.start for w in seg_words)
                s_end = max(w.end for w in seg_words)
            else:
                s_start = float(seg.get("start", 0.0))
                s_end = float(seg.get("end", s_start))
            sentences.append(Sentence(
                idx=len(sentences),
                text=seg_text,
                start=s_start,
                end=s_end,
                words=seg_words,
            ))

        transcript = Transcript(language=language, words=words, sentences=sentences)
        write_transcript(ctx.job_dir, transcript)
        ctx.progress("asr", 1.0, "done")
        return transcript

    def transcribe_slice(self, audio_path: Path, language: str | None) -> list[Word]:
        """Re-transcribe a single short clip and return its words (no file I/O,
        no segmentation). Used by the runner's diarization-driven gap-filler
        to recover dialogue Whisper missed when running over the whole stem.
        """
        result = self._mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self.repo_id,
            language=language,
            word_timestamps=True,
            temperature=self.temperature,
            condition_on_previous_text=False,
            no_speech_threshold=self.no_speech_threshold,
            verbose=False,
        )
        words: list[Word] = []
        for seg in result.get("segments", []) or []:
            for w in seg.get("words", []) or []:
                tok = (w.get("word") or "").strip()
                if not tok:
                    continue
                words.append(Word(
                    text=tok,
                    start=float(w.get("start", w.get("end", 0.0))),
                    end=float(w.get("end", w.get("start", 0.0))),
                ))
        return words


def build(config: dict[str, Any]) -> WhisperMLX:
    return WhisperMLX(
        repo_id=config["repo_id"],
        temperature=float(config.get("temperature", 0.0)),
        condition_on_previous_text=bool(config.get("condition_on_previous_text", True)),
        no_speech_threshold=float(config.get("no_speech_threshold", 0.6)),
    )
