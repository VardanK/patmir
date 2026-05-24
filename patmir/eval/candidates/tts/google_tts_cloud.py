"""Google Cloud TTS — Armenian voice (`hy-AM-Standard-A`).

Used as a cloud ceiling reference. Requires the environment variable
`GOOGLE_APPLICATION_CREDENTIALS` to point at a service-account JSON.
"""
from __future__ import annotations

import os
from typing import Any


class GoogleTTS:
    id = "google_tts_cloud"

    def __init__(self, voice: str) -> None:
        from google.cloud import texttospeech  # lazy import

        self._tts = texttospeech
        self.client = texttospeech.TextToSpeechClient()
        # Voice name like "hy-AM-Standard-A".
        self.voice_params = texttospeech.VoiceSelectionParams(
            language_code="hy-AM",
            name=voice,
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=22050,
        )

    def synthesize(self, text: str) -> bytes:
        synthesis_input = self._tts.SynthesisInput(text=text)
        resp = self.client.synthesize_speech(
            input=synthesis_input,
            voice=self.voice_params,
            audio_config=self.audio_config,
        )
        return resp.audio_content  # already a WAV (LINEAR16)


def build(config: dict[str, Any]) -> GoogleTTS:
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set; cannot run google_tts_cloud."
        )
    return GoogleTTS(voice=config["voice"])
