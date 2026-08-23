"""Tests for reference-clip decoding in the Higgs TTS candidates.

These exercise the real ffmpeg binary, which is a hard dependency of the
project (ingest, mux and output all shell out to it), so there is nothing to
mock. The point of this module is that the Higgs candidates stopped using
`torchaudio.load()` — on torch >= 2.11 that delegates to torchcodec, whose
native libraries fail to dlopen on the WSL2 box and killed the TTS stage.
"""
from __future__ import annotations

import math
import shutil
import struct
import wave

import numpy as np
import pytest

from patmir.eval.candidates.tts._audio_io import decode_mono

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _write_wav(path, *, seconds=1.0, rate=16000, channels=1, freq=440.0):
    frames = bytearray()
    for i in range(int(rate * seconds)):
        value = int(20000 * math.sin(2 * math.pi * freq * i / rate))
        frames += struct.pack("<" + "h" * channels, *([value] * channels))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return path


def test_resamples_to_the_requested_rate(tmp_path):
    src = _write_wav(tmp_path / "ref.wav", seconds=1.0, rate=16000)
    out = decode_mono(str(src), 24000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    # ffmpeg may differ by a frame or two at the boundary.
    assert abs(out.shape[0] - 24000) < 100


def test_downmixes_stereo_to_mono(tmp_path):
    src = _write_wav(tmp_path / "stereo.wav", seconds=0.5, rate=24000, channels=2)
    out = decode_mono(str(src), 24000)
    assert out.ndim == 1
    assert abs(out.shape[0] - 12000) < 100


def test_no_resample_needed_is_sample_accurate(tmp_path):
    src = _write_wav(tmp_path / "same.wav", seconds=0.25, rate=24000)
    out = decode_mono(str(src), 24000)
    assert abs(out.shape[0] - 6000) < 10


def test_output_is_writable_and_normalized(tmp_path):
    """torch.from_numpy warns and may alias on a read-only buffer, which is
    what `np.frombuffer` hands back over subprocess bytes."""
    src = _write_wav(tmp_path / "ref.wav", seconds=0.3)
    out = decode_mono(str(src), 24000)
    assert out.flags.writeable
    peak = float(np.abs(out).max())
    assert 0.1 < peak <= 1.0, peak


def test_result_survives_torch_from_numpy(tmp_path):
    torch = pytest.importorskip("torch")
    src = _write_wav(tmp_path / "ref.wav", seconds=0.3)
    wav = torch.from_numpy(decode_mono(str(src), 24000))
    while wav.ndim < 3:
        wav = wav.unsqueeze(0)
    # What the codec is handed: [batch, channels, samples].
    assert wav.ndim == 3
    assert wav.shape[0] == 1 and wav.shape[1] == 1


def test_missing_file_raises_with_ffmpeg_stderr(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        decode_mono(str(tmp_path / "nope.wav"), 24000)
    message = str(excinfo.value)
    assert "nope.wav" in message
    # The real ffmpeg diagnostic must survive, not be swallowed.
    assert len(message) > len(str(tmp_path / "nope.wav")) + 40


def test_undecodable_file_raises(tmp_path):
    bogus = tmp_path / "bogus.wav"
    bogus.write_bytes(b"this is not audio")
    with pytest.raises(RuntimeError):
        decode_mono(str(bogus), 24000)
