"""D3 throughput spike — Demucs slice probe (THROWAWAY).

Question: can separation run on a ~20s slice near the playhead (for streaming)
and still produce usable vocals, at faster-than-realtime speed?

Note: production htdemucs already calls apply_model(split=True), i.e. it
windows internally. This probe confirms a *standalone short slice* separates
cleanly (no fatal boundary artifact) and measures per-slice RTF.

Run after the TTS sweep frees the GPU:
  .venv/bin/python spike/d3_throughput/demucs_slice.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260528-010843-source"
SOURCE = json.loads((RUN / "manifest.json").read_text())["source_path"]
SLICE_START = 30.0
SLICE_DUR = 20.0


def main():
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    out_dir = REPO / "spike/d3_throughput"
    slice_wav = out_dir / "demucs_slice_in.wav"
    # Extract a 20s stereo slice at Demucs' native 44.1k.
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(SLICE_START), "-t", str(SLICE_DUR),
        "-i", SOURCE, "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        str(slice_wav),
    ], check=True, capture_output=True)

    model = get_model("htdemucs_ft")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()

    audio_np, sr = sf.read(str(slice_wav), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio_np.T.copy()).unsqueeze(0).to(device)

    # warm-up on a 2s sub-slice
    with torch.no_grad():
        apply_model(model, wav[:, :, : sr * 2], device=device, split=True, progress=False)

    t0 = time.perf_counter()
    with torch.no_grad():
        sources = apply_model(model, wav, device=device, split=True, progress=False)
    elapsed = time.perf_counter() - t0

    voc_idx = list(model.sources).index("vocals")
    vocals = sources[0, voc_idx].transpose(0, 1).cpu().numpy()
    voc_path = out_dir / "demucs_slice_vocals.wav"
    sf.write(str(voc_path), vocals, int(model.samplerate), subtype="PCM_16")

    rtf = elapsed / SLICE_DUR
    result = {
        "slice_seconds": SLICE_DUR,
        "device": device,
        "wall_s": round(elapsed, 2),
        "rtf_vs_slice": round(rtf, 3),
        "vocals_out": str(voc_path),
    }
    (out_dir / "demucs_slice_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nRTF {rtf:.3f} ({'faster' if rtf < 1 else 'slower'} than realtime). "
          f"Listen to {voc_path} to confirm vocals are clean.")


if __name__ == "__main__":
    main()
