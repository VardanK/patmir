"""Throwaway spike: demucs-mlx vs PyTorch htdemucs_ft — parity + speed.
Slices 45s of the real source, separates via (a) PyTorch htdemucs_ft (current
prod path), (b) demucs-mlx htdemucs_ft, (c) demucs-mlx htdemucs (base), and
compares the vocals/accompaniment stems + times each.
"""
import subprocess, time, sys
from pathlib import Path
import numpy as np

SRC = Path("/Users/vardan.kurkchiyan/.patmir/downloads/183f3cc80610ea04/source.webm")
TMP = Path("spike/_dm_clip.wav")
SECONDS = 45

def slice_clip():
    subprocess.run(["ffmpeg","-y","-ss","0","-t",str(SECONDS),"-i",str(SRC),
        "-vn","-ac","2","-ar","44100","-c:a","pcm_s16le",str(TMP)],check=True,capture_output=True)

def load_wav(p):
    import soundfile as sf
    y,sr = sf.read(str(p), dtype="float32", always_2d=True)  # [n, ch]
    return y.T, sr  # [ch, n]

def pytorch_ft():
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import soundfile as sf
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = get_model("htdemucs_ft"); model.to(dev); model.eval()
    names = list(model.sources)
    y, sr = load_wav(TMP)  # already 44100
    wav = torch.from_numpy(y)[None]  # [1, ch, n]
    t0=time.time()
    with torch.no_grad():
        src = apply_model(model, wav.to(dev), shifts=1, overlap=0.25, split=True)[0].cpu().numpy()
    dt=time.time()-t0
    voc = src[names.index("vocals")]
    acc = sum(src[i] for i,n in enumerate(names) if n!="vocals")
    return voc, acc, dt, dev

def mlx_sep(model_name):
    from demucs_mlx import Separator
    sep = Separator(model=model_name, shifts=1, overlap=0.25, split=True)
    t0=time.time()
    out = sep.separate_audio_file(str(TMP))
    dt=time.time()-t0
    stems = out[1] if isinstance(out, tuple) else out
    def arr(x):
        x = np.asarray(x, dtype="float32")
        return x.T if x.shape[0] > x.shape[-1] else x   # ensure [ch, n]
    voc = arr(stems["vocals"])
    acc = sum(arr(v) for k,v in stems.items() if k!="vocals")
    return voc, acc, dt

def compare(a, b, label):
    n = min(a.shape[-1], b.shape[-1])
    a = a[..., :n].reshape(-1); b = b[..., :n].reshape(-1)
    if a.shape != b.shape:
        print(f"  {label}: shape mismatch {a.shape} vs {b.shape}"); return
    corr = float(np.corrcoef(a, b)[0,1])
    diff = a - b
    sig = float(np.sqrt(np.mean(a**2))) or 1e-9
    res = float(np.sqrt(np.mean(diff**2)))
    snr = 20*np.log10(sig/ (res or 1e-9))
    print(f"  {label}: corr={corr:.4f}  residual_RMS/signal={res/sig:.4f}  SNR={snr:.1f} dB  maxabsdiff={float(np.max(np.abs(diff))):.4f}")

slice_clip()
print(f"clip: {SECONDS}s stereo 44.1k\n")
pv, pa, pdt, dev = pytorch_ft()
print(f"PyTorch htdemucs_ft ({dev}):   {pdt:6.1f}s  ({pdt/SECONDS:.2f}x realtime)")
for name in ("htdemucs_ft","htdemucs"):
    try:
        mv, ma, mdt = mlx_sep(name)
        print(f"demucs-mlx {name:12s}:    {mdt:6.1f}s  ({mdt/SECONDS:.2f}x realtime)  speedup vs MPS={pdt/mdt:.1f}x")
        if name=="htdemucs_ft":
            print("  parity vs PyTorch htdemucs_ft:")
            compare(pv, mv, "vocals     ")
            compare(pa, ma, "accompaniment")
    except Exception as e:
        import traceback; print(f"demucs-mlx {name} FAILED: {e}"); traceback.print_exc()
TMP.unlink(missing_ok=True)
