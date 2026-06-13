"""Throwaway: does Scribe's per-second cost drop with chunk size?
Times raw scribe_v2 calls at several audio lengths to settle whether the
progressive-growing-chunk idea can beat the ~3x-realtime wall we observed.
"""
import os, time, subprocess, tempfile
from pathlib import Path
import httpx

# load .env
for line in Path(".env").read_text().splitlines():
    if line.strip() and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("="); os.environ.setdefault(k.strip(), v.strip())

KEY = os.environ["ELEVENLABS_API_KEY"]
SRC = Path("/Users/vardan.kurkchiyan/.patmir/downloads/183f3cc80610ea04/source.webm")
URL = "https://api.elevenlabs.io/v1/speech-to-text"
OUT = Path("spike/d3_throughput/scribe_sweep_result.txt")

def slice_wav(seconds, ar, ac, dest):
    subprocess.run(["ffmpeg","-y","-ss","0","-t",str(seconds),"-i",str(SRC),
                    "-vn","-ac",str(ac),"-ar",str(ar),"-c:a","pcm_s16le",str(dest)],
                   check=True, capture_output=True)

def call(path, secs):
    data = {"model_id":"scribe_v2","diarize":"true","num_speakers":"4"}
    t0 = time.time()
    with open(path,"rb") as f:
        r = httpx.post(URL, headers={"xi-api-key":KEY},
                       files={"file":(path.name,f,"audio/wav")}, data=data, timeout=900)
    dt = time.time()-t0
    ok = r.status_code==200
    nwords = len(r.json().get("words",[])) if ok else -1
    return dt, ok, nwords

lines=[]
def log(s):
    print(s, flush=True); lines.append(s); OUT.write_text("\n".join(lines))

log(f"Scribe scribe_v2 size sweep — source {SRC.name}")
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    for secs in (30, 60, 120):
        w = tmp/f"s{secs}.wav"; slice_wav(secs, 16000, 1, w)
        dt, ok, nw = call(w, secs)
        log(f"  {secs:3d}s 16k-mono ({w.stat().st_size//1024}KB): {dt:6.1f}s  rt={dt/secs:.2f}x  ok={ok} words={nw}")
    # format check: 120s at 44.1k stereo (what the pipeline actually sends)
    w = tmp/"s120_st.wav"; slice_wav(120, 44100, 2, w)
    dt, ok, nw = call(w, 120)
    log(f"  120s 44k-stereo ({w.stat().st_size//1024}KB): {dt:6.1f}s  rt={dt/120:.2f}x  ok={ok} words={nw}")
log("DONE")
