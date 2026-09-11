"""Render one fixed request set through the TTS candidate with CUDA graphs ON
vs OFF and md5 each returned WAV.

Settles a contradiction in the project's own docs: ADR 2026-09-02 says graph
replay is bitwise identical to eager stepping (which is why `cuda_graphs` sits
in `ui.cache._NON_OUTPUT_KEYS` and so does not move the cache key), while
`_capture_decode_graph`'s docstring says the graph attends over `seq + max_cap`
columns against the fallback's `seq + step + 1` — "same distribution, different
draw". If the docstring is right, flipping the flag re-renders audio under an
unchanged key.

One process per setting, so neither run can be perturbed by the other's memory.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

run_dir, graphs, out_path = Path(sys.argv[1]), sys.argv[2] == "1", Path(sys.argv[3])
os.environ["PATMIR_PROFILE"] = "cuda"

from patmir.env import load_dotenv
load_dotenv()
from patmir.pipeline.config import load_pipeline_config

cfg = load_pipeline_config(Path("/home/vardan/patmir/patmir.toml"))
entry = dict(cfg.entries["tts"])
entry["cuda_graphs"] = graphs          # the flag under test
entry.pop("fixed_geometry", None)      # off, as shipped

from patmir.eval.candidates.tts.higgs_v3_torch import build
tts = build(entry)

# Real sentences and a real pinned reference, so the prompt shape matches
# production rather than a synthetic best case.
tr = json.loads((run_dir / "translated.json").read_text())
sents = tr["sentences"] if isinstance(tr, dict) else tr
refs = json.loads((run_dir / "diarize" / "refs.json").read_text())


def pick_ref():
    """The real `refs.json` shape: keys are `<speaker>__<emotion>` and the path
    lives under `ref_audio`, not `path`. Prefer a neutral cell, matching what
    `tts_pin_speaker_ref` + `tts_pin_neutral_fraction` would choose."""
    items = list(refs.items()) if isinstance(refs, dict) else [(str(i), v) for i, v in enumerate(refs)]
    items.sort(key=lambda kv: (0 if "neutral" in kv[0] else 1, kv[0]))
    for _, v in items:
        if isinstance(v, dict) and v.get("ref_audio"):
            q = Path(v["ref_audio"])
            if not q.is_absolute():
                q = run_dir / q
            if q.exists():
                return str(q), (v.get("ref_text") or "")
    return None, None


ref_audio, ref_text = pick_ref()
requests = []
for s in sents:
    t = (s.get("target_text") or "").strip()
    if t:
        requests.append({"text": t, "ref_audio": ref_audio, "ref_text": ref_text,
                         "emotion": s.get("emotion")})
    if len(requests) >= 24:            # two chunks at max_batch 12
        break

wavs = tts.synthesize_batch(requests)
out = {
    "cuda_graphs": graphs,
    "n": len(wavs),
    "ref_audio": ref_audio,
    "summary": tts.summarize() if hasattr(tts, "summarize") else {},
    "md5": [hashlib.md5(w).hexdigest() for w in wavs],
    "texts": [r["text"][:60] for r in requests],
}
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"graphs={graphs} n={len(wavs)} captures={out['summary'].get('graph_captures')} "
      f"fallbacks={out['summary'].get('graph_fallbacks')}")
