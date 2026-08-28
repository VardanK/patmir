import os, time, json, sys
import numpy as np, soundfile as sf, torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

MID="Qwen/Qwen3-ASR-1.7B-hf"
wav,sr = sf.read(sys.argv[1], dtype="float32", always_2d=False)
assert sr==16000
# batch of 8 identical-length chunks, mirroring batch_size=8 in the candidate
clips=[wav[:int(3.5*16000)+i*100] for i in range(8)]

proc = AutoProcessor.from_pretrained(MID)
out={}
for name, dt, acc in [("bf16", torch.bfloat16, False), ("fp16_f32acc", torch.float16, False), ("fp16_f16acc", torch.float16, True)]:
    torch.backends.cuda.matmul.allow_fp16_accumulation = acc
    t0=time.perf_counter()
    m = Qwen3ASRForConditionalGeneration.from_pretrained(MID, dtype=dt).to("cuda").eval()
    load=time.perf_counter()-t0
    inputs = proc.apply_transcription_request(audio=clips, language=["Russian"]*len(clips)).to("cuda", dt)
    with torch.inference_mode():
        for _ in range(2):  # warm
            ids = m.generate(**inputs, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(3):
            ids = m.generate(**inputs, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize(); ms=(time.perf_counter()-t)/3*1000
    gen = ids[:, inputs["input_ids"].shape[1]:]
    txt = proc.decode(gen, return_format="parsed")
    texts=[ (x.get("text") if isinstance(x,dict) else str(x)) for x in (txt if isinstance(txt,list) else [txt]) ]
    out[name]={"ms":round(ms,1),"load_s":round(load,1),"vram_gb":round(torch.cuda.max_memory_allocated()/1e9,2),"texts":texts[:3]}
    print(f"{name:12s} generate {ms:8.1f} ms   load {load:5.1f}s   vram {out[name]['vram_gb']:.2f} GB", flush=True)
    del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

print()
b=out["bf16"]["ms"]
for k,v in out.items(): print(f"  {k:12s} {v['ms']:8.1f} ms   {b/v['ms']:.2f}x vs bf16")
print()
ref=out["bf16"]["texts"]
for k,v in out.items():
    same = (v["texts"]==ref)
    print(f"  {k:12s} text identical to bf16: {same}")
    if not same:
        for i,(a,c) in enumerate(zip(ref,v["texts"])):
            if a!=c: print(f"     row{i} bf16: {a!r}\n     row{i} {k}: {c!r}")
print()
print("bf16 sample text:", ref[0][:200] if ref else None)
