import time, torch, sys
from patmir.eval.candidates.tts import higgs_v3_torch as H

TEXTS = [
 "Բարև ձեզ, ես եկել եմ ձեզ օգնելու համար այսօր։",
 "Նա ասաց որ մեքենան կայանված է շենքի հետևում։",
 "Ինչու՞ եք այդպես նայում ինձ, ես ոչինչ չեմ արել։",
 "Վրան արյուն կա, և ես չգիտեմ ինչ է կատարվել այստեղ։",
 "Մենք պետք է շտապ հեռանանք այս վայրից հիմա։",
 "Ես ստացել եմ քո հաղորդագրությունը երեկ երեկոյան։",
 "Դա շատ երկար պատմություն է, բայց ես կփորձեմ բացատրել։",
 "Ոչ ոք չգիտի թե ինչ է պատահել այն գիշերը իրականում։",
]

def run(dtype, acc, label):
    torch.backends.cuda.matmul.allow_fp16_accumulation = acc
    t0=time.perf_counter()
    c = H.build({"dtype": dtype, "max_batch": 8, "device":"cuda"})
    build_s=time.perf_counter()-t0
    reqs=[{"text":t} for t in TEXTS]
    steps={}
    def on_chunk(idx, wall): steps.setdefault("chunks",[]).append((len(idx), round(wall,3)))
    # warm
    c.synthesize_batch(reqs, on_chunk=None)
    torch.cuda.synchronize(); t=time.perf_counter()
    outs = c.synthesize_batch(reqs, on_chunk=on_chunk)
    torch.cuda.synchronize(); wall=time.perf_counter()-t
    audio_s = sum(len(o) for o in outs)/ (2*24000)   # rough: 16-bit mono wav bytes
    vram=torch.cuda.max_memory_allocated()/1e9
    print(f"{label:14s} wall {wall*1000:8.1f} ms  build {build_s:5.1f}s  vram {vram:5.2f} GB  chunks {steps.get('chunks')}", flush=True)
    md5s=[__import__('hashlib').md5(o).hexdigest()[:10] for o in outs]
    del c; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return wall, md5s

r={}
for label,dt,acc in [("bf16","bfloat16",False),("fp16_f16acc","float16",True),("bf16_again","bfloat16",False)]:
    try:
        r[label]=run(dt,acc,label)
    except Exception as e:
        print(f"{label:14s} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        r[label]=None
print()
if r.get("bf16"):
    b=r["bf16"][0]
    for k,v in r.items():
        if v: print(f"  {k:14s} {v[0]*1000:8.1f} ms   {b/v[0]:.3f}x")
    print()
    if r.get("fp16_f16acc"): print("  fp16 audio == bf16 audio:", r['fp16_f16acc'][1]==r['bf16'][1])
    if r.get("bf16_again"):  print("  bf16 reproducible       :", r['bf16_again'][1]==r['bf16'][1])
