import time, sys, json
import numpy as np, soundfile as sf, torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
MID="Qwen/Qwen3-ASR-1.7B-hf"
wav,sr = sf.read(sys.argv[1], dtype="float32", always_2d=False)
print("audio", len(wav)/sr, "s @", sr)
clips=[wav]*8                       # 8 x 28s, matching chunk_seconds=28 / batch_size=8
proc = AutoProcessor.from_pretrained(MID)
res={}
for name, dt, acc in [("bf16",torch.bfloat16,False),("fp16_f16acc",torch.float16,True),("bf16_again",torch.bfloat16,False)]:
    torch.backends.cuda.matmul.allow_fp16_accumulation = acc
    m = Qwen3ASRForConditionalGeneration.from_pretrained(MID, dtype=dt).to("cuda").eval()
    inputs = proc.apply_transcription_request(audio=clips, language=["Russian"]*8).to("cuda", dt)
    print(f"  [{name}] input_ids {tuple(inputs['input_ids'].shape)}")
    with torch.inference_mode():
        ids = m.generate(**inputs, max_new_tokens=256)          # warm
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(3): ids = m.generate(**inputs, max_new_tokens=256)
        torch.cuda.synchronize(); ms=(time.perf_counter()-t)/3*1000
    gen = ids[:, inputs["input_ids"].shape[1]:]
    parsed = proc.decode(gen, return_format="parsed")
    parsed = parsed if isinstance(parsed,list) else [parsed]
    texts=[(p.get("text") if isinstance(p,dict) else str(p)) for p in parsed]
    res[name]={"ms":round(ms,1),"new_tokens":int(gen.shape[1]),"texts":texts}
    print(f"  [{name}] generate {ms:8.1f} ms   new_tokens {gen.shape[1]}   text0={texts[0]!r}"[:220], flush=True)
    del m; torch.cuda.empty_cache()
print()
b=res["bf16"]["ms"]
for k,v in res.items(): print(f"  {k:12s} {v['ms']:8.1f} ms  {b/v['ms']:.3f}x  tokens {v['new_tokens']}")
print()
print("  fp16 text == bf16 text :", res["fp16_f16acc"]["texts"]==res["bf16"]["texts"])
print("  bf16 run-to-run stable :", res["bf16_again"]["texts"]==res["bf16"]["texts"])
if res["fp16_f16acc"]["texts"]!=res["bf16"]["texts"]:
    print("  bf16:", res["bf16"]["texts"][0])
    print("  fp16:", res["fp16_f16acc"]["texts"][0])
