"""Hardware roofline probes for the RTX 3090 (SM 8.6) under WSL2.

Establishes, on the actual box, the facts the 2x plan rests on:
  * fp16-with-fp16-accumulate is a SECOND tensor-core datapath PyTorch does not
    use by default, and on the decode GEMM shape it reaches the HBM roof where
    bf16 and fp16-with-fp32-accumulate both stall ~30% short of it;
  * fp8 is unavailable (torch._scaled_mm needs SM >= 8.9);
  * torch's W8A8 int8 GEMM reaches only ~60 of the datasheet's 284 TOPS, which
    is why q8 must be weight-only (W8A16) rather than W8A8;
  * kernel launch and D2H sync costs, which set the per-step overhead floor.

Run with nothing else on the card. `python probe-hardware.py`
"""
import statistics as st
import time

import torch

M = torch.backends.cuda.matmul


def t_ms(f, iters=200, warm=20):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


# Above this much VRAM already in use, the roofline numbers are meaningless.
# Measured 2026-08-28 with the patmir server resident (21.8 GB): HBM read as
# 295 GB/s instead of 798, kernel launch 44.7 us instead of 17.0, and W8A8
# _int_mm 24 TOPS instead of 60. The dtype RATIOS survived contention (fp16+accum
# 1.41x vs 1.49x) but every absolute figure did not -- and it is the absolute
# 10 ms/step bandwidth floor that decides whether q8 and fp16 are worth anything.
# So refuse rather than emit a third of the real bandwidth as if it were real.
VRAM_BUDGET_MB = 2048


def require_idle_card():
    free, total = torch.cuda.mem_get_info()
    used_mb = (total - free) / 1e6
    if used_mb > VRAM_BUDGET_MB:
        raise SystemExit(
            f"REFUSING TO MEASURE: {used_mb:.0f} MB of VRAM already in use "
            f"(budget {VRAM_BUDGET_MB} MB).\n"
            "Roofline and launch-overhead numbers are invalid on a contended card. "
            "Stop the patmir server (and any other CUDA process) and re-run."
        )
    print(f"card idle: {used_mb:.0f} MB in use\n")


def header():
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"device {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
    print(f"bf16 supported {torch.cuda.is_bf16_supported()}\n")


def bandwidth():
    """Real achievable HBM bandwidth. Preallocate the destination: allocating a
    fresh clone each iteration measures the allocator, not the memory system."""
    n = int(1e9 // 2)
    x = torch.randn(n, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    ms = t_ms(lambda: y.copy_(x), iters=30, warm=5)
    print(f"HBM read+write : {(2 * x.numel() * 2) / (ms / 1000) / 1e9:.0f} GB/s  (spec 936)")
    del x, y
    torch.cuda.empty_cache()


def decode_gemm():
    """The shape that dominates Higgs TTS decode: [B=8, 4096] @ [4096, 14336].

    Interleaved A/B/A across rounds, because sequential blocks drift with clock
    state and a 1.5x claim must not rest on drift.
    """
    K, N = 4096, 14336
    nbytes = K * N * 2
    variants = []
    for dt, fp16acc, bf16red, name in [
        (torch.bfloat16, False, False, "bf16 default"),
        (torch.bfloat16, False, True, "bf16 +reduced_reduction"),
        (torch.float16, False, False, "fp16 default (fp32 acc)"),
        (torch.float16, True, False, "fp16 +fp16_accum"),
    ]:
        a = torch.randn(8, K, device="cuda", dtype=dt)
        W = torch.randn(K, N, device="cuda", dtype=dt)
        variants.append((name, a, W, fp16acc, bf16red))

    res = {n: [] for n, *_ in variants}
    for _ in range(5):
        for name, a, W, fp16acc, bf16red in variants:
            M.allow_fp16_accumulation = fp16acc
            M.allow_bf16_reduced_precision_reduction = bf16red
            res[name].append(t_ms(lambda: a @ W, iters=300, warm=30))

    print(f"\ndecode GEMM [8,{K}]@[{K},{N}]  weight {nbytes/1e6:.0f} MB"
          f"  HBM roof {nbytes/798e9*1000:.3f} ms")
    base = None
    for name, v in res.items():
        m = st.median(v)
        base = base if base is not None else m
        print(f"  {name:26s} {m:.3f} ms  {nbytes/(m/1000)/1e9:5.0f} GB/s  {base/m:.2f}x")
    M.allow_fp16_accumulation = False
    M.allow_bf16_reduced_precision_reduction = False


def roofline_sweep():
    """Where the fp16-accum win comes from, by arithmetic intensity (2B flops/byte)."""
    K, N = 4096, 14336
    print("\nfp16+accum speedup vs bf16 by batch (flops/byte = 2B):")
    for B in [1, 8, 16, 64, 256, 1024]:
        row = []
        for dt, flag in [(torch.bfloat16, False), (torch.float16, True)]:
            M.allow_fp16_accumulation = flag
            a = torch.randn(B, K, device="cuda", dtype=dt)
            W = torch.randn(K, N, device="cuda", dtype=dt)
            row.append(t_ms(lambda: a @ W, iters=200))
            del a, W
            torch.cuda.empty_cache()
        print(f"  B={B:5d}  ai={2*B:6d}   bf16 {row[0]:6.3f} ms   fp16+acc {row[1]:6.3f} ms"
              f"   {row[0]/row[1]:.2f}x")
    M.allow_fp16_accumulation = False


def quantization_paths():
    """fp8 availability, and the W8A8 rate that rules W8A8 out (not W8A16)."""
    print()
    try:
        a = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
        one = torch.tensor(1.0, device="cuda")
        torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        print("fp8 _scaled_mm : OK")
    except Exception as e:  # noqa: BLE001 - the message IS the result
        print(f"fp8 _scaled_mm : UNAVAILABLE - {str(e)[:110]}")

    for n in (4096, 8192):
        a = torch.randint(-8, 8, (n, n), device="cuda", dtype=torch.int8)
        b = torch.randint(-8, 8, (n, n), device="cuda", dtype=torch.int8)
        ms = t_ms(lambda: torch._int_mm(a, b), iters=30, warm=5)
        print(f"W8A8 _int_mm n={n}: {ms:6.2f} ms -> {2*n**3/(ms/1000)/1e12:5.1f} TOPS  (spec 284)")
        del a, b
        torch.cuda.empty_cache()


def overheads():
    """The per-step overhead constants. The decode loop pays one .cpu() PER ROW
    PER STEP, so the sync figure is multiplied by the batch size."""
    print()
    z = torch.zeros(1, device="cuda")
    print(f"tiny kernel launch : {t_ms(lambda: z.add_(1), iters=2000, warm=100)*1000:.1f} us")
    a = torch.zeros(4, device="cuda")
    print(f"tensor.cpu() sync  : {t_ms(lambda: a.cpu(), iters=500, warm=50)*1000:.1f} us")


def accum_numerics():
    """fp16 accumulate is not free: quantify the precision cost and check how
    far real activations sit from fp16's 65504 ceiling."""
    print()
    torch.manual_seed(0)
    K, N = 4096, 14336
    a = (torch.randn(8, K, device="cuda") * 0.5).half()
    W = (torch.randn(K, N, device="cuda") * 0.05).half()
    exact = a.float() @ W.float()
    M.allow_fp16_accumulation = False
    f32 = (a @ W).float()
    M.allow_fp16_accumulation = True
    f16 = (a @ W).float()
    M.allow_fp16_accumulation = False
    den = exact.abs().mean()
    print(f"fp32-accum rel err : {(f32-exact).abs().mean()/den:.3e}")
    print(f"fp16-accum rel err : {(f16-exact).abs().mean()/den:.3e}"
          f"  ({((f16-exact).abs().mean()/(f32-exact).abs().mean()):.0f}x worse)")
    print(f"activation absmax  : {f16.abs().max():.1f}  (fp16 ceiling 65504)")


if __name__ == "__main__":
    header()
    require_idle_card()
    bandwidth()
    decode_gemm()
    roofline_sweep()
    quantization_paths()
    overheads()
    accum_numerics()
