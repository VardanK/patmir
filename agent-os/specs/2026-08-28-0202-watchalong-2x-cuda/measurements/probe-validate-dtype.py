"""Prove the bf16 / fp16 / fp16+accumulate arms are genuinely distinct computations,
and that a null result at the Higgs decode shape is a real null rather than a harness bug.

Asked because the datasheet lists fp16-with-fp16-accumulate at ~2x bf16, yet at the
model's true MLP shape the two measure the same (in fact fp16 is ~7% slower). The
explanation is the roofline -- arithmetic intensity equals the batch size for a
weight-dominated GEMM, so at B=8 we sit ~10x below the ridge point and FLOPS does not
enter the time equation at all -- but that explanation is only worth anything if the
harness can be shown to detect the difference when it IS there.

Three checks:
  1. the arms differ numerically (non-zero pairwise max|diff|), including the
     same-dtype pair that isolates the accumulate FLAG;
  2. a positive control at a compute-bound shape, where the datapath must show;
  3. a high-repeat interleaved run at the true shape, to separate a tight null from noise.

Run on an IDLE card. `python probe-validate-dtype.py`
"""
import hashlib
import statistics as st
import time

import torch

K, N = 2560, 9728          # the true Higgs MLP shape
M = torch.backends.cuda.matmul
ARMS = [("bf16", torch.bfloat16, False),
        ("fp16_f16acc", torch.float16, True),
        ("fp16_f32acc", torch.float16, False)]


def t_ms(f, iters=300, warm=30):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def distinct_computations():
    print("=== 1: are the arms actually different computations? ===")
    torch.manual_seed(0)
    base = torch.randn(8, K, device="cuda")
    Wb = torch.randn(K, N, device="cuda")
    out = {}
    for name, dt, flag in ARMS:
        M.allow_fp16_accumulation = flag
        a, W = base.to(dt), Wb.to(dt)
        o = a @ W
        md5 = hashlib.md5(o.float().cpu().numpy().tobytes()).hexdigest()[:12]
        print(f"  {name:12s} in={a.dtype} out={o.dtype} flag_readback={M.allow_fp16_accumulation}"
              f"  md5={md5}")
        out[name] = o.float()
    M.allow_fp16_accumulation = False

    print("\n  pairwise max|diff| (exactly 0.0 would mean an identical computation):")
    ks = list(out)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = (out[ks[i]] - out[ks[j]]).abs().max().item()
            note = "  <- same dtype; the FLAG alone changes the result" \
                if {ks[i], ks[j]} == {"fp16_f16acc", "fp16_f32acc"} else ""
            print(f"    {ks[i]:12s} vs {ks[j]:12s}  {d:.6e}{note}")


def positive_control():
    print("\n=== 2: positive control at a COMPUTE-bound shape (AI ~ 8192 >> ridge 78) ===")
    for name, dt, flag in ARMS[:2]:
        M.allow_fp16_accumulation = flag
        n = 8192
        a = torch.randn(n, n, device="cuda", dtype=dt)
        b = torch.randn(n, n, device="cuda", dtype=dt)
        ms = st.median([t_ms(lambda: a @ b, iters=30, warm=10) for _ in range(3)])
        print(f"    {name:12s} {ms:7.2f} ms   {2*n**3/(ms/1000)/1e12:6.1f} TFLOP/s")
        del a, b
        torch.cuda.empty_cache()
    M.allow_fp16_accumulation = False
    print("    -> the harness CAN see the datapath, so a null elsewhere is a real null")


def tight_null():
    print("\n=== 3: true shape, B=8, interleaved A/B x 9 rounds ===")
    res = {n: [] for n, _, _ in ARMS[:2]}
    tens = {n: (torch.randn(8, K, device="cuda", dtype=d),
                torch.randn(K, N, device="cuda", dtype=d), f)
            for n, d, f in ARMS[:2]}
    for _ in range(9):
        for name, (a, W, flag) in tens.items():
            M.allow_fp16_accumulation = flag
            res[name].append(t_ms(lambda: a @ W))
    M.allow_fp16_accumulation = False
    for name, v in res.items():
        print(f"    {name:12s} median {st.median(v):.4f} ms  min {min(v):.4f}  "
              f"max {max(v):.4f}  n={len(v)}")
    r = st.median(res["bf16"]) / st.median(res["fp16_f16acc"])
    print(f"    ratio bf16/fp16acc = {r:.3f}x  "
          f"({'fp16 slower' if r < 1 else 'fp16 faster'})")


def ridge_sweep():
    """Where the crossover actually is, at the true shape. AI == batch size."""
    print("\n=== 4: ridge-point sweep at the true shape (AI = B; bf16 ridge ~78) ===")
    print("   B    bf16 GB/s  fp16acc GB/s   ratio   regime")
    for B in (8, 16, 24, 32, 64, 128):
        got = {}
        for name, dt, flag in ARMS[:2]:
            M.allow_fp16_accumulation = flag
            a = torch.randn(B, K, device="cuda", dtype=dt)
            W = torch.randn(K, N, device="cuda", dtype=dt)
            got[name] = st.median([t_ms(lambda: a @ W) for _ in range(3)])
            del a, W
            torch.cuda.empty_cache()
        gb = lambda ms: K * N * 2 / (ms / 1000) / 1e9  # noqa: E731
        print(f"{B:4d}      {gb(got['bf16']):6.0f}        {gb(got['fp16_f16acc']):6.0f}"
              f"    {got['bf16']/got['fp16_f16acc']:5.2f}x   "
              f"{'memory-bound' if B < 78 else 'COMPUTE-bound'}")
    M.allow_fp16_accumulation = False


if __name__ == "__main__":
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e6
    if used > 2048:
        raise SystemExit(f"REFUSING: {used:.0f} MB VRAM in use. Stop the server and re-run.")
    print(f"card idle: {used:.0f} MB in use\n")
    distinct_computations()
    positive_control()
    tight_null()
    ridge_sweep()
