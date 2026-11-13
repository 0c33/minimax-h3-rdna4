"""Does Triton on this GPU compile tl.dot for int8 and fp8, and how fast is a big GEMM in each type?"""
import time, torch, triton, triton.language as tl

@triton.jit
def mm(A, B, C, M, N, K, sa, sb, sc, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, OUT_INT: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], tl.int32 if OUT_INT else tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:, None] * sa + (k + rk)[None, :])
        b = tl.load(B + rn[:, None] * sb + (k + rk)[None, :])
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(C + rm[:, None] * sc + rn[None, :], acc.to(tl.float32))

M = N = 8192; K = 4096
for name, dt in [("bf16", torch.bfloat16), ("int8", torch.int8), ("fp8 e4m3fn", torch.float8_e4m3fn)]:
    try:
        if dt == torch.int8:
            a = torch.randint(-100, 100, (M, K), device="cuda", dtype=dt); b = torch.randint(-100, 100, (N, K), device="cuda", dtype=dt)
        else:
            a = torch.randn(M, K, device="cuda").to(dt); b = torch.randn(N, K, device="cuda").to(dt)
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        best = None
        for bm, bn, bk, w in [(128, 128, 64, 8), (128, 128, 32, 8), (256, 128, 64, 8), (128, 64, 64, 4)]:
            f = lambda: mm[(M // bm, N // bn)](a, b, c, M, N, K, K, K, N, BM=bm, BN=bn, BK=bk, OUT_INT=dt == torch.int8, num_warps=w, num_stages=2)
            f(); torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10): f()
            torch.cuda.synchronize(); dt_ = (time.time() - t0) / 10
            r = 2 * M * N * K / dt_ / 1e12
            best = max(best or 0, r)
        ref = a.float() @ b.float().T if dt != torch.int8 else a.float() @ b.float().T
        print(f"{name:12s} tl.dot OK  best {best:6.1f} TFLOPS/TOPS  max err {(c - ref).abs().max().item():.3g}", flush=True)
    except Exception as e:
        print(f"{name:12s} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
