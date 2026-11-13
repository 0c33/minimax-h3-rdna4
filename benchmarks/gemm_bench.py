"""gemm_bench.py [rows] - bf16 linear layers of one MiniMax H3 block at a given token count.

Times each projection with the BLAS backend the environment selects (default hipBLASLt; set GEMM_BACKEND=rocblas
for rocBLAS; PYTORCH_TUNABLEOP_* env vars enable PyTorch's TunableOp), and checks the result against an fp32
product on 256 rows.
"""
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F

HW = glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*")[0]
SHAPES = {"qkv_proj": (5376, 21504), "out_proj": (7168, 5376), "mlp.fc1": (5376, 28672), "mlp.fc2": (14336, 5376)}

if os.environ.get("GEMM_BACKEND") == "rocblas":
    torch.backends.cuda.preferred_blas_library("cublas")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 38196
g = torch.Generator(device="cuda").manual_seed(0)
total = 0.0
print(f"--- N={N} torch {torch.__version__} blas {torch.backends.cuda.preferred_blas_library()} tunableop={os.environ.get('PYTORCH_TUNABLEOP_ENABLED', '0')}", flush=True)
ONLY = os.environ.get("GEMM_ONLY")
for name, (k, n) in SHAPES.items():
    if ONLY and name != ONLY:
        continue
    x = torch.randn(N, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5)
    y = F.linear(x, w)
    torch.cuda.synchronize()
    t_end = time.time() + 3
    while time.time() < t_end:
        F.linear(x, w)
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        F.linear(x, w)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 10
    total += dt
    ref = x[:256].float() @ w.float().T
    rel = ((y[:256].float() - ref).norm() / ref.norm()).item()
    clk = int(open(f"{HW}/freq1_input").read()) / 1e6
    print(f"{name:9s} {dt * 1000:7.2f} ms {2 * N * k * n / dt / 1e12:6.1f} TFLOPS err {rel:.2e} {clk:.0f} MHz", flush=True)
    del x, w, y
print(f"one block: {total * 1000:.1f} ms -> {total * 50:.1f} s per step (50 blocks)", flush=True)
