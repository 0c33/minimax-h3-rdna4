"""fa_sweep.py S CONFIG... - times triton_fa with each pinned config "BM,BN,warps,stages,wpe" at sequence length S.
Triton AMD knobs (TRITON_HIP_*, AMDGCN_*) come from the environment; use a separate TRITON_CACHE_DIR per knob set."""
import importlib
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from attn_compare import H, D, gpu_state, make  # noqa: E402

S = int(sys.argv[1])
q, k, v = make(S)
o_ref = None
for cfg in sys.argv[2:]:
    os.environ["TRITON_FA_CONFIG"] = cfg
    import triton_fa
    triton_fa = importlib.reload(triton_fa)
    try:
        o = triton_fa.attention(q, k, v)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"{cfg:16s} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
        continue
    t_end = time.time() + 3
    while time.time() < t_end:
        triton_fa.attention(q, k, v)
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        triton_fa.attention(q, k, v)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 5
    if o_ref is None:
        o_ref = o
    same = torch.equal(o, o_ref)
    clk, pwr = gpu_state()
    print(f"{cfg:16s} {dt * 1000:8.1f} ms {4 * H * S * S * D / dt / 1e12:6.1f} TFLOPS {'same output' if same else 'output differs'} {clk:.0f} MHz {pwr:.0f} W", flush=True)
