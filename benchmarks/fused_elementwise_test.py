"""fused_test.py - fused SwiGLU and AdaLN scale/shift kernels vs the eager ops: bit-exactness and speed."""
import importlib.util
import sys
import time

import torch
import torch.nn.functional as F

import os
spec = importlib.util.spec_from_file_location("fk", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_nodes", "fused_elementwise", "kernels.py"))
fk = importlib.util.module_from_spec(spec); spec.loader.exec_module(fk)
torch.set_grad_enabled(False)
g = torch.Generator(device="cuda").manual_seed(0)


def bench(fn, iters=10):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


for name, rows, width, dt in [("H3 DiT MLP 5 s", 38196, 2 * 14336, torch.bfloat16), ("H3 VAE FFN batch", 4 * 1797, 2 * 8192, torch.float16),
                              ("odd width", 1000, 2 * 1001, torch.bfloat16)]:
    x = torch.randn(rows, width, device="cuda", dtype=dt, generator=g) * 3
    gate, up = x.chunk(2, dim=-1)
    ref = F.silu(gate).mul_(up)
    out = fk.swiglu(x)
    mism = (out != ref).sum().item()
    te = bench(lambda: F.silu(x.chunk(2, dim=-1)[0]).mul_(x.chunk(2, dim=-1)[1]))
    tf = bench(lambda: fk.swiglu(x))
    print(f"swiglu {name}: {mism} of {ref.numel()} values differ from eager | eager {te * 1000:.2f} ms, fused {tf * 1000:.2f} ms")
    del x, gate, up, ref, out
    torch.cuda.empty_cache()

for name, rows, hidden, dt in [("H3 DiT 5 s", 38196, 5376, torch.bfloat16), ("odd", 999, 1001, torch.bfloat16)]:
    h0 = torch.randn(rows, hidden, device="cuda", dtype=dt, generator=g) * 2
    table = torch.randn(9, 6 * hidden, device="cuda", dtype=torch.float32, generator=g) * 0.3  # like adaln_proj output, viewed
    scale, shift = table[:, :hidden], table[:, hidden:2 * hidden]  # non-contiguous views, like x.chunk()
    for mode in ("one row", "per-token rows"):
        if mode == "one row":
            ref = h0.clone().mul_(1.0 + scale[4].to(dt)).add_(shift[4].to(dt))
            out = fk.scale_shift_(h0.clone(), (1.0 + scale[4].to(dt)).contiguous(), shift[4].to(dt).contiguous())
        else:
            idx = torch.randint(0, 9, (rows,), device="cuda", generator=g)
            ref = h0.clone().mul_(1.0 + scale[idx].to(dt)).add_(shift[idx].to(dt))
            out = fk.scale_shift_(h0.clone(), (1.0 + scale.to(dt)).contiguous(), shift.to(dt).contiguous(), idx)
        print(f"scale_shift {name} {mode}: {(out != ref).sum().item()} of {ref.numel()} values differ from eager")
    h = h0.clone()
    a, c = (1.0 + scale[4].to(dt)).contiguous(), shift[4].to(dt).contiguous()
    te = bench(lambda: h.mul_(a).add_(c))
    tf = bench(lambda: fk.scale_shift_(h, a, c))
    print(f"   eager {te * 1000:.2f} ms, fused {tf * 1000:.2f} ms")
    del h0, h, ref, out
    torch.cuda.empty_cache()
