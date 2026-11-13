"""Checks custom_nodes/triton_attention/kernel.py against SDPA on the cases ComfyUI can send it."""
import sys
import torch
import torch.nn.functional as F
import importlib.util
spec = importlib.util.spec_from_file_location("ta_kernel", __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "custom_nodes", "triton_attention", "kernel.py"))
kmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(kmod)
attention = kmod.attention

def ref(q, k, v, scale=None):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)

torch.manual_seed(0)
cases = [
    ("H3-like strided bf16 B1", dict(B=1, H=56, NQ=4097, NKV=4097, dt=torch.bfloat16, layout="bshd")),
    ("contiguous bf16 B2 (CFG batch)", dict(B=2, H=24, NQ=4608, NKV=4608, dt=torch.bfloat16, layout="bhsd")),
    ("cross-attn NQ != NKV", dict(B=2, H=24, NQ=4096, NKV=1333, dt=torch.bfloat16, layout="bhsd")),
    ("fp16", dict(B=1, H=16, NQ=2048, NKV=2048, dt=torch.float16, layout="bshd")),
    ("custom scale", dict(B=1, H=8, NQ=3000, NKV=2500, dt=torch.bfloat16, layout="bhsd", scale=0.05)),
]
ok = True
for name, c in cases:
    B, H, NQ, NKV, dt = c["B"], c["H"], c["NQ"], c["NKV"], c["dt"]
    def mk(n):
        if c["layout"] == "bshd":
            return torch.randn(B, n, H, 128, device="cuda", dtype=dt).transpose(1, 2)
        return torch.randn(B, H, n, 128, device="cuda", dtype=dt)
    q, k, v = mk(NQ), mk(NKV), mk(NKV)
    o = attention(q, k, v, c.get("scale"))
    s = F.scaled_dot_product_attention(q, k, v, scale=c.get("scale"))
    r = ref(q, k, v, c.get("scale"))
    e_tri = ((o.float() - r).norm() / r.norm()).item()
    e_sdpa = ((s.float() - r).norm() / r.norm()).item()
    same_layout = o.stride() == s.stride()
    good = o.shape == s.shape and e_tri <= e_sdpa * 1.05 + 1e-6
    ok &= good
    print(f"{name:32s} err triton {e_tri:.2e} sdpa {e_sdpa:.2e} strides {'match' if same_layout else f'{o.stride()} vs {s.stride()}'} {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "SOME FAILED")
