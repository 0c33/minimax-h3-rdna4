"""attention_addon_test.py - (run with COMFYUI_DIR=/path/to/ComfyUI) the triton_attention add-on as ComfyUI imports it: HIP path for bf16, Triton for fp16, SDPA for
everything else; accuracy vs fp32 and speed at the H3 shape."""
import os, sys, time
sys.argv = [sys.argv[0], "--disable-comfy-compiler"]
COMFY = os.environ.get("COMFYUI_DIR", "."); os.chdir(COMFY); sys.path.insert(0, os.path.abspath("."))
import comfy.options; comfy.options.enable_args_parsing()
import torch, comfy.ops
import importlib
node = importlib.import_module("custom_nodes.triton_attention")
print("HIP kernel loaded:", node._hip is not None, "| patched:", comfy.ops.scaled_dot_product_attention is node.scaled_dot_product_attention)
sdpa = comfy.ops.scaled_dot_product_attention
g = torch.Generator(device="cuda").manual_seed(5)
def case(name, B, H, NQ, NKV, dt, view=True, **kw):
    mk = lambda n: (torch.randn(B, n, H, 128, device="cuda", dtype=dt, generator=g).transpose(1, 2) if view else torch.randn(B, H, n, 128, device="cuda", dtype=dt, generator=g))
    q, k, v = mk(NQ), mk(NKV), mk(NKV)
    path = ("hip" if node._hip is not None and node.hip_kernel.covered(q, k, v) else "triton") if node._covered(q, k, v, (), kw) else "sdpa"
    out = sdpa(q, k, v, **kw)
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), **kw)
    e = ((out.float() - ref).norm() / ref.norm()).item()
    print(f"{name:34s} -> {path:6s} err {e:.2e} out stride {out.stride()}")
case("bf16 H3-like 2k", 1, 56, 2048, 2048, torch.bfloat16)
case("bf16 batch 2 cross 1200x3000", 2, 8, 1200, 3000, torch.bfloat16, view=False)
case("fp16 2k", 1, 8, 2048, 2048, torch.float16)
case("bf16 custom scale 0.05", 1, 8, 2048, 2048, torch.bfloat16, scale=0.05)
case("bf16 causal -> sdpa", 1, 8, 2048, 2048, torch.bfloat16, is_causal=True)
case("bf16 small 512 -> sdpa", 1, 8, 512, 512, torch.bfloat16)
for dt in (torch.float16, torch.bfloat16):
    qkv = torch.randn(4, 1797, 32 * 3 * 64, device="cuda", dtype=dt, generator=g)
    q, k, v = (t.transpose(1, 2) for t in qkv.view(4, 1797, 32, 3 * 64).chunk(3, dim=-1))
    path = ("hip" if node._hip is not None and node.hip_kernel.covered(q, k, v) else "triton") if node._covered(q, k, v, (), {}) else "sdpa"
    out = sdpa(q, k, v); ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    print(f"VAE layout head 64 {str(dt)[6:]:9s}          -> {path:6s} err {((out.float() - ref).norm() / ref.norm()).item():.2e} out stride {out.stride()}")
case("head 96 -> sdpa", 1, 8, 2048, 2048, torch.bfloat16) if False else None
S, H = 38196, 56
qkv = torch.randn(S, 3 * H * 128, device="cuda", dtype=torch.bfloat16, generator=g)
q, k, v = (t.view(S, H, 128).transpose(0, 1).unsqueeze(0) for t in qkv.split(H * 128, dim=-1))
for name, fn in [("triton", lambda: node.attention(q, k, v)), ("add-on (hip)", lambda: sdpa(q, k, v))] * 2:
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(5): fn()
    torch.cuda.synchronize(); dt = (time.time() - t) / 5
    print(f"{name:14s} {dt * 1000:.1f} ms per H3 attention call -> {dt * 50:.1f} s per 50-block step")
