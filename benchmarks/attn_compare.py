"""attn_compare.py [S ...] - speed and accuracy of attention kernels at MiniMax H3 shapes (56 heads x 128, bf16).

q, k, v are laid out like H3's (views into one [S, 3*H*D] qkv buffer). Accuracy is measured against an fp32
reference on 512 random query rows of every head: relative L2 error and max abs error of the output.
"""
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, os.path.dirname(__file__))
H, D = 56, 128
HW = glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*")[0]


def gpu_state():
    rd = lambda f: int(open(f"{HW}/{f}").read())
    return rd("freq1_input") / 1e6, rd("power1_average") / 1e6


def make(S, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(S, 3 * H * D, device="cuda", dtype=torch.bfloat16, generator=g)
    q, k, v = (t.view(S, H, D).transpose(0, 1).unsqueeze(0) for t in qkv.split(H * D, dim=-1))
    return q, k, v


def reference(q, k, v, rows):
    out = []
    for h in range(H):
        s = (q[0, h, rows].float() @ k[0, h].float().T) * D ** -0.5
        out.append(torch.softmax(s, -1) @ v[0, h].float())
    return torch.stack(out)  # [H, R, D]


def bench(name, fn, q, k, v, rows, ref, iters=3):
    try:
        o = fn(q, k, v)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"{name:28s} failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return
    t_end = time.time() + 4  # sustained load so clocks settle
    while time.time() < t_end:
        fn(q, k, v)
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(q, k, v)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    clk, pwr = gpu_state()
    S = q.shape[2]
    got = o[0][:, rows].float()
    rel = ((got - ref).norm() / ref.norm()).item()
    mx = (got - ref).abs().max().item()
    print(f"{name:28s} {dt * 1000:8.1f} ms/layer {4 * H * S * S * D / dt / 1e12:6.1f} TFLOPS -> {dt * 50:5.1f} s/step | "
          f"err rel {rel:.2e} max {mx:.2e} | {clk:.0f} MHz {pwr:.0f} W", flush=True)


def sdpa(q, k, v):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v)


def sdpa_contig(q, k, v):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())


KERNELS = {"sdpa_flash (current)": sdpa, "sdpa_flash contiguous": sdpa_contig}
if os.environ.get("WITH_TRITON_FA", "1") == "1":
    import triton_fa
    KERNELS["triton_fa (autotuned)"] = lambda q, k, v: triton_fa.attention(q, k, v)
if os.environ.get("SAGE_PATH"):
    sys.path.insert(0, os.environ["SAGE_PATH"])
    from sageattention import sageattn
    KERNELS["sageattention (int8 QK)"] = lambda q, k, v: sageattn(q, k, v, tensor_layout="HND", is_causal=False)

if os.environ.get("AITER_FA_PATH"):
    sys.path.insert(0, os.environ["AITER_FA_PATH"])
    from flash_attn_triton_amd import interface_v2 as aiter_fa

    def aiter(q, k, v):  # takes (B, S, H, D): H3's q/k/v are such views, transposed
        out = aiter_fa.fwd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None, None, 0.0, D ** -0.5, False, -1, -1, 0.0, False)[0]
        return out.transpose(1, 2)
    KERNELS["aiter triton FA (AMD)"] = aiter

if os.environ.get("WITH_SAGE_RDNA4"):
    import sage_rdna4 as sr
    for label, kw in [("bf16 (masked loop)", {}), ("bf16 + FA4 lazy rescale", dict(lazy=True)),
                      ("int8 QK + bf16 PV", dict(qk_mode=1)), ("int8 QK + fp8 PV (Sage2)", dict(qk_mode=1, pv_mode=1)),
                      ("fp8 QK + fp8 PV", dict(qk_mode=2, pv_mode=1)), ("int8 QK + bf16 PV + lazy", dict(qk_mode=1, lazy=True))]:
        KERNELS["rdna4 " + label] = (lambda kw: lambda q, k, v: sr.attention(q, k, v, **kw))(kw)

if os.environ.get("WITH_SOL"):
    import comfy_kitchen as ck
    for tau in [float(t) for t in os.environ["WITH_SOL"].split(",")]:
        KERNELS[f"sol_attn tau={tau} (sparse)"] = (lambda tau: lambda q, k, v: ck.sol_attn(
            q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), tau=tau).transpose(1, 2))(tau)

if __name__ == "__main__":
    for S in [int(a) for a in sys.argv[1:]] or [38196]:
        q, k, v = make(S)
        rows = torch.randperm(S, device="cuda")[:512]
        ref = reference(q, k, v, rows)
        print(f"--- S={S} (torch {torch.__version__})", flush=True)
        for name, fn in KERNELS.items():
            t0 = time.time()
            bench(name, fn, q, k, v, rows, ref)
            if name.startswith("triton_fa"):
                cfg = triton_fa._fwd.best_config
                print(f"    best config {cfg} (tuning took {time.time() - t0:.0f} s)", flush=True)
        del q, k, v, ref
        torch.cuda.empty_cache()
