"""real_compare.py FILE... - kernel speed and error on real MiniMax H3 attention inputs (8 captured heads)."""
import glob, os, sys, time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, os.path.dirname(__file__))
import triton_fa, sage_rdna4 as sr
import comfy_kitchen as ck

def ref_rows(q, k, v, rows):
    out = []
    for h in range(q.shape[1]):
        s = (q[0, h, rows].float() @ k[0, h].float().T) * q.shape[-1] ** -0.5
        out.append(torch.softmax(s, -1) @ v[0, h].float())
    return torch.stack(out)

def sdpa(q, k, v):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v)

def sol(tau):
    return lambda q, k, v: ck.sol_attn(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous(), tau=tau).transpose(1, 2)

KERNELS = {"sdpa (current)": sdpa, "triton exact": lambda q, k, v: triton_fa.attention(q, k, v),
           "sage2-style int8+fp8": lambda q, k, v: sr.attention(q, k, v, qk_mode=1, pv_mode=1),
           "int8 QK + bf16 PV": lambda q, k, v: sr.attention(q, k, v, qk_mode=1)}
for tau in (0.5, 0.8, 1.3, 2.0):
    KERNELS[f"sol_attn tau={tau}"] = sol(tau)

for f in sys.argv[1:]:
    d = torch.load(f)
    q, k, v = (d[x].cuda() for x in "qkv")
    S = q.shape[2]
    rows = torch.randperm(S, device="cuda")[:512]
    ref = ref_rows(q, k, v, rows)
    # how concentrated is real attention? share of probability mass in the top 5% of keys
    s = (q[0, :, rows[:64]].float() @ k[0].float().transpose(1, 2)) * 128 ** -0.5
    p = torch.softmax(s, -1)
    top = p.topk(int(S * 0.05), dim=-1).values.sum(-1).mean().item()
    print(f"--- {os.path.basename(f)} S={S}: top 5% of keys hold {top * 100:.0f}% of attention", flush=True)
    for name, fn in KERNELS.items():
        o = fn(q, k, v); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3): fn(q, k, v)
        torch.cuda.synchronize(); dt = (time.time() - t0) / 3 * 7  # 8 captured heads -> 56
        got = o[0][:, rows].float()
        rel = ((got - ref).norm() / ref.norm()).item()
        print(f"  {name:22s} {dt * 1000:7.1f} ms per layer (56 heads) err rel {rel:.2e}", flush=True)
