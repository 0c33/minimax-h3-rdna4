"""fa_next.py [filter] - next-round exact attention experiments at H3's 5 s shape (56 heads x 128, 38196 tokens, bf16).

Variants (all exact softmax attention, fp32 accumulation, same math as the shipped kernel):
  base   the shipped kernel (custom_nodes/triton_attention/kernel.py)
  fold   base, but the softmax scale is folded into the exp2 argument (one FMA instead of mul + sub per score)
  trans  computes S^T = K Q^T and O^T = V^T P^T. On gfx12 the WMMA accumulator layout equals the B-operand layout,
         so P^T can feed the second matmul without a shuffle, and the softmax reductions run over the values each
         lane already holds instead of across lanes.
Each line: ms per attention call, TFLOPS, seconds per 50-block step, error vs fp32, VGPRs/spills and instruction
counts from the generated AMDGCN.
"""
import os
import re
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(__file__))
from attn_compare import make, reference, gpu_state, H, D  # noqa: E402


@triton.jit
def fwd_base(Q, K, V, O, sm_scale,
             s_qb, s_qh, s_qm, s_kb, s_kh, s_kn, s_vb, s_vh, s_vn, s_ob, s_oh, s_om,
             H, N_Q, N_KV, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, FOLD: tl.constexpr, P_MODE: tl.constexpr = 0):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + b * s_qb + h * s_qh + offs_m[:, None] * s_qm + offs_d[None, :], mask=offs_m[:, None] < N_Q, other=0.0)
    k_base = K + b * s_kb + h * s_kh
    v_base = V + b * s_vb + h * s_vh
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    n_full = (N_KV // BLOCK_N) * BLOCK_N
    for start_n in range(0, n_full, BLOCK_N):
        k = tl.load(k_base + (start_n + offs_n)[:, None] * s_kn + offs_d[None, :])
        if FOLD:
            qk = tl.dot(q, tl.trans(k))
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        else:
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + (start_n + offs_n)[:, None] * s_vn + offs_d[None, :])
        if P_MODE == 3:
            v = v.to(tl.float16)
        if P_MODE == 0:
            acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        elif P_MODE == 1:
            pi = p.to(tl.uint32, bitcast=True)
            pi = pi + 0x7FFF + ((pi >> 16) & 1)
            acc = tl.dot((pi >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True), v, acc * alpha[:, None])
        elif P_MODE == 2:
            acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc * alpha[:, None])
        else:
            acc = tl.dot(p.to(tl.float16), v, acc * alpha[:, None])
        m_i = m_ij
    if n_full < N_KV:
        cols = n_full + offs_n
        k = tl.load(k_base + cols[:, None] * s_kn + offs_d[None, :], mask=cols[:, None] < N_KV, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(cols[None, :] < N_KV, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + cols[:, None] * s_vn + offs_d[None, :], mask=cols[:, None] < N_KV, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
    acc = acc / l_i[:, None]
    tl.store(O + b * s_ob + h * s_oh + offs_m[:, None] * s_om + offs_d[None, :], acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_Q)


@triton.jit
def fwd_trans(Q, K, V, O, sm_scale,
              s_qb, s_qh, s_qm, s_kb, s_kh, s_kn, s_vb, s_vh, s_vn, s_ob, s_oh, s_om,
              H, N_Q, N_KV, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, FOLD: tl.constexpr):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    qt = tl.load(Q + b * s_qb + h * s_qh + offs_m[None, :] * s_qm + offs_d[:, None], mask=offs_m[None, :] < N_Q, other=0.0)
    k_base = K + b * s_kb + h * s_kh
    v_base = V + b * s_vb + h * s_vh
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([HEAD_DIM, BLOCK_M], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    n_full = (N_KV // BLOCK_N) * BLOCK_N
    for start_n in range(0, n_full, BLOCK_N):
        k = tl.load(k_base + (start_n + offs_n)[:, None] * s_kn + offs_d[None, :])
        s = tl.dot(k, qt)  # [BLOCK_N keys, BLOCK_M queries]
        m_ij = tl.maximum(m_i, tl.max(s, 0) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_ij[None, :])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 0)
        vt = tl.load(v_base + (start_n + offs_n)[None, :] * s_vn + offs_d[:, None])
        acc = tl.dot(vt, p.to(vt.dtype), acc * alpha[None, :])
        m_i = m_ij
    if n_full < N_KV:
        cols = n_full + offs_n
        k = tl.load(k_base + cols[:, None] * s_kn + offs_d[None, :], mask=cols[:, None] < N_KV, other=0.0)
        s = tl.dot(k, qt)
        s = tl.where(cols[:, None] < N_KV, s, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(s, 0) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_ij[None, :])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 0)
        vt = tl.load(v_base + cols[None, :] * s_vn + offs_d[:, None], mask=cols[None, :] < N_KV, other=0.0)
        acc = tl.dot(vt, p.to(vt.dtype), acc * alpha[None, :])
    acc = acc / l_i[None, :]
    tl.store(O + b * s_ob + h * s_oh + offs_m[None, :] * s_om + offs_d[:, None], acc.to(O.dtype.element_ty), mask=offs_m[None, :] < N_Q)


KERNELS = {"base": (fwd_base, False), "fold": (fwd_base, True), "trans": (fwd_trans, True)}
P_MODES = {"fold": 0, "rne": 1, "fp16pv": 2, "fp16pv_load": 3}


def attention(variant, q, k, v, bm, bn, warps, stages, wpe, **opts):
    kern, fold = KERNELS["fold" if variant in P_MODES else variant]
    if variant in P_MODES:
        opts = dict(opts, P_MODE=P_MODES[variant])
    B, Hh, NQ, Dd = q.shape
    out = torch.empty((B, NQ, Hh, Dd), dtype=q.dtype, device=q.device).transpose(1, 2)
    grid = (triton.cdiv(NQ, bm), B * Hh)
    return out, kern[grid](q, k, v, out, Dd ** -0.5,
                           q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                           v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
                           Hh, NQ, k.shape[2], HEAD_DIM=Dd, BLOCK_M=bm, BLOCK_N=bn, FOLD=fold,
                           num_warps=warps, num_stages=stages, waves_per_eu=wpe, **opts)


def asm_stats(ck):
    a = ck.asm.get("amdgcn", "")
    g = lambda pat: (re.search(pat, a) or [None, "?"])[1]
    c = lambda pat: len(re.findall(pat, a, re.M))
    return (f"vgpr {g(r'\.vgpr_count:\s+(\d+)')} spill {g(r'\.vgpr_spill_count:\s+(\d+)')} "
            f"wmma {c(r'^\s+v_wmma')} ds_ld {c(r'^\s+ds_(?:load|read)')} ds_st {c(r'^\s+ds_(?:store|write)')} "
            f"xlane {c(r'^\s+(?:ds_bpermute|ds_swizzle|v_permlane)|_dpp|dpp8|row_')} exp {c(r'^\s+v_exp_f32')}")


def bench(label, variant, q, k, v, rows, ref, cfg, opts, warm=2.0, iters=3):
    try:
        out, ck = attention(variant, q, k, v, *cfg, **opts)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"{label:58s} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None
    t_end = time.time() + warm
    while time.time() < t_end:
        attention(variant, q, k, v, *cfg, **opts)
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        attention(variant, q, k, v, *cfg, **opts)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    clk, pwr = gpu_state()
    S = q.shape[2]
    got = out[0][:, rows].float()
    rel = ((got - ref).norm() / ref.norm()).item()
    tf = 4 * H * S * S * D / dt / 1e12
    print(f"{label:58s} {dt * 1000:7.1f} ms {tf:6.1f} TFLOPS {dt * 50:5.1f} s/step | err {rel:.2e} | "
          f"{clk:.0f} MHz {pwr:.0f} W | {asm_stats(ck)}", flush=True)
    return tf, rel


BASE_CFG = (128, 32, 8, 1, 2)
PLAN = [
    ("base", BASE_CFG, {}),
    ("fold", BASE_CFG, {}),
    ("trans", BASE_CFG, {}),
    ("trans", (128, 64, 8, 1, 2), {}),
    ("trans", (256, 32, 8, 1, 2), {}),
    ("trans", (64, 32, 4, 1, 2), {}),
    ("trans", (128, 32, 4, 1, 2), {}),
    ("trans", (128, 16, 8, 1, 2), {}),
    ("trans", (128, 32, 8, 2, 2), {}),
    ("trans", (256, 64, 8, 1, 2), {}),
    ("trans", (256, 32, 16, 1, 2), {}),
    ("trans", (128, 32, 8, 1, 1), {}),
    ("fold", (128, 64, 8, 1, 2), {}),
    ("fold", BASE_CFG, {"llvm_fn_attrs": "amdgpu-sched-strategy=max-ilp"}),
    ("fold", BASE_CFG, {"llvm_fn_attrs": "amdgpu-sched-strategy=iterative-ilp"}),
    ("fold", BASE_CFG, {"llvm_fn_attrs": "amdgpu-sched-strategy=max-memory-clause"}),
    ("trans", BASE_CFG, {"llvm_fn_attrs": "amdgpu-sched-strategy=max-ilp"}),
    ("trans", BASE_CFG, {"llvm_fn_attrs": "amdgpu-sched-strategy=iterative-ilp"}),
    ("trans", BASE_CFG, {"allow_flush_denorm": True}),
]

if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    S = int(os.environ.get("S", 38196))
    q, k, v = make(S)
    rows = torch.randperm(S, device="cuda")[:512]
    ref = reference(q, k, v, rows)
    print(f"--- S={S} torch {torch.__version__} triton {triton.__version__} env "
          f"{ {k_: v_ for k_, v_ in os.environ.items() if k_.startswith(('TRITON_HIP', 'AMDGCN'))} }", flush=True)
    for variant, cfg, opts in PLAN:
        label = f"{variant:5s} M{cfg[0]} N{cfg[1]} w{cfg[2]} s{cfg[3]} wpe{cfg[4]} {' '.join(f'{a}={b}' for a, b in opts.items())}"
        if flt in label:
            bench(label, variant, q, k, v, rows, ref, cfg, opts)
