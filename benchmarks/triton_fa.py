"""Exact flash attention forward (no mask, no dropout) in Triton, autotuned for the GPU it runs on.

Same math as PyTorch's SDPA flash path: bf16/fp16 inputs, fp32 scores and accumulators, online softmax in base 2,
probabilities rounded to the input dtype before the P@V product (as FlashAttention 2 does).
Takes (B, H, S, D) tensors with any strides as long as D is contiguous.
"""
import itertools
import os

import torch
import triton
import triton.language as tl


def _configs():
    out = []
    for bm, bn, warps, stages, wpe in itertools.product((64, 128, 256), (32, 64, 128), (4, 8), (1, 2), (0, 2)):
        if bm * bn > 256 * 64 or (bm == 256 and warps < 4):
            continue
        out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "waves_per_eu": wpe}, num_warps=warps, num_stages=stages))
    if os.environ.get("TRITON_FA_CONFIG"):  # "BM,BN,warps,stages,wpe" pins one config (no autotune)
        bm, bn, warps, stages, wpe = map(int, os.environ["TRITON_FA_CONFIG"].split(","))
        out = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "waves_per_eu": wpe}, num_warps=warps, num_stages=stages)]
    return out


@triton.autotune(configs=_configs(), key=["N_CTX", "HEAD_DIM", "H"])
@triton.jit
def _fwd(Q, K, V, O, sm_scale,
         s_qb, s_qh, s_qm, s_kb, s_kh, s_kn, s_vb, s_vh, s_vn, s_ob, s_oh, s_om,
         H, N_CTX, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + b * s_qb + h * s_qh + offs_m[:, None] * s_qm + offs_d[None, :]
    k_base = K + b * s_kb + h * s_kh
    v_base = V + b * s_vb + h * s_vh
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    n_full = (N_CTX // BLOCK_N) * BLOCK_N
    for start_n in range(0, n_full, BLOCK_N):
        k = tl.load(k_base + (start_n + offs_n)[:, None] * s_kn + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + (start_n + offs_n)[:, None] * s_vn + offs_d[None, :])
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m_i = m_ij
    if n_full < N_CTX:
        cols = n_full + offs_n
        k = tl.load(k_base + cols[:, None] * s_kn + offs_d[None, :], mask=cols[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(cols[None, :] < N_CTX, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + cols[:, None] * s_vn + offs_d[None, :], mask=cols[:, None] < N_CTX, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m_i = m_ij
    acc = acc / l_i[:, None]
    o_ptrs = O + b * s_ob + h * s_oh + offs_m[:, None] * s_om + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def attention(q, k, v, sm_scale=None, out=None):
    """q, k, v: (B, H, S, D), D contiguous. Returns (B, H, S, D) contiguous unless out is given."""
    B, H, S, D = q.shape
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    if out is None:
        out = torch.empty((B, H, S, D), dtype=q.dtype, device=q.device)
    sm_scale = D ** -0.5 if sm_scale is None else sm_scale
    grid = lambda meta: (triton.cdiv(S, meta["BLOCK_M"]), B * H)
    _fwd[grid](q, k, v, out, sm_scale,
               q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
               v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
               H, S, HEAD_DIM=D)
    return out
