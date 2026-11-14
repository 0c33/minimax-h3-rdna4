import torch
import triton
import triton.language as tl

# Measured on an RX 9070 XT at 56 heads x 128 and 38k tokens (bf16): 97 TFLOPS vs 72 for SDPA (AOTriton 0.13).
# Searched over BLOCK_M 64-256, BLOCK_N 32-128, 4/8 warps, 1/2 stages, waves_per_eu 0/2.
BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES, WAVES_PER_EU = 128, 32, 8, 1, 2


def supported_device(device):
    return (device.type == "cuda" and torch.version.hip is not None
            and torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] in ("gfx1200", "gfx1201"))


@triton.jit
def _attn_fwd(Q, K, V, O, sm_scale,
              s_qb, s_qh, s_qm, s_kb, s_kh, s_kn, s_vb, s_vh, s_vn, s_ob, s_oh, s_om,
              H, N_Q, N_KV, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, P_BF16: tl.constexpr):
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
    qk_scale = sm_scale * 1.4426950408889634  # softmax in base 2
    n_full = (N_KV // BLOCK_N) * BLOCK_N
    for start_n in range(0, n_full, BLOCK_N):
        k = tl.load(k_base + (start_n + offs_n)[:, None] * s_kn + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + (start_n + offs_n)[:, None] * s_vn + offs_d[None, :])
        if P_BF16:
            # gfx12 has no fp32->bf16 instruction, and the compiler's round-to-nearest-even also handles NaN, which
            # p (in [0, 1]) never is: the same rounding without it is ~4% faster and bit-identical
            pi = p.to(tl.uint32, bitcast=True)
            pi = pi + 0x7FFF + ((pi >> 16) & 1)
            acc = tl.dot((pi >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True), v, acc * alpha[:, None])
        else:
            acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
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


def attention(q, k, v, scale=None):
    """q (B, H, Nq, D), k and v (B, H, Nkv, D), D contiguous. Like SDPA, the output keeps the query's token/head
    order, so a (B, Nq, H, D) query (the usual transposed view) gives an output whose later reshape is free."""
    B, H, NQ, D = q.shape
    if q.stride(1) < q.stride(2):
        out = torch.empty((B, NQ, H, D), dtype=q.dtype, device=q.device).transpose(1, 2)
    else:
        out = torch.empty((B, H, NQ, D), dtype=q.dtype, device=q.device)
    grid = (triton.cdiv(NQ, BLOCK_M), B * H)
    _attn_fwd[grid](q, k, v, out, D ** -0.5 if scale is None else scale,
                    q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
                    H, NQ, k.shape[2], HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, P_BF16=q.dtype == torch.bfloat16,
                    num_warps=NUM_WARPS, num_stages=NUM_STAGES, waves_per_eu=WAVES_PER_EU)
    return out
