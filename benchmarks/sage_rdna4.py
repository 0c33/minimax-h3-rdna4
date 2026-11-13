"""SageAttention-2-style attention for RDNA4 (gfx1201) in Triton, plus exact FlashAttention-4 rescaling.

QK_MODE 0: bf16 Q.K^T (exact), 1: int8 (per-token scales, K mean-smoothed), 2: fp8 e4m3 (per-token scales, smoothed)
PV_MODE 0: bf16 P.V (exact),   1: fp8 e4m3 (P scaled by 448, V per-channel scales)
LAZY 1: FlashAttention-4 conditional rescaling: the running max only moves (and the accumulator is only rescaled)
        when a row's max grows by more than 2^8, so most key blocks skip the rescale. Same math.
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
P_SCALE = 448.0  # e4m3 max: P in [0, 1] is stored as P * 448


def _pad(x, mult=128):
    pad = -x.shape[2] % mult
    return torch.nn.functional.pad(x, (0, 0, 0, pad)) if pad else x


def smooth_quant(x, mode, smooth):
    """(B, H, S, D) -> quantized (B, H, Spad, D) contiguous (zero rows up to a multiple of 128) and per-token fp32
    scales (B, H, Spad)."""
    xf = x.float()
    if smooth:
        xf = xf - xf.mean(dim=2, keepdim=True)  # K only: q.mean(k) is constant per query row, softmax ignores it
    qmax = 127.0 if mode == 1 else 448.0
    s = xf.abs().amax(-1).clamp_min(1e-12) / qmax
    y = xf / s[..., None]
    y = y.round().clamp(-127, 127).to(torch.int8) if mode == 1 else y.to(FP8)
    return _pad(y).contiguous(), _pad(s[..., None])[..., 0].contiguous()


def quant_v(v):
    vf = v.float()
    s = vf.abs().amax(2).clamp_min(1e-12) / 448.0  # (B, H, D) per channel
    return _pad((vf / s[:, :, None, :]).to(FP8)).contiguous(), s.contiguous()


@triton.jit
def _fwd(Q, K, V, O, QS, KS, VS, qk_scale,
         s_qb, s_qh, s_qm, s_kb, s_kh, s_kn, s_vb, s_vh, s_vn, s_ob, s_oh, s_om,
         H, N_Q, N_KV, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
         QK_MODE: tl.constexpr, PV_MODE: tl.constexpr, LAZY: tl.constexpr, N_QP, N_KVP):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    if QK_MODE != 0:  # quantized q is zero-padded to a multiple of 128 rows
        q = tl.load(Q + b * s_qb + h * s_qh + offs_m[:, None] * s_qm + offs_d[None, :])
        qsc = tl.load(QS + bh * N_QP + offs_m) * qk_scale
    else:
        q = tl.load(Q + b * s_qb + h * s_qh + offs_m[:, None] * s_qm + offs_d[None, :], mask=offs_m[:, None] < N_Q, other=0.0)
    k_base = K + b * s_kb + h * s_kh
    v_base = V + b * s_vb + h * s_vh
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    for start_n in range(0, N_KV, BLOCK_N):
        cols = start_n + offs_n
        kmask = cols < N_KV
        if QK_MODE != 0:
            k = tl.load(k_base + cols[:, None] * s_kn + offs_d[None, :])
        else:
            k = tl.load(k_base + cols[:, None] * s_kn + offs_d[None, :], mask=kmask[:, None], other=0.0)
        if QK_MODE == 1:
            s = tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
        else:
            s = tl.dot(q, tl.trans(k))
        if QK_MODE != 0:
            ksc = tl.load(KS + bh * N_KVP + cols)
            s = s * qsc[:, None] * ksc[None, :]
        else:
            s = s * qk_scale
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_blk = tl.max(s, 1)
        if LAZY:
            m_new = tl.where(m_blk > m_i + 8.0, m_blk, m_i)
        else:
            m_new = tl.maximum(m_i, m_blk)
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        if PV_MODE == 1:  # fp8 v is zero-padded like k
            v = tl.load(v_base + cols[:, None] * s_vn + offs_d[None, :])
        else:
            v = tl.load(v_base + cols[:, None] * s_vn + offs_d[None, :], mask=kmask[:, None], other=0.0)
        if PV_MODE == 1:
            acc = tl.dot((p * 448.0).to(v.dtype), v, acc * alpha[:, None])
        else:
            acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        m_i = m_new
    acc = acc / l_i[:, None]
    if PV_MODE == 1:
        vsc = tl.load(VS + bh * HEAD_DIM + offs_d) / 448.0
        acc = acc * vsc[None, :]
    tl.store(O + b * s_ob + h * s_oh + offs_m[:, None] * s_om + offs_d[None, :], acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_Q)


def attention(q, k, v, qk_mode=0, pv_mode=0, lazy=False, scale=None, block_m=128, block_n=32, warps=8, stages=1, wpe=2):
    B, H, NQ, D = q.shape
    NKV = k.shape[2]
    out = torch.empty((B, NQ, H, D), dtype=q.dtype, device=q.device).transpose(1, 2)
    qs = ks = vs = out  # unused placeholders
    if qk_mode:
        q, qs = smooth_quant(q, qk_mode, smooth=False)
        k, ks = smooth_quant(k, qk_mode, smooth=True)
    if pv_mode:
        v, vs = quant_v(v)
    sm = (D ** -0.5 if scale is None else scale) * 1.4426950408889634
    grid = (triton.cdiv(NQ, block_m), B * H)
    _fwd[grid](q, k, v, out, qs, ks, vs, sm,
               q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
               v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
               H, NQ, NKV, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
               QK_MODE=qk_mode, PV_MODE=pv_mode, LAZY=int(lazy), N_QP=q.shape[2], N_KVP=k.shape[2],
               num_warps=warps, num_stages=stages, waves_per_eu=wpe)
    return out
