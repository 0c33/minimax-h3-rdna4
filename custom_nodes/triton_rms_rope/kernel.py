import triton
import triton.language as tl


@triton.jit
def _rms_rope_row(X, W, b, h, t, tm, j, jm, r, rm, s_b, s_t, s_h, f00, f01, f10, f11, eps,
                  D: tl.constexpr, HALF: tl.constexpr, REST: tl.constexpr):
    base = X + b * s_b + h * s_h + t[:, None] * s_t
    m2 = tm[:, None] & jm[None, :]
    x0 = tl.load(base + j[None, :], mask=m2, other=0.0).to(tl.float32)
    x1 = tl.load(base + HALF + j[None, :], mask=m2, other=0.0).to(tl.float32)
    ss = tl.sum(x0 * x0, 1) + tl.sum(x1 * x1, 1)
    if REST > 0:
        mr = tm[:, None] & rm[None, :]
        xr = tl.load(base + 2 * HALF + r[None, :], mask=mr, other=0.0).to(tl.float32)
        ss += tl.sum(xr * xr, 1)
    inv = tl.math.rsqrt(ss / D + eps)
    dt = X.dtype.element_ty
    # the normalized row is materialized in the input dtype before the rotation, and each rotation product is rounded
    # like the eager reference (and the HIP kernel) does
    n0 = (x0 * inv[:, None] * tl.load(W + j, mask=jm, other=0.0).to(tl.float32)[None, :]).to(dt).to(tl.float32)
    n1 = (x1 * inv[:, None] * tl.load(W + HALF + j, mask=jm, other=0.0).to(tl.float32)[None, :]).to(dt).to(tl.float32)
    o0 = (n0 * f00).to(dt).to(tl.float32) + n1 * f01
    o1 = (n1 * f11).to(dt).to(tl.float32) + n0 * f10
    tl.store(base + j[None, :], o0.to(dt), mask=m2)
    tl.store(base + HALF + j[None, :], o1.to(dt), mask=m2)
    if REST > 0:
        wr = tl.load(W + 2 * HALF + r, mask=rm, other=0.0).to(tl.float32)
        tl.store(base + 2 * HALF + r[None, :], (xr * inv[:, None] * wr[None, :]).to(dt), mask=mr)


@triton.jit
def _rms_rope_kernel(Q, K, F, WQ, WK, S, H, s_qb, s_qt, s_qh, s_kb, s_kt, s_kh, s_fb, s_ft, s_fj, eps,
                     D: tl.constexpr, HALF: tl.constexpr, HALF_P: tl.constexpr, REST: tl.constexpr, REST_P: tl.constexpr,
                     BLOCK_T: tl.constexpr, HEADS: tl.constexpr):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    h0 = tl.program_id(1) * HEADS
    b = tl.program_id(2)
    tm = t < S
    j = tl.arange(0, HALF_P)
    jm = j < HALF
    r = tl.arange(0, REST_P)
    rm = r < REST
    fb = F + b * s_fb + t[:, None] * s_ft + j[None, :] * s_fj
    fm = tm[:, None] & jm[None, :]
    f00 = tl.load(fb, mask=fm, other=0.0).to(tl.float32)
    f01 = tl.load(fb + 1, mask=fm, other=0.0).to(tl.float32)
    f10 = tl.load(fb + 2, mask=fm, other=0.0).to(tl.float32)
    f11 = tl.load(fb + 3, mask=fm, other=0.0).to(tl.float32)
    for hh in range(HEADS):
        h = h0 + hh
        if h < H:
            _rms_rope_row(Q, WQ, b, h, t, tm, j, jm, r, rm, s_qb, s_qt, s_qh, f00, f01, f10, f11, eps, D, HALF, REST)
            _rms_rope_row(K, WK, b, h, t, tm, j, jm, r, rm, s_kb, s_kt, s_kh, f00, f01, f10, f11, eps, D, HALF, REST)


def rms_rope_split_half_(q, k, freqs, q_scale, k_scale, epsilon, rot):
    """q, k [B, S, H, D] in place; freqs [B or 1, S, 1, rot/2, 2, 2] with contiguous 2x2 matrices."""
    B, S, H, D = q.shape
    half = rot // 2
    rest = D - rot
    heads = 4 if H >= 48 else 8  # measured on gfx1201: H3 DiT (56 heads) and its video VAE (32 heads)
    grid = (triton.cdiv(S, 16), triton.cdiv(H, heads), B)
    _rms_rope_kernel[grid](q, k, freqs, q_scale, k_scale, S, H,
                           q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                           freqs.stride(0) if freqs.shape[0] > 1 else 0, freqs.stride(1), freqs.stride(3), epsilon,
                           D=D, HALF=half, HALF_P=triton.next_power_of_2(half), REST=rest,
                           REST_P=max(triton.next_power_of_2(rest), 1), BLOCK_T=16, HEADS=heads, num_warps=4)
    return q, k
