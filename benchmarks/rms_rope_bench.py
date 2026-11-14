"""rmsrope_triton.py - fused per-head RMSNorm + partial split-half RoPE, in place on q and k, as a Triton kernel.

Same contract as comfy_kitchen.rms_rope_split_half_(q, k, freqs, q_scale, k_scale, epsilon, rot_dim):
  q, k   [B, S, H, D] (any strides, last dim contiguous), modified in place
  freqs  [B or 1, S, 1, rot/2, 2, 2] rotation matrices, rot <= D
  RMSNorm spans all D dims (rounded to the input dtype, like the reference), the first rot dims are rotated in split-half
  pairs (i, i + rot/2) and the rest pass through.
ROUND selects where the rotation rounds: 0 = fp32 FMA then one rounding, 1 = like the eager reference (products rounded).
"""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _one(X, W, b, h, t, tm, j, jm, r, rm, s_b, s_t, s_h, f00, f01, f10, f11, eps,
         D: tl.constexpr, HALF: tl.constexpr, REST: tl.constexpr, ROUND: tl.constexpr):
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
    w0 = tl.load(W + j, mask=jm, other=0.0).to(tl.float32)
    w1 = tl.load(W + HALF + j, mask=jm, other=0.0).to(tl.float32)
    dt = X.dtype.element_ty
    n0 = (x0 * inv[:, None] * w0[None, :]).to(dt).to(tl.float32)
    n1 = (x1 * inv[:, None] * w1[None, :]).to(dt).to(tl.float32)
    if ROUND == 0:
        o0 = n0 * f00 + n1 * f01
        o1 = n0 * f10 + n1 * f11
    else:
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
                     BLOCK_T: tl.constexpr, HEADS: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(0)
    h0 = tl.program_id(1) * HEADS
    b = tl.program_id(2)
    t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
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
            _one(Q, WQ, b, h, t, tm, j, jm, r, rm, s_qb, s_qt, s_qh, f00, f01, f10, f11, eps, D, HALF, REST, ROUND)
            _one(K, WK, b, h, t, tm, j, jm, r, rm, s_kb, s_kt, s_kh, f00, f01, f10, f11, eps, D, HALF, REST, ROUND)


def rms_rope_split_half_(q, k, freqs, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0, block_t=16, heads=8, num_warps=4, round_mode=0):
    k_scale = q_scale if k_scale is None else k_scale
    B, S, H, D = q.shape
    rot = rot_dim or D
    half = rot // 2
    rest = D - rot
    assert q.stride(3) == 1 and k.stride(3) == 1 and freqs.stride(-1) == 1 and freqs.stride(-2) == 2
    fbs = freqs.stride(0) if freqs.shape[0] > 1 else 0
    grid = (triton.cdiv(S, block_t), triton.cdiv(H, heads), B)
    _rms_rope_kernel[grid](q, k, freqs, q_scale.contiguous(), k_scale.contiguous(), S, H,
                           q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                           fbs, freqs.stride(1), freqs.stride(3), epsilon,
                           D=D, HALF=half, HALF_P=triton.next_power_of_2(half), REST=rest,
                           REST_P=max(triton.next_power_of_2(rest), 1), BLOCK_T=block_t, HEADS=heads, ROUND=round_mode,
                           num_warps=num_warps)
    return q, k


if __name__ == "__main__":
    import comfy_kitchen

    def bench(fn, iters=20):
        fn(); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters

    for name, S, H, D, rot, dt, layout in [("DiT 5 s", 38196, 56, 128, 96, torch.bfloat16, "h3"),
                                         ("VAE batch", 4 * 1797, 32, 64, 48, torch.float16, "vae")]:
        g = torch.Generator(device="cuda").manual_seed(0)
        if layout == "h3":  # [S, 3*H*D]: q, k, v blocks per token
            qkv = torch.randn(S, 3 * H * D, device="cuda", dtype=dt, generator=g) * 3
            views = lambda buf: (buf[:, :H * D].view(1, S, H, D), buf[:, H * D:2 * H * D].view(1, S, H, D))
        else:  # VAE: [B, L, H, 3*dh] chunked into q, k, v
            qkv = torch.randn(1, S, H * 3 * D, device="cuda", dtype=dt, generator=g) * 3
            views = lambda buf: tuple(buf.view(1, S, H, 3 * D).chunk(3, dim=-1)[:2])
        ang = torch.rand(S, rot // 2, device="cuda", generator=g) * 50
        c, s = ang.cos(), ang.sin()
        freqs = torch.stack([c, -s, s, c], -1).reshape(1, S, 1, rot // 2, 2, 2).to(dt)
        w = torch.rand(D, device="cuda", dtype=dt, generator=g) + 0.5
        ref_buf = qkv.clone()
        comfy_kitchen.rms_rope_split_half_(*views(ref_buf), freqs, w, w, epsilon=1e-5, rot_dim=rot)
        t_hip = bench(lambda: comfy_kitchen.rms_rope_split_half_(*views(qkv.clone()), freqs, w, w, epsilon=1e-5, rot_dim=rot))
        t_clone = bench(lambda: qkv.clone())
        gb = 2 * 2 * S * H * D * qkv.element_size() / 1e9
        print(f"{name}: HIP {(t_hip - t_clone) * 1000:.2f} ms ({gb / (t_hip - t_clone):.0f} GB/s)")
        for rm_ in (1,):
            for bt, hd, nw in [(16, 8, 4), (32, 4, 4), (16, 4, 4), (8, 8, 2), (32, 8, 8), (64, 2, 8)]:
                buf = qkv.clone()
                rms_rope_split_half_(*views(buf), freqs, w, w, epsilon=1e-5, rot_dim=rot, block_t=bt, heads=hd, num_warps=nw, round_mode=rm_)
                mism, mx = 0, 0.0
                for a, b_ in zip(views(buf), views(ref_buf)):
                    for c0 in range(0, S, 4096):
                        x, y = a[:, c0:c0 + 4096], b_[:, c0:c0 + 4096]
                        mism += (x != y).sum().item()
                        mx = max(mx, (x.float() - y.float()).abs().max().item())
                del x, y, a, b_, buf
                t = bench(lambda: rms_rope_split_half_(*views(qkv.clone()), freqs, w, w, epsilon=1e-5, rot_dim=rot,
                                                       block_t=bt, heads=hd, num_warps=nw, round_mode=rm_)) - t_clone
                print(f"   triton round{rm_} T{bt} heads{hd} w{nw}: {t * 1000:.2f} ms ({gb / t:.0f} GB/s) | "
                      f"differs from HIP in {mism} of {2 * S * H * D} q/k values, max {mx:.3g}")
        del qkv, ref_buf, freqs
        torch.cuda.empty_cache()
