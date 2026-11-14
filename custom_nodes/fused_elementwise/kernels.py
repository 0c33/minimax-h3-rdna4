import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 2048


@triton.jit
def _swiglu_kernel(X, OUT, n_rows, half, s_x, s_o, BLOCK: tl.constexpr):
    # out = silu(gate) * up with gate = x[:, :half], up = x[:, half:], rounded like the eager ops:
    # silu in fp32 then to the dtype, the product in fp32 then to the dtype
    row = tl.program_id(0)  # rows on the grid's x axis: up to 2^31 (y stops at 65535)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < half
    base = X + row.to(tl.int64) * s_x
    g = tl.load(base + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(base + half + cols, mask=m, other=0.0).to(tl.float32)
    dt = OUT.dtype.element_ty
    s = (g / (1.0 + libdevice.exp(-g))).to(dt).to(tl.float32)  # ROCm's expf, as ATen's silu uses
    tl.store(OUT + row.to(tl.int64) * s_o + cols, (s * u).to(dt), mask=m)


def swiglu(x):
    """silu(x[..., :h]) * x[..., h:] for a contiguous-last-dim 2D-able tensor."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    half = shape[-1] // 2
    out = torch.empty((x2.shape[0], half), dtype=x.dtype, device=x.device)
    grid = (x2.shape[0], triton.cdiv(half, BLOCK))
    _swiglu_kernel[grid](x2, out, x2.shape[0], half, x2.stride(0), out.stride(0), BLOCK=BLOCK, num_warps=8)
    return out.view(*shape[:-1], half)


@triton.jit
def _scale_shift_kernel(H, A, C, IDX, n_cols, s_h, s_t, BLOCK: tl.constexpr, PER_ROW: tl.constexpr):
    # h = h * a + c in place, rounded like mul_ then add_: round(round(h * a) + c);
    # a and c are row t of their tables: t = IDX[row] with per-token modulation rows, else 0
    row = tl.program_id(0)  # rows on the grid's x axis: up to 2^31 (y stops at 65535)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < n_cols
    ptr = H + row.to(tl.int64) * s_h + cols
    dt = H.dtype.element_ty
    t = tl.load(IDX + row).to(tl.int64) * s_t if PER_ROW else 0
    h = tl.load(ptr, mask=m, other=0.0).to(tl.float32)
    a = tl.load(A + t + cols, mask=m, other=0.0).to(tl.float32)
    c = tl.load(C + t + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(ptr, ((h * a).to(dt).to(tl.float32) + c).to(dt), mask=m)


def scale_shift_(h, a, c, idx=None):
    """h.mul_(a[idx]).add_(c[idx]) in one pass: h 2D with contiguous rows; a, c contiguous [R, width] tables in h's
    dtype and idx an int64 row index per row of h, or a, c of shape [width] for one row for all."""
    grid = (h.shape[0], triton.cdiv(h.shape[1], BLOCK))
    _scale_shift_kernel[grid](h, a, c, h if idx is None else idx, h.shape[1], h.stride(0), a.stride(0) if idx is not None else 0,
                              BLOCK=BLOCK, PER_ROW=idx is not None, num_warps=8)
    return h
