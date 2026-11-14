"""Faster fused RMSNorm + partial split-half RoPE (comfy_kitchen.rms_rope_split_half_) for RDNA4.

MiniMax H3 normalizes and rotates q and k in every attention layer: 50 times per sampling step in the diffusion model
and 2016 times per 5 s video decode. comfy_kitchen's HIP kernel for this runs at ~150 GB/s on an RX 9070 XT (14.3 ms
per DiT layer); this Triton kernel does the same math at ~450 GB/s (4.7 ms), about 0.5 s per step.

Same rounding as the reference: the normalized row is rounded to the input dtype before the rotation and each
rotation product is rounded, as the eager implementation and the HIP kernel do. Measured against the HIP kernel on
H3 shapes it agrees in 99.999% of values; the rest differ by one rounding step of the dtype.

Only the in-place inference op is replaced, and only for the layout it was verified on (q/k [B, S, H, D] with the
head dim contiguous, bf16/fp16, rotation matrices in the input dtype); anything else goes to comfy_kitchen.
Set COMFY_TRITON_RMS_ROPE=0 to disable.
"""
import logging
import os

import torch

import comfy_kitchen
from comfy_kitchen._rope_utils import check_rope_inplace

from .kernel import rms_rope_split_half_ as _triton_rms_rope_

_orig = comfy_kitchen.rms_rope_split_half_


def _covered(q, k, freqs, q_scale, k_scale, rot):
    if torch.is_grad_enabled() or not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if q.dim() != 4 or k.shape != q.shape or k.dtype != q.dtype or q.stride(3) != 1 or k.stride(3) != 1:
        return False
    B, S, H, D = q.shape
    if rot % 2 or not 0 < rot <= D:
        return False
    if (freqs.dtype != q.dtype or freqs.device != q.device or freqs.dim() != 6 or freqs.shape[0] not in (1, B)
            or tuple(freqs.shape[1:]) != (S, 1, rot // 2, 2, 2) or freqs.stride(5) != 1 or freqs.stride(4) != 2):
        return False
    return all(w.dim() == 1 and w.numel() == D and w.is_contiguous() and w.device == q.device for w in (q_scale, k_scale))


def rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    k_scale = q_scale if k_scale is None else k_scale
    rot = rot_dim or q.shape[-1]
    if not _covered(q, k, freqs_cis, q_scale, k_scale, rot):
        return _orig(q, k, freqs_cis, q_scale, k_scale, epsilon=epsilon, rot_dim=rot_dim)
    check_rope_inplace(q, k, readonly=(freqs_cis, q_scale, k_scale))
    return _triton_rms_rope_(q, k, freqs_cis, q_scale, k_scale, epsilon, rot)


if os.environ.get("COMFY_TRITON_RMS_ROPE", "1") != "0" and torch.cuda.is_available() and torch.version.hip is not None:
    comfy_kitchen.rms_rope_split_half_ = rms_rope_split_half_
    logging.info("triton_rms_rope: fused RMSNorm + RoPE runs as a Triton kernel (in-place q/k layouts).")

NODE_CLASS_MAPPINGS = {}
