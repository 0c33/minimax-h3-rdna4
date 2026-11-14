"""Faster exact attention for RDNA4 (RX 9070 / 9070 XT): a hand-written HIP flash-attention kernel (bf16/fp16, head
size 64 or 128), with a Triton kernel (head size 128) as the fallback.

On gfx1201 at MiniMax H3's shape (56 heads x 128, 38k tokens) PyTorch's SDPA flash path (AOTriton 0.13) reaches
~72 TFLOPS. The Triton kernel in kernel.py reaches ~97 and the HIP kernel in hip_kernel.hip ~110; at the H3 video
VAE's shape (32 heads x 64, 1.8k tokens) the HIP kernel takes 1.2 ms per call against 1.67 for SDPA. Same math as
FlashAttention 2: fp32 scores and accumulators, online softmax, probabilities rounded to bf16/fp16 before the P@V
product. Against an fp32 reference their error is the same as SDPA's.

It replaces comfy.ops.scaled_dot_product_attention for the calls it covers (no mask, not causal, no dropout, same
number of q and kv heads, head size 64 or 128, fp16/bf16, at least 1024 x 1024 tokens, inference) and passes every
other call to ComfyUI unchanged.

Environment:
  COMFY_TRITON_ATTENTION=0   disables both kernels
  COMFY_HIP_ATTENTION=0      uses the Triton kernel (head size 128 only) instead of the HIP one
Triton caches its compiled kernel in TRITON_CACHE_DIR (default ~/.triton/cache); the HIP code object lives in build/.
"""
import logging
import os

import torch

import comfy.model_management
import comfy.ops

from . import hip_kernel
from .kernel import attention, supported_device

MIN_WORK = 1024 * 1024  # q tokens x kv tokens; from 1024 to 38k tokens the kernels beat SDPA, smaller calls stay on SDPA

_sdpa = comfy.ops.scaled_dot_product_attention
_hip = None


def _covered(q, k, v, args, kwargs):
    return (not args and kwargs.get("attn_mask") is None and not kwargs.get("dropout_p") and not kwargs.get("is_causal")
            and not kwargs.get("enable_gqa") and q.ndim == 4 and q.shape[-1] in (64, 128)
            and q.dtype in (torch.bfloat16, torch.float16) and k.dtype == q.dtype and v.dtype == q.dtype
            and q.shape[:2] == k.shape[:2] == v.shape[:2] and k.shape[2] == v.shape[2]
            and q.shape[2] * k.shape[2] >= MIN_WORK and q.is_cuda
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
            and not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)))


def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    if _covered(q, k, v, args, kwargs):
        if _hip is not None and hip_kernel.covered(q, k, v):
            return _hip(q, k, v, kwargs.get("scale"))
        if q.shape[-1] == 128:
            return attention(q, k, v, kwargs.get("scale"))
    return _sdpa(q, k, v, *args, **kwargs)


device = comfy.model_management.get_torch_device()
if os.environ.get("COMFY_TRITON_ATTENTION", "1") != "0" and supported_device(device):
    if os.environ.get("COMFY_HIP_ATTENTION", "1") != "0":
        _hip = hip_kernel.load(device)
    comfy.ops.scaled_dot_product_attention = scaled_dot_product_attention
    logging.info("triton_attention: exact flash attention for long unmasked sequences: %s",
                 "HIP kernel (head size 64/128)" if _hip is not None else "Triton kernel (head size 128)")

NODE_CLASS_MAPPINGS = {}
