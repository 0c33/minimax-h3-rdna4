"""Faster exact attention for RDNA4 (RX 9070 / 9070 XT) through a Triton flash-attention kernel.

On gfx1201, PyTorch's SDPA flash path (AOTriton 0.13) reaches 66-72 TFLOPS (56 heads x 128, 1k to 38k tokens).
The kernel in kernel.py, with block sizes tuned on this card, reaches 85-94 TFLOPS with the same math: bf16/fp16
inputs, fp32 scores and accumulators, online softmax, probabilities rounded to the input dtype before the P@V product,
as in FlashAttention 2. Against an fp32 reference its error is the same as SDPA's.

It replaces comfy.ops.scaled_dot_product_attention for the calls it covers (no mask, not causal, no dropout, same
number of q and kv heads, head size 128, fp16/bf16, at least 1024 x 1024 tokens, inference) and passes every other
call to ComfyUI unchanged.

Environment:
  COMFY_TRITON_ATTENTION=0   disables it
Triton caches the compiled kernel in TRITON_CACHE_DIR (default ~/.triton/cache).
"""
import logging
import os

import torch

import comfy.model_management
import comfy.ops

from .kernel import attention, supported_device

MIN_WORK = 1024 * 1024  # q tokens x kv tokens; the kernel is 25-31% faster than SDPA from 1024 to 38k tokens, smaller calls stay on SDPA

_sdpa = comfy.ops.scaled_dot_product_attention


def _covered(q, k, v, args, kwargs):
    return (not args and kwargs.get("attn_mask") is None and not kwargs.get("dropout_p") and not kwargs.get("is_causal")
            and not kwargs.get("enable_gqa") and q.ndim == 4 and q.shape[-1] == 128
            and q.dtype in (torch.bfloat16, torch.float16) and k.dtype == q.dtype and v.dtype == q.dtype
            and q.shape[:2] == k.shape[:2] == v.shape[:2] and k.shape[2] == v.shape[2]
            and q.shape[2] * k.shape[2] >= MIN_WORK and q.is_cuda
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
            and not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)))


def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    if _covered(q, k, v, args, kwargs):
        return attention(q, k, v, kwargs.get("scale"))
    return _sdpa(q, k, v, *args, **kwargs)


if os.environ.get("COMFY_TRITON_ATTENTION", "1") != "0" and supported_device(comfy.model_management.get_torch_device()):
    comfy.ops.scaled_dot_product_attention = scaled_dot_product_attention
    logging.info("triton_attention: exact Triton flash attention for long unmasked sequences (head size 128).")

NODE_CLASS_MAPPINGS = {}
