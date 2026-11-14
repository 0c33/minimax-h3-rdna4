"""One-pass versions of two elementwise steps that run every layer of MiniMax H3 (and SwiGLU models in general).

- SwiGLU, silu(gate) * up: ComfyUI's eager path (comfy.ops INPUT_ACT_EAGER["swiglu"]) runs silu and the product as two
  kernels, 5.5 GB of memory traffic per H3 MLP at 5 s; one Triton kernel does it in 3.3 GB.
- AdaLN modulation in the H3 DiT, h.mul_(1 + scale).add_(shift): two in-place passes over the hidden state, twice per
  block; one kernel does both.

Both round exactly where the eager ops do (each op computes in fp32 and rounds to the tensor dtype), so the results
match the eager ones. Only inference calls on CUDA bf16/fp16 tensors with contiguous rows are replaced; anything else
runs the original code. COMFY_FUSED_ELEMENTWISE=0 disables both.
"""
import logging
import os

import torch

import comfy.ldm.minimax.model
import comfy.ops

from .kernels import scale_shift_, swiglu

_swiglu_eager = comfy.ops.INPUT_ACT_EAGER["swiglu"]
_mod_scale_shift = comfy.ldm.minimax.model._mod_scale_shift


def _fast(x):
    return (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and x.stride(-1) == 1
            and not (torch.is_grad_enabled() and x.requires_grad))


def swiglu_act(x):
    if _fast(x) and x.shape[-1] % 2 == 0 and x.numel() and x.reshape(-1, x.shape[-1]).stride(-1) == 1:
        return swiglu(x)
    return _swiglu_eager(x)


def mod_scale_shift(h, shift, scale, segments):
    if not (_fast(h) and h.dim() == 2):
        return _mod_scale_shift(h, shift, scale, segments)
    for a, b, row in segments:
        if isinstance(row, int):
            scale_shift_(h[a:b], (1.0 + scale[row].to(h.dtype)).contiguous(), shift[row].to(h.dtype).contiguous())
        elif isinstance(row, torch.Tensor) and row.dim() == 1 and row.device == h.device and row.numel() == b - a:
            # per-token rows: gathering then adding 1 equals adding 1 then gathering, element by element
            scale_shift_(h[a:b], (1.0 + scale.to(h.dtype)).contiguous(), shift.to(h.dtype).contiguous(), row.to(torch.int64))
        else:
            _mod_scale_shift(h, shift, scale, [(a, b, row)])
    return h


if os.environ.get("COMFY_FUSED_ELEMENTWISE", "1") != "0" and torch.cuda.is_available():
    comfy.ops.INPUT_ACT_EAGER["swiglu"] = swiglu_act
    comfy.ldm.minimax.model._mod_scale_shift = mod_scale_shift
    logging.info("fused_elementwise: SwiGLU and H3 AdaLN modulation run as one-pass Triton kernels.")

NODE_CLASS_MAPPINGS = {}
