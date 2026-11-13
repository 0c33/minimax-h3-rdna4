"""Temporary ComfyUI custom node (copy into custom_nodes/, remove after use): saves real attention inputs for
real_compare.py / skip_oracle.py.

  H3_CAPTURE_DIR=/path H3_CAPTURE_CALLS=102,125,147 python main.py ...

Counts only attention calls with at least 30k tokens (H3: 50 per sampling step, so 125 = step 2, block 25) and saves
8 evenly spaced heads of q/k/v for the listed calls."""
import logging
import os

import torch

import comfy.ops

DIR = os.environ.get("H3_CAPTURE_DIR")
CALLS = {int(c) for c in os.environ.get("H3_CAPTURE_CALLS", "").split(",") if c}
_inner = comfy.ops.scaled_dot_product_attention
_count = [0]


def scaled_dot_product_attention(q, k, v, *args, **kwargs):
    if q.ndim == 4 and q.shape[2] >= 30000:
        i = _count[0]
        _count[0] += 1
        if i in CALLS:
            heads = torch.linspace(0, q.shape[1] - 1, 8).long().tolist()
            torch.save({"q": q[:, heads].cpu().clone(), "k": k[:, heads].cpu().clone(), "v": v[:, heads].cpu().clone(),
                        "heads": heads, "call": i}, os.path.join(DIR, f"qkv_call{i}.pt"))
            logging.info("attn_capture: saved call %d %s", i, tuple(q.shape))
    return _inner(q, k, v, *args, **kwargs)


if DIR:
    os.makedirs(DIR, exist_ok=True)
    comfy.ops.scaled_dot_product_attention = scaled_dot_product_attention

NODE_CLASS_MAPPINGS = {}
