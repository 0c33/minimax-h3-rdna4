"""Keeps long videos (MiniMax H3 up to 15 s) inside 16 GB of VRAM and 32 GB of RAM.

VRAM: the feed-forward layer of each H3 block expands every token to 28672 values at once. For a 15 s clip at
768p (about 108k tokens) that one buffer is 6.2 GB, which does not fit next to the rest of the block on a 16 GB
card ("VRAM grow failed"). For long clips only, the feed-forward runs in slices of 16384 tokens: every token goes
through the same layers, only the size of the matrix products changes. Clips up to about 9 s (65536 tokens) keep
ComfyUI's code path unchanged.

RAM: a video VAE writes its whole decoded clip to RAM as float32 (4.5 GB for 15 s at 768p) while ComfyUI keeps the
weights of the models that just ran pinned in RAM for the next run. Before such a decode, if the clip would leave
less than 4 GB of RAM available, pinned weights of models that are not running are released first. They are read
from disk again the next time they are used.
"""
import logging
import math

import comfy.ldm.minimax.model
import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.system_memory

LONG = 65536  # tokens; a 10 s 768p clip is about 73k
SLICE = 16384
RAM_HEADROOM = 4 << 30

_mlp_forward = comfy.ldm.minimax.model.MLP.forward
_vae_decode = comfy.sd.VAE.decode


def mlp_forward(self, x):
    if x.shape[0] <= LONG:
        return _mlp_forward(self, x)
    first = comfy.ops.linear_input_act(self.fc2, self.fc1(x[:SLICE]), "swiglu")
    out = first.new_empty((x.shape[0], first.shape[1]))
    out[:SLICE] = first
    del first
    for a in range(SLICE, x.shape[0], SLICE):
        out[a:a + SLICE] = comfy.ops.linear_input_act(self.fc2, self.fc1(x[a:a + SLICE]), "swiglu")
    return out


def vae_decode(self, samples_in, vae_options={}):
    if getattr(self.first_stage_model, "comfy_has_chunked_io", False) and samples_in.ndim == 5:
        need = math.prod(self.first_stage_model.decode_output_shape(samples_in.shape)) * comfy.model_management.dtype_size(self.vae_output_dtype())
        shortfall = need + RAM_HEADROOM - comfy.system_memory.virtual_memory_available()
        if shortfall > 0:
            freed = comfy.model_management.free_pins(shortfall)
            logging.info("long_video_memory: the decoded video needs %.1f GB of RAM, released %.1f GB of pinned weights first.", need / 2**30, freed / 2**30)
    return _vae_decode(self, samples_in, vae_options)


comfy.ldm.minimax.model.MLP.forward = mlp_forward
comfy.sd.VAE.decode = vae_decode

NODE_CLASS_MAPPINGS = {}
