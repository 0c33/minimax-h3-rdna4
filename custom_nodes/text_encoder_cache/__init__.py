"""Remembers text-encoder results on disk, so a prompt that was encoded once is never encoded again.

MiniMax H3's text encoder is 48 GB and streams from disk: about 85 s per encode on a 32 GB RAM machine. ComfyUI only
reuses a result while the node's inputs are unchanged and the server keeps running, so changing the clip length
(H3 encodes inside its conditioning node) or restarting ComfyUI encodes the same prompt again.

The key is a SHA-256 of everything the encoder reads: the tokens (including any image tensors), the encode options
and the identity (path, size, mtime) of the encoder's weight files. A hit returns the stored result unchanged. Encoders
with LoRA weight patches or hooks are never cached.

Environment:
  COMFY_TE_CACHE_DIR    where results go (default: <ComfyUI>/user/text_encoder_cache; "" disables)
  COMFY_TE_CACHE_GB     size limit, oldest entries go first (default 2)
"""
import hashlib
import logging
import os
import time

import torch

import comfy.sd
import folder_paths

CACHE_DIR = os.environ.get("COMFY_TE_CACHE_DIR", os.path.join(folder_paths.get_user_directory(), "text_encoder_cache"))
LIMIT = int(float(os.environ.get("COMFY_TE_CACHE_GB", "2")) * 2**30)
MIN_ENCODER_BYTES = 4 << 30  # only big encoders, where an encode costs real time

_encode = comfy.sd.CLIP.encode_from_tokens_scheduled


def _feed(h, obj):
    if isinstance(obj, torch.Tensor):
        t = obj.detach().to("cpu").contiguous()
        h.update(f"T{t.dtype}{tuple(t.shape)}".encode())
        if t.numel():
            h.update(t.view(torch.uint8).numpy().tobytes())
    elif isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=str):
            _feed(h, str(k))
            _feed(h, obj[k])
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for x in obj:
            _feed(h, x)
        h.update(b"]")
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        h.update(repr(obj).encode())
    else:
        raise TypeError(type(obj).__name__)  # unknown objects could hide state: no caching


def _plain(obj):
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _plain(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return all(_plain(x) for x in obj)
    return obj is None or isinstance(obj, (torch.Tensor, bool, int, float, str))


def _encoder_files(clip):
    paths = set()
    for t in clip.cond_stage_model.state_dict().values():
        path = getattr(t.untyped_storage(), "_comfy_source_path", None) if type(t) in (torch.Tensor, torch.nn.Parameter) else None
        if path is not None:
            paths.add(path)
    return sorted(paths)


def _key(clip, tokens, unprojected, add_dict):
    if clip.patcher.forced_hooks is not None or clip.patcher.patches:
        return None
    files = _encoder_files(clip)
    if not files or sum(os.path.getsize(p) for p in files) < MIN_ENCODER_BYTES:
        return None
    h = hashlib.sha256()
    try:
        _feed(h, [type(clip.cond_stage_model).__name__, [(p, os.path.getsize(p), os.path.getmtime(p)) for p in files],
                  clip.layer_idx, unprojected, add_dict, tokens])
    except TypeError:
        return None
    return h.hexdigest()


def _prune():
    entries = sorted((e for e in os.scandir(CACHE_DIR) if e.name.endswith(".pt")), key=lambda e: e.stat().st_mtime)
    total = sum(e.stat().st_size for e in entries)
    for e in entries:
        if total <= LIMIT:
            break
        total -= e.stat().st_size
        os.remove(e.path)


def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
    key = _key(self, tokens, unprojected, add_dict)
    path = os.path.join(CACHE_DIR, key + ".pt") if key else None
    if path and os.path.exists(path):
        try:
            out = torch.load(path, map_location="cpu", weights_only=True)
            os.utime(path)
            logging.info("text_encoder_cache: reused the stored encoding (%s...)", key[:12])
            return out
        except Exception as e:  # truncated or foreign file: encode again and overwrite it
            logging.warning("text_encoder_cache: ignoring unreadable %s (%s)", path, e)
    t0 = time.monotonic()
    out = _encode(self, tokens, unprojected=unprojected, add_dict=add_dict, show_pbar=show_pbar)
    if path and not _plain(out):
        logging.info("text_encoder_cache: this encoder's output holds non-tensor objects, not cached")
    elif path:
        tmp = path + ".tmp"
        torch.save([[c.cpu(), {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in d.items()}] for c, d in out], tmp)
        os.replace(tmp, path)
        _prune()
        logging.info("text_encoder_cache: stored the encoding (%.0f s saved next time)", time.monotonic() - t0)
    return out


if CACHE_DIR:
    os.makedirs(CACHE_DIR, exist_ok=True)
    comfy.sd.CLIP.encode_from_tokens_scheduled = encode_from_tokens_scheduled

NODE_CLASS_MAPPINGS = {}
