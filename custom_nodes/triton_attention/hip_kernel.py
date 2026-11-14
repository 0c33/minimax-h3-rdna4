"""Loads hip_kernel.hip (hand-written RDNA4 flash attention: bf16/fp16, head dim 64/128) through PyTorch's own HIP
runtime and launches it.

The code object is built once per source version and GPU architecture into build/ (hipcc from HIPCC, ROCM_PATH or
/opt/rocm); if that or loading fails, load() returns None and the Triton kernel is used instead.
"""
import ctypes
import hashlib
import logging
import math
import os
import shutil
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "hip_kernel.hip")
BM = 128  # queries per block (8 waves of 16)


def _hipcc():
    for c in (os.environ.get("HIPCC"), os.path.join(os.environ.get("ROCM_PATH", ""), "bin", "hipcc"),
              "/opt/rocm/core/bin/hipcc", "/opt/rocm/bin/hipcc", shutil.which("hipcc")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _code_object(arch):
    src = open(SOURCE, "rb").read()
    path = os.path.join(HERE, "build", f"fa_{arch}_{hashlib.sha256(src).hexdigest()[:12]}.hsaco")
    if os.path.exists(path):
        return path
    hipcc = _hipcc()
    if hipcc is None:
        logging.info("triton_attention: no hipcc found, the HIP kernel is not built")
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    r = subprocess.run([hipcc, f"--offload-arch={arch}", "-O3", "--genco", "-o", tmp, SOURCE], capture_output=True, text=True)
    if r.returncode != 0:
        logging.warning("triton_attention: building the HIP kernel failed:\n%s", r.stderr[-2000:])
        return None
    os.replace(tmp, path)
    return path


class _Kernel:
    def __init__(self, path):
        lib = next(l.split()[-1] for l in open("/proc/self/maps") if "libamdhip64" in l)
        self.hip = ctypes.CDLL(lib)
        self.hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.hip.hipModuleLaunchKernel.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 7
                                                   + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
        data = open(path, "rb").read()
        self.image = ctypes.create_string_buffer(data, len(data))
        self.module = ctypes.c_void_p()
        err = self.hip.hipModuleLoadData(ctypes.byref(self.module), ctypes.cast(self.image, ctypes.c_void_p))
        self.fns = {}
        for dtype, dname in ((torch.bfloat16, "bf16"), (torch.float16, "f16")):
            for d in (64, 128):
                fn = ctypes.c_void_p()
                if err == 0:
                    err = self.hip.hipModuleGetFunction(ctypes.byref(fn), self.module, f"fa_{dname}_d{d}_n32".encode())
                self.fns[(dtype, d)] = fn
        if err != 0:
            raise RuntimeError(f"loading {path} failed with HIP error {err}")

    def __call__(self, q, k, v, scale):
        B, H, NQ, D = q.shape
        if q.stride(1) < q.stride(2):
            out = torch.empty((B, NQ, H, D), dtype=q.dtype, device=q.device).transpose(1, 2)
        else:
            out = torch.empty((B, H, NQ, D), dtype=q.dtype, device=q.device)
        L, I = ctypes.c_int64, ctypes.c_int32
        args = [ctypes.c_void_p(t.data_ptr()) for t in (q, k, v, out)]
        for t in (q, k, v, out):
            args += [L(t.stride(0)), L(t.stride(1)), L(t.stride(2))]
        args += [I(H), I(NQ), I(k.shape[2]), ctypes.c_float((D ** -0.5 if scale is None else scale) * math.log2(math.e))]
        ptrs = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        err = self.hip.hipModuleLaunchKernel(self.fns[(q.dtype, D)], (NQ + BM - 1) // BM, B * H, 1, BM * 2, 1, 1, 0,
                                             torch.cuda.current_stream(q.device).cuda_stream, ptrs, None)
        if err != 0:
            raise RuntimeError(f"hipModuleLaunchKernel failed with HIP error {err}")
        return out


def covered(q, k, v):
    """bf16/fp16, head dim 64 or 128, 16-byte aligned rows (the kernel loads 8 elements at a time)."""
    if q.dtype not in (torch.bfloat16, torch.float16) or q.shape[-1] not in (64, 128) or q.shape[0] * q.shape[1] > 65535:
        return False
    return all(t.data_ptr() % 16 == 0 and t.stride(0) % 8 == 0 and t.stride(1) % 8 == 0 and t.stride(2) % 8 == 0 for t in (q, k, v, ))


def load(device):
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    try:
        path = _code_object(arch)
        return _Kernel(path) if path else None
    except (OSError, RuntimeError, StopIteration) as e:
        logging.warning("triton_attention: HIP kernel unavailable (%s), using the Triton kernel", e)
        return None
