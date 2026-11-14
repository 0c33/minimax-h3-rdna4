"""Background read-ahead for models that stream their weights from disk (dynamic VRAM, fast_disk).

When a model is larger than VRAM + pinned RAM (MiniMax H3: 38 GB model, 48 GB text encoder), ComfyUI reads the
missing weights of each block from disk right before the block runs. That read cannot start until the previous
block's GPU work has finished, so the disk and the GPU take turns instead of working at the same time.

Here one background thread reads the weights of the coming blocks into the page cache while the current block
computes, and a block only starts once its weights are in RAM, so ComfyUI's own reads become memory copies and the
disk sees a single sequential reader (two readers at once are slower than one on ntfs-3g).

It only reads files: it never changes a weight, a tensor or what ComfyUI computes, so outputs are bit-identical
with and without it. Weights already in VRAM or pinned RAM are skipped, and pages are dropped from the page cache
once their block has run, so the read-ahead does not push other programs into swap.

During the sampler's last step, once that step's weights are all read, it also reads the weights of the loaded VAEs
(smallest first, as far as free RAM allows with 4 GB to spare), so the decode that follows does not wait for the disk.

Environment:
  COMFY_READAHEAD_MB   how far ahead of the running block to read, in MB (default 2048, 0 disables)
"""
import collections
import logging
import os
import threading
import time
import weakref

import torch

import comfy.model_prefetch
import comfy.sd

WINDOW = int(float(os.environ.get("COMFY_READAHEAD_MB", "2048")) * 2**20)
CHUNK = 8 << 20  # ntfs-3g peaks with 1-16 MB reads here (0.9-1.3 GB/s); 64 MB reads are slower
VBAR_PAGE = 32 << 20  # comfy-aimdo VBAR_PAGE_SIZE
WAIT_LIMIT = 120  # seconds a block waits for its weights before ComfyUI reads them itself
RAM_SPARE = 4 << 30  # the VAE prefetch leaves this much RAM available
UNTIMED = 1 << 62  # tick of prefetch reads no block waits for
_vaes = weakref.WeakSet()


def _disk_ranges(block, residency, stats):
    """File ranges (fd, offset, size) of the block's weights that are neither in VRAM nor pinned in RAM."""
    out = []
    for s in block.modules():
        v = getattr(s, "_v", None)
        if v is None:
            continue
        pins = s.__dict__.get("_pins") or {}
        if any(p.get("pin") is not None for p in pins.values()):
            stats["pinned"] += 1
            continue
        vbar, addr, size = v
        flags = residency.get(id(vbar))
        if flags is None:
            flags = residency[id(vbar)] = vbar.get_residency()
        first = (addr - vbar.base_addr) // VBAR_PAGE
        last = (addr - vbar.base_addr + max(size, 1) - 1) // VBAR_PAGE
        if last < len(flags) and all(f & 1 for f in flags[first:last + 1]):
            stats["resident"] += 1
            continue
        for t in (s.weight, getattr(s, "bias", None)):
            # quantized weights are tensor subclasses without a plain storage; ComfyUI reads those itself
            info = getattr(t.untyped_storage(), "_comfy_tensor_file_slice", None) if type(t) in (torch.Tensor, torch.nn.Parameter) else None
            if info is not None and isinstance(info.file_ref, int) and info.size > 0:
                out.append((info.file_ref, info.offset, info.size))
    return out


def _file_ranges(model, residency, stats):
    """File ranges of a not yet used model's weights: modules prepared for dynamic VRAM go through their residency,
    the rest by their own tensors."""
    out = _disk_ranges(model, residency, stats)
    for m in model.modules():
        if getattr(m, "_v", None) is not None:
            continue
        for t in list(m.parameters(recurse=False)) + list(m.buffers(recurse=False)):
            info = getattr(t.untyped_storage(), "_comfy_tensor_file_slice", None) if type(t) in (torch.Tensor, torch.nn.Parameter) else None
            if info is not None and isinstance(info.file_ref, int) and info.size > 0:
                out.append((info.file_ref, info.offset, info.size))
    return out


def _ram_available():
    with open("/proc/meminfo") as f:
        return next(int(l.split()[1]) * 1024 for l in f if l.startswith("MemAvailable:"))


class ReadAhead:
    def __init__(self):
        self.cv = threading.Condition()
        self.todo = collections.deque()  # (tick the block runs at, fd, offset, size)
        self.pending = collections.Counter()  # tick -> ranges the reader has not finished yet
        self.tick = 0  # +1 per block that starts running, across all passes
        self.blocks, self.index, self.repeated, self.last = [], {}, False, False
        self.vae_queued = False
        self.planned = {}  # tick -> ranges read ahead for the block running at that tick
        self.failed = False
        self.passes = 0
        self.stats = collections.Counter()
        threading.Thread(target=self._run, name="disk_readahead", daemon=True).start()

    # --- main thread ---

    def start_pass(self, blocks, last=False):
        repeated = len(blocks) == len(self.blocks) and all(a is b for a, b in zip(blocks, self.blocks))
        self.last = last
        if not repeated:
            if self.passes:
                self.report()
            self._drop(self.tick + 1)
            with self.cv:
                self.todo.clear()
                self.pending.clear()
            self.planned = {}
            self.vae_queued = False
        self.passes = self.passes + 1 if repeated else 1
        self.blocks, self.index, self.repeated = blocks, {id(b): i for i, b in enumerate(blocks)}, repeated

    def on_block(self, module):
        i = self.index.get(id(module))
        if i is None:
            return
        self.tick += 1
        self._drop(self.tick - 1)
        n = len(self.blocks)
        # sampling steps repeat the same blocks, so their read-ahead continues into the next pass (not after the last)
        horizon = 2 * n - i if self.repeated and not self.last else n - i
        ahead, residency, todo = 0, {}, []
        for d in range(horizon):
            t = self.tick + d
            ranges = self.planned.get(t)
            if ranges is None:
                ranges = self.planned[t] = _disk_ranges(self.blocks[(i + d) % n], residency, self.stats)
                todo += [(t, *r) for r in ranges]
            ahead += sum(r[2] for r in ranges)
            if ahead >= WINDOW:
                break
        else:
            if self.last and not self.vae_queued:  # the last step is fully planned: the VAEs come next
                self.vae_queued = True
                todo += [(UNTIMED, *r) for r in self._vae_ranges()]
        with self.cv:
            for t, *_ in todo:
                if t != UNTIMED:
                    self.pending[t] += 1
            self.todo.extend(todo)
            self.cv.notify_all()
            if self.pending[self.tick]:
                t0 = time.monotonic()
                self.cv.wait_for(lambda: not self.pending[self.tick], timeout=WAIT_LIMIT)
                self.stats["wait_ms"] += int((time.monotonic() - t0) * 1000)

    def _vae_ranges(self):
        budget = _ram_available() - RAM_SPARE
        out, stats = [], collections.Counter()
        # smallest VAE first; a VAE that does not fit is read from its start (its first layers run first)
        for ranges in sorted((_file_ranges(v.first_stage_model, {}, stats) for v in list(_vaes)), key=lambda r: sum(x[2] for x in r)):
            for r in ranges:
                if r[2] > budget:
                    break
                out.append(r)
                budget -= r[2]
        self.stats["vae"] += sum(x[2] for x in out)
        logging.info("disk_readahead: reading %.1f GB of VAE weights during the last step", sum(x[2] for x in out) / 1e9)
        return out

    def report(self):
        st = self.stats
        if not st["read"] and not st["wait_ms"]:
            return
        logging.info("disk_readahead: %d blocks x %d passes, read %.1f GB ahead, waited %.1f s for the disk, skipped weights: %d in VRAM, %d pinned%s",
                     len(self.blocks), self.passes, st["read"] / 1e9, st["wait_ms"] / 1000, st["resident"], st["pinned"],
                     f", VAE weights queued: {st['vae'] / 1e9:.1f} GB" if st["vae"] else "")
        self.stats = collections.Counter()

    def _drop(self, before_tick):
        """Drops the page cache of read-ahead blocks that already ran (ran at a tick before before_tick)."""
        for t in [t for t in self.planned if t < before_tick]:
            for fd, off, size in self.planned.pop(t):
                try:
                    os.posix_fadvise(fd, off, size, os.POSIX_FADV_DONTNEED)
                except OSError:  # the model was unloaded and its file closed
                    pass

    # --- reader thread ---

    def _run(self):
        buf = memoryview(bytearray(CHUNK))
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.todo)
                t, fd, off, size = self.todo.popleft()
            end = off + size
            try:
                while off < end and t >= self.tick:  # a block that already ran was read by ComfyUI itself
                    n = os.preadv(fd, [buf[:min(CHUNK, end - off)]], off)
                    if n <= 0:
                        break
                    off += n
                    self.stats["read"] += n
            except OSError:
                pass  # the model was unloaded and its file closed
            with self.cv:
                if self.pending[t]:
                    self.pending[t] -= 1
                if not self.pending[t]:
                    del self.pending[t]
                    self.cv.notify_all()


_state = ReadAhead() if WINDOW > 0 else None
_orig_make = comfy.model_prefetch.make_prefetch_queue
_orig_pop = comfy.model_prefetch.prefetch_queue_pop
_orig_vae_init = comfy.sd.VAE.__init__


def vae_init(self, *args, **kwargs):
    _orig_vae_init(self, *args, **kwargs)
    _vaes.add(self)


def _guard(fn, *args):
    if _state.failed:
        return
    try:
        fn(*args)
    except Exception:
        _state.failed = True
        logging.exception("disk_readahead: disabled after an error (inference is unaffected)")


def _last_pass(transformer_options):
    """True on the sampler's last model call (the step that ends at sigma 0), so nothing is read for a next pass."""
    sigmas, schedule = transformer_options.get("sigmas"), transformer_options.get("sample_sigmas")
    if not isinstance(sigmas, torch.Tensor) or not isinstance(schedule, torch.Tensor) or schedule.numel() < 2:
        return False
    return float(sigmas.max()) <= float(schedule[-2])


def make_prefetch_queue(queue, device, transformer_options):
    # the block list arrives here even when ComfyUI then runs without a queue (text encoders)
    _guard(_state.start_pass, list(queue), _last_pass(transformer_options))
    return _orig_make(queue, device, transformer_options)


def prefetch_queue_pop(queue, device, module, *args, **kwargs):
    if module is not None:
        _guard(_state.on_block, module)
    out = _orig_pop(queue, device, module, *args, **kwargs)
    if module is None and _state.passes == 1:
        _guard(_state.report)
    return out


if _state is not None:
    comfy.model_prefetch.make_prefetch_queue = make_prefetch_queue
    comfy.model_prefetch.prefetch_queue_pop = prefetch_queue_pop
    comfy.sd.VAE.__init__ = vae_init
    logging.info("disk_readahead: reading up to %d MB ahead of the running block for disk-streamed models.", WINDOW >> 20)

NODE_CLASS_MAPPINGS = {}
