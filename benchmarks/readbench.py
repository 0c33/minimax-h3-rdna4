"""readbench.py FILE - read throughput of FILE with pread at several chunk sizes and thread counts.
Each test reads a fresh 2 GiB region and drops it from the page cache first (fadvise DONTNEED)."""
import os, sys, time, threading
path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
size = os.fstat(fd).st_size
REGION = 2 << 30
region_start = [1 << 30]

def bench(chunk, threads):
    start = region_start[0]; region_start[0] += REGION
    if start + REGION > size:
        region_start[0] = 1 << 30; start = region_start[0]; region_start[0] += REGION
    os.posix_fadvise(fd, start, REGION, os.POSIX_FADV_DONTNEED)
    per = REGION // threads
    def work(i):
        buf = bytearray(chunk); mv = memoryview(buf)
        off = start + i * per; end = off + per
        while off < end:
            n = os.preadv(fd, [mv], off)
            off += n
    t0 = time.time()
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    [t.start() for t in ts]; [t.join() for t in ts]
    dt = time.time() - t0
    os.posix_fadvise(fd, start, REGION, os.POSIX_FADV_DONTNEED)
    print(f"chunk {chunk>>10:6d} KiB threads {threads}: {REGION/dt/1e9:5.2f} GB/s", flush=True)

for chunk in (128 << 10, 1 << 20, 16 << 20, 64 << 20):
    bench(chunk, 1)
for threads in (2, 4, 8):
    bench(16 << 20, threads)
