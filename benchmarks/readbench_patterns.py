"""readbench_patterns.py FILE - buffered preadv chunk sizes and reader counts through ntfs-3g, fresh region of the 38 GB H3 file each test."""
import mmap, os, threading, time
import sys
F = sys.argv[1]  # a large model file on the drive to test (needs ~30 GB, read in fresh regions)
GB = 1 << 30
TEST = 3 * GB
region = [1 * GB]
def take():
    off = region[0]; region[0] += TEST + GB // 2
    assert region[0] < os.path.getsize(F)
    return off
def worker(fd, a, b, chunk, drop):
    buf = mmap.mmap(-1, chunk); off = a
    while off < b:
        n = os.preadv(fd, [memoryview(buf)[:min(chunk, b - off)]], off)
        if drop: os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
        off += n
def test(label, chunk, readers=1, drop=True):
    start = take(); fd = os.open(F, os.O_RDONLY); per = TEST // readers
    ths = [threading.Thread(target=worker, args=(fd, start + i * per, start + (i + 1) * per, chunk, drop)) for i in range(readers)]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; dt = time.time() - t0; os.close(fd)
    print(f"{label:36s} {TEST / dt / 1e9:5.2f} GB/s", flush=True)
for rep in range(2):
    test("8 MB", 8 << 20)
    test("32 MB", 32 << 20)
    test("128 MB", 128 << 20)
test("32 MB, no DONTNEED drop", 32 << 20, drop=False)
test("32 MB x 2 readers", 32 << 20, 2)
