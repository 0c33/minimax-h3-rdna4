"""Two readers over the same 2 GiB range (lockstep) vs one reader vs two readers on different ranges."""
import os, sys, threading, time
fd = os.open(sys.argv[1], os.O_RDONLY)
R = 2 << 30
def run(starts, chunks):
    for s in set(starts): os.posix_fadvise(fd, s, R, os.POSIX_FADV_DONTNEED)
    def work(s, c):
        b = memoryview(bytearray(c)); off = s
        while off < s + R: off += os.preadv(fd, [b], off)
    ts = [threading.Thread(target=work, args=(s, c)) for s, c in zip(starts, chunks)]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; dt = time.time() - t0
    for s in set(starts): os.posix_fadvise(fd, s, R, os.POSIX_FADV_DONTNEED)
    return len(set(starts)) * R / dt / 1e9
base = 20 << 30
print("one reader, 8 MB chunks:            %.2f GB/s of file" % run([base], [8 << 20]))
print("two readers same range (8MB + 1MB): %.2f GB/s of file" % run([base + R, base + R], [8 << 20, 1 << 20]))
print("two readers different ranges:       %.2f GB/s of file" % run([base + 2 * R, base + 3 * R], [8 << 20, 1 << 20]))
