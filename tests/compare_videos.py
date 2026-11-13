"""compare_videos.py A.mp4 B.mp4 [sheet.jpg] - how close two renders are, frame by frame.

Prints PSNR (dB, higher = closer; above ~40 dB differences are invisible) and SSIM (1.0 = identical) per second of
video, and optionally writes a sheet with A on top, B in the middle and the amplified difference at the bottom.
"""
import sys

import av
import numpy as np
from PIL import Image


def frames(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def ssim(a, b):
    """Mean SSIM of two grayscale float images (8x8 box windows)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    h, w = (a.shape[0] // 8) * 8, (a.shape[1] // 8) * 8
    a = a[:h, :w].reshape(h // 8, 8, w // 8, 8)
    b = b[:h, :w].reshape(h // 8, 8, w // 8, 8)
    ma, mb = a.mean((1, 3)), b.mean((1, 3))
    va, vb = a.var((1, 3)), b.var((1, 3))
    cov = ((a - ma[:, None, :, None]) * (b - mb[:, None, :, None])).mean((1, 3))
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def main():
    fa, fb = frames(sys.argv[1]), frames(sys.argv[2])
    n = min(len(fa), len(fb))
    psnr, ss = [], []
    for x, y in zip(fa[:n], fb[:n]):
        mse = np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2)
        psnr.append(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse))
        ss.append(ssim(x.astype(np.float32).mean(-1), y.astype(np.float32).mean(-1)))
    print(f"{n} frames (A has {len(fa)}, B has {len(fb)})")
    for s in range(0, n, 24):
        print(f"  second {s // 24}: PSNR {np.mean(psnr[s:s + 24]):5.1f} dB  SSIM {np.mean(ss[s:s + 24]):.4f}")
    print(f"  whole clip: PSNR {np.mean(psnr):5.1f} dB (worst frame {np.min(psnr):.1f})  SSIM {np.mean(ss):.4f} (worst {np.min(ss):.4f})")
    if len(sys.argv) > 3:
        pick = np.linspace(0, n - 1, 6).astype(int)
        h, w = fa[0].shape[:2]
        tw, th = w // 4, h // 4
        sheet = Image.new("RGB", (tw * 6, th * 3))
        for i, j in enumerate(pick):
            diff = np.clip(np.abs(fa[j].astype(np.int16) - fb[j].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            for row, img in enumerate((fa[j], fb[j], diff)):
                sheet.paste(Image.fromarray(img).resize((tw, th), Image.LANCZOS), (i * tw, row * th))
        sheet.save(sys.argv[3], quality=88)


if __name__ == "__main__":
    main()
