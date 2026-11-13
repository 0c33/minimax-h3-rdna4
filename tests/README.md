# MiniMax-H3 tests

`h3_test.py` runs the two saved H3 workflows (`user/default/workflows/MiniMax-H3 bf16 - *.json`) end to end through the running ComfyUI, the same way the UI does, and checks the result. Run it with ComfyUI's Python (it needs PyAV, numpy and Pillow):

```
cd ComfyUI  # your ComfyUI folder, with this folder copied to user/h3_tests
.venv/bin/python user/h3_tests/h3_test.py t2v                        # text to video, 5 s, 20 steps, remote prompt writer
.venv/bin/python user/h3_tests/h3_test.py i2v --image my_photo.png   # image to video, image from ComfyUI/input
.venv/bin/python user/h3_tests/h3_test.py t2v --idea "a fox jumps over a frozen creek" --duration 8
.venv/bin/python user/h3_tests/h3_test.py t2v --turbo                # 8-step draft mode
.venv/bin/python user/h3_tests/h3_test.py t2v --steps 3              # quick smoke test (blurry video, checks speed and memory)
```

Options: `--duration S`, `--steps N`, `--turbo`, `--no-writer` (send the idea box to H3 as written), `--idea TEXT`, `--image NAME`, `--seed N`, `--tag NAME`.

It passes when:

- the job finished and the log has no error, traceback, NaN or out-of-memory
- the video is 24 fps with the expected frame count, with no black, flat or frozen frames
- the soundtrack covers the whole clip and is not silent
- the prompt H3 received is in H3's format (with the writer on), and for image to video starts with the `<Picture 1>` reference line
- RAM never dropped below 1 GB available

`compare_videos.py A.mp4 B.mp4 [sheet.jpg]` measures how close two renders are (PSNR/SSIM per second, plus an optional side-by-side sheet with the amplified difference).

Results go to `user/h3_tests/runs/`: `<tag>.json` (timings per phase and per step, peak VRAM, lowest free RAM, swap, disk reads, drive temperature, the final prompt), `<tag>.log` (ComfyUI's log for the run) and `<tag>_frames.jpg` (8 frames from the video). The video itself is saved as `output/video/test_<tag>_00001_.mp4`.
