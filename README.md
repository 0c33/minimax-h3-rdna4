# MiniMax H3 at full quality on a 16 GB Radeon RX 9070 XT

ComfyUI add-ons, workflows and tests that run [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (text/image to video with sound) at full precision on a single 16 GB AMD card with 32 GB of RAM: bf16 weights, full attention, 1344x768, 20 steps, clips up to 15 s. No fp8, no int8, no sparse attention.

Measured on an RX 9070 XT (gfx1201), Ryzen 7 7800X3D, 32 GB RAM, Linux (CachyOS), ROCm 10.0 PyTorch 2.13, ComfyUI 0.44. The H3 files (38 GB video model, 48 GB text encoder) sit on an NVMe drive formatted NTFS (ntfs-3g, about 1 GB/s).

## Results

5 s clip, 1344x768, 20 steps:

| | Stock ComfyUI | + `disk_readahead` | + `triton_attention` | + fast mode (EasyCache), approximate |
|---|---|---|---|---|
| Each sampling step | 80 s | 43 s | **35.7 s** | 6 of 20 steps skipped |
| Text encoder (new prompt) | 2-2.3 min | 1.3-1.4 min | 1.4 min (0 s for a cached prompt) | same |
| Whole clip | 29.5 min | 18.9 min | **14.9 min** | about 10 min (prompt cached) |

- **10 s:** about 135 s per step before `triton_attention`.
- **15 s:** ran out of VRAM on stock ComfyUI. It works with `long_video_memory`, at about 280 s per step before `triton_attention`.
- **Memory:** VRAM peaks at about 15.6 GB, and at least about 3 GB of RAM stays free.

Other models gain from `triton_attention` where their attention is unmasked: MOSS-Image went from 1.20 to 1.11 s per step (1024x1024, CFG). Qwen Image 2.1 stays at 1.61 it/s: its attention calls carry a mask, which leaves them on SDPA.

## What is in here

| Folder | What it does |
|---|---|
| `custom_nodes/disk_readahead` | When a model is larger than VRAM + pinned RAM, ComfyUI reads each block's weights from disk only after the previous block's GPU work is done, so the disk and the GPU take turns. This reads the next blocks into the page cache in one background thread while the GPU computes. On the 5 s clip the GPU went from 72% to about 100% busy. It only reads files, so outputs are bit-identical with it on or off. |
| `custom_nodes/triton_attention` | Exact flash attention in Triton with block sizes tuned on gfx1201. It runs 85-94 TFLOPS against 66-72 for PyTorch SDPA (AOTriton 0.13) from 1k to 38k tokens, with the same error against an fp32 reference. It only takes over unmasked, non-causal, head-size-128 calls. |
| `custom_nodes/long_video_memory` | For H3 clips over about 9 s, the feed-forward layer runs in 16k-token slices: a 15 s clip otherwise needs one 6.2 GB buffer. Before a long video decode, it frees pinned weights of idle models so RAM does not run out. |
| `custom_nodes/text_encoder_cache` | Keeps text-encoder results on disk (9 MB each, 2 GB max), keyed by a hash of the tokens, image data, options and encoder files. A prompt encoded once skips H3's 85 s encoder after a restart or a duration change. Encoders with LoRA patches or hooks are not cached. |
| `custom_nodes/remote_text_generate` | Runs ComfyUI's Generate Text nodes on an OpenAI-compatible server (`REMOTE_LLM_URL`) instead of a local LLM. The workflows' prompt writer uses it. |
| `workflows/` | H3 text-to-video and image-to-video at full quality. A short idea goes through the remote **prompt writer**, which also sees the picture for image to video, before H3. |
| `prompt_writer/` | The writer's system prompts, condensed from MiniMax's prompt guide. |
| `tests/` | `h3_test.py` runs a workflow through ComfyUI's API and checks it. It checks speed per phase and step, VRAM/RAM/swap peaks, errors in the log, frame count, black/frozen frames, the soundtrack, and the prompt format. `compare_videos.py` gives PSNR/SSIM between two renders. |
| `benchmarks/` | The kernel and disk measurements behind the numbers in this README. |

## Install

1. Copy the folders in `custom_nodes/` into `ComfyUI/custom_nodes/`, `workflows/*.json` into `ComfyUI/user/default/workflows/`, and `tests/` to `ComfyUI/user/h3_tests/`.
2. Set `REMOTE_LLM_URL` (for example `http://127.0.0.1:8000/v1`) if you use the prompt writer, and `TRITON_CACHE_DIR` if your home partition is short on space.
3. Get the H3 files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):
   - `minimax_h3_fl2va_pruned_bf16`, `qwen3vl_32b_minimax_h3_bf16`
   - `minimax_h3_video_vae_fp16`, `minimax_h3_audio_vae_fp32`
   - optionally the 8-step turbo LoRA

   The weights are under the MiniMax H3 Community License, not this repo's.

Each add-on is independent. `COMFY_READAHEAD_MB=0` and `COMFY_TRITON_ATTENTION=0` switch the first two off. `triton_attention` only enables itself on gfx1200/gfx1201.

## Attention on gfx1201

MiniMax H3 shape (56 heads x 128, bf16, 38,196 tokens), error against fp32 on 512 rows per head:

| Kernel | TFLOPS | Relative error |
|---|---|---|
| PyTorch SDPA flash (AOTriton 0.13, ROCm 10.0) | 72 | 2.3e-3 |
| **`triton_attention`** (BLOCK_M 128, BLOCK_N 32, 8 warps, 1 stage) | **94** | **2.3e-3** |
| AMD aiter Triton FA2 (RDNA config) | 81 | 2.3e-3 |
| PyTorch SDPA flash (AOTriton 0.14, ROCm 10.1, PyTorch 2.14) | 81 | 3.7e-3 |
| SageAttention 1.0.6 (int8 QK, Triton) | 66-77 | 1.3e-2 |

The card sits at its 340 W limit (2.6-2.8 GHz) during these kernels.

## What did not help (same quality required)

- **SageAttention 1.0.6:** slower than SDPA on this card and 6x less accurate. SageAttention 2.x has no AMD kernels.
- **ROCm 10.1 + PyTorch 2.14:** its SDPA is faster than 10.0's but less accurate, and the matrix multiplies are unchanged.
- **PyTorch TunableOp / rocBLAS:** H3's projections already run at 130-137 TFLOPS with hipBLASLt's defaults, and the best tuned kernel gained under 1%.
- **Triton AMD switches:** block ping-pong and async copy changed nothing; turning off in-thread transpose or buffer ops was slower.

## Research round: what else was tried

Attention timings below are at H3's size (56 heads x 128, 38k tokens), and errors are against fp32. The "real data" rows use q/k/v captured from an H3 render with `benchmarks/capture_attention.py`. On that data the top 5% of keys hold 85-97% of each query's attention.

| Idea (source) | Result on the RX 9070 XT | Used |
|---|---|---|
| SageAttention 2 recipe (int8 Q.K^T, fp8 P.V; [paper](https://arxiv.org/abs/2411.10958)), ported to RDNA4 WMMA in Triton (`benchmarks/sage_rdna4.py`) | 45 TFLOPS vs 94 exact, 10-16x the error | no |
| int8 Q.K^T only | 66 TFLOPS, 4x the error | no |
| FlashAttention-4 conditional rescaling ([paper](https://arxiv.org/abs/2603.05451)) | +1% | no |
| Skip only key tiles that cannot matter at bf16 precision (each skipped key below e^-20 of the row max) | 0% of tiles skippable on real H3 data (0.5-12% at e^-12): its attention has a long tail | no |
| ComfyUI "Model Sparse Attention" (Sol-Attn, defaults) | 1.55x whole clip, but a different take: same scene and motion, different face and details (PSNR 18.7 dB vs exact) | optional |
| ComfyUI EasyCache (defaults) | 1.43x whole clip, PSNR 30.4 dB / SSIM 0.90 vs exact | **fast-mode switch, off by default** |
| MIOpen full algorithm search for the VAE convolutions | Decode about 9% faster (4 s per clip), plus a one-time search for every new shape | no |
| Text-encoder results kept on disk | 85 s saved per repeated prompt | **yes** |

- **Why the int8/fp8 attention loses here:** RDNA4's int8/fp8 matrix instructions are about 1.5x faster than bf16 in a plain GEMM. In attention, though, every score has to be converted and rescaled per element, and P has to be converted to fp8. That vector work costs more than the faster matrix math saves: during those kernels the clocks rise to 3.2 GHz, which means the matrix units sit idle.
- **Why not translate SageAttention 2's CUDA code:** its kernels are written in NVIDIA PTX matrix instructions with NVIDIA register layouts (and FP8 with 22-bit accumulators). HIPIFY converts the CUDA runtime API but not that code, so a port means rewriting the kernels for RDNA4's WMMA instructions. That is what `sage_rdna4.py` measures.
- **Same-quality check for approximations:** render the exact and the approximate version in one ComfyUI session and compare them with `tests/compare_videos.py`. Within one session an exact re-render is bit-identical. After a restart it already differs by 19-23 dB, which is the scale to judge an approximation against.

## Notes

- With the read-ahead on, 5 s and longer clips at 768p are compute-bound. Shorter clips, the text encoder and the first step of each new prompt are still limited by the disk, so faster storage than NTFS on Linux would shorten those.
- The 9070 XT is power-limited (340 W, the default maximum). An undervolt through the amdgpu overdrive interface (`amdgpu.ppfeaturemask` with bit 0x4000 set) should raise clocks at the same power. That is not covered here.
- `triton_attention` has the same accuracy as SDPA but is not bit-identical to it. Over 20 sampling steps that becomes small differences in fine detail, with the same scene, motion and timing. It is no bigger than ComfyUI's own run-to-run variation here: renders are bit-identical within one session, but the same seed and kernel after a restart already differed by 19-23 dB PSNR in these tests. Triton vs SDPA with the same input measured 20.9-21.4 dB.

## License

GPL-3.0, like ComfyUI. MiniMax H3 weights are covered by their own license.
