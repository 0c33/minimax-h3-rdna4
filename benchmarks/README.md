# Benchmarks

Run with ComfyUI's Python while ComfyUI is idle (it holds VRAM otherwise: `curl -X POST http://127.0.0.1:8188/free -d '{"unload_models": true, "free_memory": true}'`). Set `TRITON_CACHE_DIR` if your home partition is small.

| Script | Measures |
|---|---|
| `attn_compare.py [S ...]` | Attention speed and error vs fp32 at H3 shapes: SDPA, `triton_fa.py` (autotunes on first run, about 8 min), optionally SageAttention 1.0.6 (`SAGE_PATH=` unpacked wheel) and AMD aiter's Triton FA (`AITER_FA_PATH=` `aiter/ops/triton/_triton_kernels`). |
| `fa_sweep.py S BM,BN,warps,stages,wpe ...` | `triton_fa.py` with pinned configs; Triton AMD switches come from the environment (`TRITON_HIP_USE_BLOCK_PINGPONG`, ...). |
| `kernel_cases.py` | `custom_nodes/triton_attention` against SDPA on batch, cross-attention, fp16, custom scale and both memory layouts. |
| `gemm_bench.py [rows]` | H3's four projections; `GEMM_BACKEND=rocblas` or `PYTORCH_TUNABLEOP_*` to compare backends. |
| `readbench.py FILE`, `dupbench.py FILE` | Read speed of a model file by chunk size and thread count, and with two readers at once. |
