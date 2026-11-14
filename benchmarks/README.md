# Benchmarks

Run with ComfyUI's Python while ComfyUI is idle (it holds VRAM otherwise: `curl -X POST http://127.0.0.1:8188/free -d '{"unload_models": true, "free_memory": true}'`). Set `TRITON_CACHE_DIR` if your home partition is small.

| Script | Measures |
|---|---|
| `attn_compare.py [S ...]` | Attention speed and error vs fp32 at H3 shapes: SDPA, `triton_fa.py` (autotunes on first run, about 8 min), optionally SageAttention 1.0.6 (`SAGE_PATH=` unpacked wheel) and AMD aiter's Triton FA (`AITER_FA_PATH=` `aiter/ops/triton/_triton_kernels`). |
| `fa_sweep.py S BM,BN,warps,stages,wpe ...` | `triton_fa.py` with pinned configs; Triton AMD switches come from the environment (`TRITON_HIP_USE_BLOCK_PINGPONG`, ...). |
| `kernel_cases.py` | `custom_nodes/triton_attention` against SDPA on batch, cross-attention, fp16, custom scale and both memory layouts. |
| `gemm_bench.py [rows]` | H3's four projections; `GEMM_BACKEND=rocblas` or `PYTORCH_TUNABLEOP_*` to compare backends. |
| `readbench.py FILE`, `dupbench.py FILE` | Read speed of a model file by chunk size and thread count, and with two readers at once. |
| `readbench_patterns.py FILE` | Buffered vs O_DIRECT reads, chunk sizes and reader counts through ntfs-3g, each test on a fresh region of the file. Compare the bytes read with `/proc/diskstats` to see ntfs-3g's read amplification (1.3x measured here). |
| `attention_addon_test.py` | (`COMFYUI_DIR=/path/to/ComfyUI`) `custom_nodes/triton_attention` as ComfyUI imports it: which kernel each call takes (HIP, Triton or SDPA), error vs fp32, output layout, and speed at the H3 shape. |
| `fa_variants.py [filter]` | Triton attention variants: the shipped kernel, the softmax scale folded into exp2, a hand-transposed S^T = K Q^T version, bf16 rounding without the NaN path (`rne`, bit-identical and ~3% faster), fp16 P.V, plus LLVM scheduler strategies, all with VGPR/spill and instruction counts. |
| `attention_ablation.hip` | The HIP attention kernel with switches that remove one part (`-DABL_NOSOFTMAX`, `-DABL_NORESCALE`, `-DABL_NOSTAGE`, `-DABL_NOK`, `-DABL_NOV`, `-DABL_HOTLOAD`) to see where its time goes. Results below. |
| `wmma_overlap.hip` | `hipcc --offload-arch=gfx1201 -O3`: how much vector work hides under WMMA in one wave (exp2 vs FMA per 8 WMMAs). |
| `rms_rope_bench.py` | comfy_kitchen's HIP RMSNorm + RoPE vs the Triton kernel at H3 DiT and VAE shapes, with the count of values that differ. |
| `fused_elementwise_test.py` | Fused SwiGLU and AdaLN scale/shift vs the eager ops: values that differ (0 for bf16) and speed. |

## gfx1201 findings behind the HIP kernel

- **WMMA vs vector work (`wmma_overlap.hip`).** Per 8 independent bf16 WMMAs: +16 FMAs cost 4.8%, +32 FMAs 13.7%, +4 exp2 13.2%, +8 exp2 22.6%. Plain VALU work partly hides under WMMA, the transcendental unit does not. A FlashAttention-like mix (4 exp2 + 16 FMA) costs 21%, so the ceiling for exact attention is about 145 TFLOPS against ~177 for WMMA alone.
- **No fp32 to bf16 instruction on gfx12.** The compiler's round-to-nearest-even takes 5 instructions per value, three of them for NaN, which softmax probabilities never are. Doing the rounding with integer ops and no NaN path is bit-identical and made the Triton kernel ~3% faster (fp16 has `v_cvt_f16_f32`).
- **Accumulator layout = B-operand layout.** Computing S^T = K Q^T gives each lane 8 keys of one query: the softmax reductions stay in-lane plus one `v_permlanex16`, P^T feeds the next WMMA directly, and the rescale is lane-local. Triton already picks this layout for its own kernel (4 cross-lane ops per loop); writing it transposed by hand in Triton broke that.
- **Attention is power-limited.** All exact kernels sit at 339 W. AMD's FlyDSL gfx1201 kernel does 9% more work per clock than the Triton one but clocks 2.5 instead of 2.7 GHz, so both end at ~94 TFLOPS. In the ablation, staging K and V into LDS costs 19% of the time (K 10%, V 8%), mostly as power: without the K staging the clock rises from 2.53 to 2.70 GHz; loading a cache-hot tile instead changed only 1.4%.
- **Ablation (`attention_ablation.hip`, H3 5 s shape, 382 ms in full):** no softmax 351 ms, no rescale 375 ms, no K/V staging 309 ms, neither 274 ms (153 TFLOPS).
- **GEMMs:** hipBLASLt runs H3's qkv/out/fc1 projections at 132-133 TFLOPS, 89% of the per-clock peak at the 2.27 GHz the power limit allows (fc2: 81%). A hand-written GEMM has little room.
- **Not available:** PyTorch's CK flash attention refuses gfx1201 ("architecture supported for CK: 0"); rocprofv3's SQ wave counters read zero without elevated rights.
