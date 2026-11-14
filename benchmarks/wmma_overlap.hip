// overlap3.hip - cost of softmax-like vector work interleaved with WMMA in the same wave (gfx1201).
// Per 8 independent WMMAs (16x16x16 bf16) each wave also does NEXP exp2 and NFMA fma on independent registers.
// Configs run alternately for many rounds; the median time of each is compared with the WMMA-only loop.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

template <int NEXP, int NFMA>
__global__ void __launch_bounds__(256) kern(float* out, int iters) {
    int lane = threadIdx.x % 32;
    bf16x8 a, b;
#pragma unroll
    for (int e = 0; e < 8; e++) { a[e] = (__bf16)(0.001f * (lane + e)); b[e] = (__bf16)(0.002f * (lane - e)); }
    f32x8 c[8];
#pragma unroll
    for (int j = 0; j < 8; j++) c[j] = (f32x8){};
    float x[NEXP + 1], y[NFMA + 1];
#pragma unroll
    for (int j = 0; j <= NEXP; j++) x[j] = 0.01f * (lane + j);
#pragma unroll
    for (int j = 0; j <= NFMA; j++) y[j] = 0.02f * (lane - j);
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c[j]);
#pragma unroll
        for (int j = 0; j < NEXP; j++) x[j] = __builtin_amdgcn_exp2f(x[j] * -0.75f);
#pragma unroll
        for (int j = 0; j < NFMA; j++) y[j] = __builtin_fmaf(y[j], 0.999f, 0.001f);
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) for (int e = 0; e < 8; e++) s += c[j][e];
    for (int j = 0; j < NEXP; j++) s += x[j];
    for (int j = 0; j < NFMA; j++) s += y[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int NEXP, int NFMA>
float once(float* out, int blocks, int iters) {
    hipEvent_t e0, e1;
    (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);
    (void)hipEventRecord(e0);
    for (int r = 0; r < 5; r++) kern<NEXP, NFMA><<<blocks, 256>>>(out, iters);
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    float ms; (void)hipEventElapsedTime(&ms, e0, e1);
    (void)hipEventDestroy(e0); (void)hipEventDestroy(e1);
    return ms / 5;
}

int main() {
    int blocks = 64 * 4, iters = 3000;
    float* out; (void)hipMalloc(&out, blocks * 256 * sizeof(float));
    const char* names[] = {"wmma only", "+4 exp", "+8 exp", "+16 fma", "+32 fma", "+4 exp +16 fma (FA-like)", "+8 exp +32 fma"};
    std::vector<float> t[7];
    for (int w = 0; w < 30; w++) once<0, 0>(out, blocks, iters);  // reach the power-limited steady clock
    for (int r = 0; r < 25; r++) {
        t[0].push_back(once<0, 0>(out, blocks, iters));
        t[1].push_back(once<4, 0>(out, blocks, iters));
        t[2].push_back(once<8, 0>(out, blocks, iters));
        t[3].push_back(once<0, 16>(out, blocks, iters));
        t[4].push_back(once<0, 32>(out, blocks, iters));
        t[5].push_back(once<4, 16>(out, blocks, iters));
        t[6].push_back(once<8, 32>(out, blocks, iters));
    }
    double flops = double(blocks) * 8 * iters * 8 * 16 * 16 * 16 * 2;
    float base = 0;
    for (int k = 0; k < 7; k++) {
        std::sort(t[k].begin(), t[k].end());
        float med = t[k][t[k].size() / 2];
        if (k == 0) base = med;
        printf("%-28s median %.3f ms (%5.1f%% of wmma-only time, %.0f TFLOPS of wmma)\n", names[k], med, 100 * med / base, flops / med / 1e9);
    }
    return 0;
}
