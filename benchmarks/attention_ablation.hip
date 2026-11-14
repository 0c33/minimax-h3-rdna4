// Exact flash attention forward for RDNA4 (gfx1200/gfx1201): bf16 or fp16 in/out, head dim 64 or 128, no mask, not causal.
//
// gfx12 WMMA layouts make the transposed problem cheap: computing S^T = K Q^T gives each lane 8 keys of ONE query,
//  - so the softmax max/sum over keys run inside the lane plus one swap with the lane holding the other 8 keys,
//  - the accumulator layout equals the B-operand layout, so P^T feeds O^T = V^T P^T without any shuffle,
//  - and O^T's accumulator also gives each lane one query, so the online-softmax rescale is lane-local too.
// The rescale is skipped when no lane's running max changed (alpha == 1 exactly, so skipping is exact).
// K is staged row-major in LDS and V transposed (dims x keys), double buffered: one barrier per key tile.
// bf16: p is rounded half-up with integer ops (gfx12 has no fp32->bf16 instruction; ties are 1 in 65536).
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

#define NW 8            // waves per block, 16 queries each
#define BM (16 * NW)
#define NT (NW * 32)

template <bool F16> struct Ty;
template <> struct Ty<false> {
    typedef __bf16 T; typedef bf16x8 V8;
    static __device__ __forceinline__ f32x8 wmma(V8 a, V8 b, f32x8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c); }
    static __device__ __forceinline__ uint32_t pack(float a, float b) {
        uint32_t ua = __float_as_uint(a) + 0x8000u, ub = __float_as_uint(b) + 0x8000u;
        return __builtin_amdgcn_perm(ub, ua, 0x07060302u);  // high halves: a -> bits 0-15, b -> bits 16-31
    }
};
template <> struct Ty<true> {
    typedef _Float16 T; typedef f16x8 V8;
    static __device__ __forceinline__ f32x8 wmma(V8 a, V8 b, f32x8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c); }
    static __device__ __forceinline__ uint32_t pack(float a, float b) {  // round to nearest even (v_cvt_f16_f32)
        return (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)a) | ((uint32_t)__builtin_bit_cast(uint16_t, (_Float16)b) << 16);
    }
};

__device__ __forceinline__ float xlane16(float x) {  // the value of lane ^ 16
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__float_as_uint(x), __float_as_uint(x), 0x76543210, 0xfedcba98, false, false));
}

template <int D, int BN, bool F16>
__device__ __forceinline__ void fa_body(const typename Ty<F16>::T* __restrict__ Q, const typename Ty<F16>::T* __restrict__ K,
       const typename Ty<F16>::T* __restrict__ V, typename Ty<F16>::T* __restrict__ O,
       long s_qb, long s_qh, long s_qm, long s_kb, long s_kh, long s_kn, long s_vb, long s_vh, long s_vn,
       long s_ob, long s_oh, long s_om, int H, int NQ, int NKV, float qk_scale, typename Ty<F16>::T* smem)
{
    typedef typename Ty<F16>::T T;
    typedef typename Ty<F16>::V8 V8;
    constexpr int KS = D + 8, VS = BN + 8;  // LDS row strides; the extra 16 bytes avoid bank conflicts
    constexpr int KTILE = BN * KS, VTILE = D * VS;
    constexpr int CH = BN * D / 8 / NT;     // 16-byte chunks of K and of V each thread stages per tile
    constexpr int DT = D / 16;
    static_assert(CH >= 1 && NT % BN == 0 && (NT / BN) * CH == D / 8, "staging layout");

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int c = lane & 15, g = lane >> 4;
    const int b = blockIdx.y / H, h = blockIdx.y % H;
    const T* Qp = Q + b * s_qb + h * s_qh;
    const T* Kp = K + b * s_kb + h * s_kh;
    const T* Vp = V + b * s_vb + h * s_vh;
    const int qrow = blockIdx.x * BM + wave * 16 + c;

    // Q^T as the B operand of S^T = K Q^T: lane (query c, half g) holds dims 16ks + 8g .. +7
    V8 qf[DT];
#pragma unroll
    for (int ks = 0; ks < DT; ks++)
        qf[ks] = qrow < NQ ? *(const V8*)(Qp + (long)qrow * s_qm + 16 * ks + 8 * g) : (V8){};

    f32x8 o[DT];
#pragma unroll
    for (int dt = 0; dt < DT; dt++) o[dt] = (f32x8){};
    float m = -INFINITY, l = 0.f;

    // K chunks: consecutive threads read one key row (coalesced) and store it row-major.
    // V chunks: lane = key, so the transposed 2-byte stores of a wave go to consecutive addresses.
    u32x4 kreg[CH], vreg[CH];
    auto gload = [&](int n0) {
#pragma unroll
        for (int r = 0; r < CH; r++) {
            int ck = tid + r * NT, key = ck / (D / 8), d8 = (ck % (D / 8)) * 8;
#ifndef ABL_NOK
            kreg[r] = n0 + key < NKV ? *(const u32x4*)(Kp + (long)(n0 + key) * s_kn + d8) : (u32x4){};
#endif
            int vk = tid % BN, vd8 = ((tid / BN) + r * (NT / BN)) * 8;
#ifndef ABL_NOV
            vreg[r] = n0 + vk < NKV ? *(const u32x4*)(Vp + (long)(n0 + vk) * s_vn + vd8) : (u32x4){};
#endif
        }
    };
    auto sstore = [&](int buf) {
        T* ks_ = smem + buf * (KTILE + VTILE);
        uint16_t* vs_ = (uint16_t*)(ks_ + KTILE);
#pragma unroll
        for (int r = 0; r < CH; r++) {
            int ck = tid + r * NT, key = ck / (D / 8), d8 = (ck % (D / 8)) * 8;
#ifndef ABL_NOK
            *(u32x4*)(ks_ + key * KS + d8) = kreg[r];
#endif
            int vk = tid % BN, vd8 = ((tid / BN) + r * (NT / BN)) * 8;
            const uint16_t* e = (const uint16_t*)&vreg[r];
#ifndef ABL_NOV
#pragma unroll
            for (int i = 0; i < 8; i++) vs_[(vd8 + i) * VS + vk] = e[i];
#endif
        }
    };

    const int ntiles = (NKV + BN - 1) / BN;
    gload(0);
    sstore(0);
    __syncthreads();

    for (int t = 0; t < ntiles; t++) {
        const int n0 = t * BN;
#ifndef ABL_NOSTAGE
#ifdef ABL_HOTLOAD
        if (t + 1 < ntiles) gload(0);
#else
        if (t + 1 < ntiles) gload(n0 + BN);
#endif
#endif
#ifdef ABL_NOSTAGE
        const T* ks_ = smem;
#else
        const T* ks_ = smem + (t & 1) * (KTILE + VTILE);
#endif
        const T* vs_ = ks_ + KTILE;

        // S^T: s[mt][e] = score of key n0 + 16mt + 8g + e for query c (K fragments loaded one step ahead)
        f32x8 s[BN / 16];
        V8 kf[BN / 16], kn[BN / 16];
#pragma unroll
        for (int mt = 0; mt < BN / 16; mt++) {
            s[mt] = (f32x8){};
            kf[mt] = *(const V8*)(ks_ + (16 * mt + c) * KS + 8 * g);
        }
#pragma unroll
        for (int ks = 0; ks < DT; ks++) {
            if (ks < DT - 1) {
#pragma unroll
                for (int mt = 0; mt < BN / 16; mt++) kn[mt] = *(const V8*)(ks_ + (16 * mt + c) * KS + 16 * (ks + 1) + 8 * g);
            }
#pragma unroll
            for (int mt = 0; mt < BN / 16; mt++) s[mt] = Ty<F16>::wmma(kf[mt], qf[ks], s[mt]);
#pragma unroll
            for (int mt = 0; mt < BN / 16; mt++) kf[mt] = kn[mt];
        }
        if (n0 + BN > NKV) {  // last tile: keys past the end do not exist
#pragma unroll
            for (int mt = 0; mt < BN / 16; mt++)
#pragma unroll
                for (int e = 0; e < 8; e++)
                    if (n0 + 16 * mt + 8 * g + e >= NKV) s[mt][e] = -INFINITY;
        }

#ifdef ABL_NOSOFTMAX
        V8 pf[BN / 16];
#pragma unroll
        for (int mt = 0; mt < BN / 16; mt++) {
            u32x4 packed = {Ty<F16>::pack(s[mt][0], s[mt][1]), Ty<F16>::pack(s[mt][2], s[mt][3]), Ty<F16>::pack(s[mt][4], s[mt][5]), Ty<F16>::pack(s[mt][6], s[mt][7])};
            pf[mt] = __builtin_bit_cast(V8, packed);
        }
        l = 1.f;
#else
        float mx = s[0][0];
#pragma unroll
        for (int mt = 0; mt < BN / 16; mt++)
#pragma unroll
            for (int e = 0; e < 8; e++) mx = __builtin_fmaxf(mx, s[mt][e]);
        mx = __builtin_fmaxf(mx, xlane16(mx));
        const float mnew = __builtin_fmaxf(m, mx * qk_scale);
        const float alpha = __builtin_amdgcn_exp2f(m - mnew);
#ifdef ABL_NORESCALE
        if (0) {
#else
        if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#endif
#pragma unroll
            for (int dt = 0; dt < DT; dt++) o[dt] *= alpha;
            l *= alpha;
        }
        m = mnew;

        V8 pf[BN / 16];
#pragma unroll
        for (int mt = 0; mt < BN / 16; mt++) {
            float p[8];
#pragma unroll
            for (int e = 0; e < 8; e++) {
                p[e] = __builtin_amdgcn_exp2f(s[mt][e] * qk_scale - mnew);
                l += p[e];
            }
            u32x4 packed = {Ty<F16>::pack(p[0], p[1]), Ty<F16>::pack(p[2], p[3]), Ty<F16>::pack(p[4], p[5]), Ty<F16>::pack(p[6], p[7])};
            pf[mt] = __builtin_bit_cast(V8, packed);
        }
#endif

        // O^T[dims 16dt + 8g + e, query c] += V^T[dims, keys] P^T[keys, query c], independent accumulators interleaved
#pragma unroll
        for (int kk = 0; kk < BN / 16; kk++)
#pragma unroll
            for (int dt = 0; dt < DT; dt++) {
                V8 vf = *(const V8*)(vs_ + (16 * dt + c) * VS + 16 * kk + 8 * g);
                o[dt] = Ty<F16>::wmma(vf, pf[kk], o[dt]);
            }

#ifndef ABL_NOSTAGE
        if (t + 1 < ntiles) sstore((t + 1) & 1);
        __syncthreads();
#endif
    }

    l += xlane16(l);
    if (qrow < NQ) {
        T* Op = O + b * s_ob + h * s_oh + (long)qrow * s_om;
#pragma unroll
        for (int dt = 0; dt < DT; dt++) {
            u32x4 packed = {Ty<F16>::pack(o[dt][0] / l, o[dt][1] / l), Ty<F16>::pack(o[dt][2] / l, o[dt][3] / l),
                            Ty<F16>::pack(o[dt][4] / l, o[dt][5] / l), Ty<F16>::pack(o[dt][6] / l, o[dt][7] / l)};
            *(u32x4*)(Op + 16 * dt + 8 * g) = packed;
        }
    }
}

#define INSTANCE(NAME, D, BN, F16)                                                                                       \
    extern "C" __global__ void __launch_bounds__(NT, 1)                                                                 \
    NAME(const Ty<F16>::T* Q, const Ty<F16>::T* K, const Ty<F16>::T* V, Ty<F16>::T* O,                                   \
         long s_qb, long s_qh, long s_qm, long s_kb, long s_kh, long s_kn, long s_vb, long s_vh, long s_vn,             \
         long s_ob, long s_oh, long s_om, int H, int NQ, int NKV, float qk_scale) {                                     \
        __shared__ __attribute__((aligned(16))) Ty<F16>::T smem[2 * (BN * (D + 8) + D * (BN + 8))];                     \
        fa_body<D, BN, F16>(Q, K, V, O, s_qb, s_qh, s_qm, s_kb, s_kh, s_kn, s_vb, s_vh, s_vn, s_ob, s_oh, s_om,        \
                            H, NQ, NKV, qk_scale, smem);                                                                 \
    }

INSTANCE(fa_bf16_d128_n32, 128, 32, false)
INSTANCE(fa_f16_d128_n32, 128, 32, true)
INSTANCE(fa_bf16_d64_n32, 64, 32, false)
INSTANCE(fa_f16_d64_n32, 64, 32, true)
INSTANCE(fa_bf16_d64_n64, 64, 64, false)
INSTANCE(fa_f16_d64_n64, 64, 64, true)
