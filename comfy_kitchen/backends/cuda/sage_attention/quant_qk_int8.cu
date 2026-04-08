// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Optimized INT8 per-thread quantization for Q and K (SageAttention).
//
// All tensors assumed contiguous [B, H, L, D] (HND layout).
// Fused single-launch kernel: both Q and K blocks in one grid.
//   Q path: warp-per-group, single-pass, vectorized float2 loads / int32
//   stores. K path: warp-per-group, double-read (L1-cached), vectorized loads /
//   stores. No __syncthreads anywhere – pure warp-level reductions.

#include "dtype_dispatch.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ int8_t quant_int8(float v, float scale) {
  float t = v / scale;
  t += (t >= 0.f ? 0.5f : -0.5f);
  return static_cast<int8_t>(t);
}

__device__ __forceinline__ float warp_reduce_fmax(float v) {
  constexpr unsigned M = 0xffffffff;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_xor_sync(M, v, o));
  return v;
}

__device__ __forceinline__ void store4_i8(int8_t *ptr, int8_t a, int8_t b,
                                          int8_t c, int8_t d) {
  *reinterpret_cast<int32_t *>(ptr) =
      (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
      ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

// ---------------------------------------------------------------------------
// Q processing: warp-per-group, 4 warps, each warp handles 2 off_tld.
// Single-pass: cache values in registers, reduce, quantize, store.
// Vectorized float2 loads (4 bf16/fp16 = 8 bytes) and int32 stores (4 int8).
//
// Pointers are pre-offset to the (b, h) slice by the caller.
// ---------------------------------------------------------------------------
#pragma nv_diag_suppress 1056
template <typename T, int NR>
__device__ void process_q(const T *__restrict__ in, int8_t *__restrict__ out,
                          float *__restrict__ sc_buf, const int oblk,
                          const int L, const int C, const int BLKQ,
                          const int WARPQ) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int nsub = BLKQ / WARPQ;
  const int ch = lane << 2;

  for (int g = 0; g < 2; ++g) {
    const int otld = wid * 2 + g;
    const int base = (oblk / nsub) * BLKQ + (oblk % nsub) * WARPQ + otld;

    float v[NR * 4];
    float mx = 0.f;

    if (ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        const int n = base + j * 8;
        if (n < L) {
          float2 raw =
              *reinterpret_cast<const float2 *>(&in[(int64_t)n * C + ch]);
          const T *vals = reinterpret_cast<const T *>(&raw);
          v[j * 4] = static_cast<float>(vals[0]);
          v[j * 4 + 1] = static_cast<float>(vals[1]);
          v[j * 4 + 2] = static_cast<float>(vals[2]);
          v[j * 4 + 3] = static_cast<float>(vals[3]);
          mx =
              fmaxf(mx, fmaxf(fmaxf(fabsf(v[j * 4]), fabsf(v[j * 4 + 1])),
                              fmaxf(fabsf(v[j * 4 + 2]), fabsf(v[j * 4 + 3]))));
        } else {
          v[j * 4] = v[j * 4 + 1] = v[j * 4 + 2] = v[j * 4 + 3] = 0.f;
        }
      }
    } else if (ch < C) {
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        const int n = base + j * 8;
        if (n < L) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            v[j * 4 + c] = (ch + c < C)
                               ? static_cast<float>(in[(int64_t)n * C + ch + c])
                               : 0.f;
            mx = fmaxf(mx, fabsf(v[j * 4 + c]));
          }
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c)
            v[j * 4 + c] = 0.f;
        }
      }
    }

    mx = warp_reduce_fmax(mx);
    const float sc = mx / 127.f + 1e-7f;

    if (lane == 0)
      sc_buf[oblk * 8 + otld] = sc;

    if (ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        const int n = base + j * 8;
        if (n < L) {
          store4_i8(&out[(int64_t)n * C + ch], quant_int8(v[j * 4], sc),
                    quant_int8(v[j * 4 + 1], sc), quant_int8(v[j * 4 + 2], sc),
                    quant_int8(v[j * 4 + 3], sc));
        }
      }
    } else if (ch < C) {
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        const int n = base + j * 8;
        if (n < L) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            if (ch + c < C)
              out[(int64_t)n * C + ch + c] = quant_int8(v[j * 4 + c], sc);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// K processing: warp-per-group, 4 warps, each warp handles 1 off_tld.
// Single-pass: cache values in registers, reduce, quantize, store.
// Vectorized float2 loads and int32 stores.
// ---------------------------------------------------------------------------
template <typename T, int NL>
__device__ void process_k(const T *__restrict__ in, int8_t *__restrict__ out,
                          float *__restrict__ sc_buf, const int oblk,
                          const int L, const int C, const int BLKK,
                          const int WARPK) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int ch = lane << 2;
  const int otld = wid;

  float v[2 * NL * 4];
  float mx = 0.f;

  if (ch + 3 < C) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int n = oblk * WARPK + j * 8 + otld * 2 + p;
        const int vi = (j * 2 + p) * 4;
        if (n < L) {
          float2 raw =
              *reinterpret_cast<const float2 *>(&in[(int64_t)n * C + ch]);
          const T *vals = reinterpret_cast<const T *>(&raw);
          v[vi] = static_cast<float>(vals[0]);
          v[vi + 1] = static_cast<float>(vals[1]);
          v[vi + 2] = static_cast<float>(vals[2]);
          v[vi + 3] = static_cast<float>(vals[3]);
          mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])),
                               fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
        } else {
          v[vi] = v[vi + 1] = v[vi + 2] = v[vi + 3] = 0.f;
        }
      }
    }
  } else if (ch < C) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int n = oblk * WARPK + j * 8 + otld * 2 + p;
        const int vi = (j * 2 + p) * 4;
        if (n < L) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            v[vi + c] = (ch + c < C)
                            ? static_cast<float>(in[(int64_t)n * C + ch + c])
                            : 0.f;
            mx = fmaxf(mx, fabsf(v[vi + c]));
          }
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c)
            v[vi + c] = 0.f;
        }
      }
    }
  }

  mx = warp_reduce_fmax(mx);
  const float sc = mx / 127.f + 1e-7f;

  if (lane == 0)
    sc_buf[oblk * 4 + otld] = sc;

  if (ch + 3 < C) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int n = oblk * WARPK + j * 8 + otld * 2 + p;
        const int vi = (j * 2 + p) * 4;
        if (n < L) {
          store4_i8(&out[(int64_t)n * C + ch], quant_int8(v[vi], sc),
                    quant_int8(v[vi + 1], sc), quant_int8(v[vi + 2], sc),
                    quant_int8(v[vi + 3], sc));
        }
      }
    }
  } else if (ch < C) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int n = oblk * WARPK + j * 8 + otld * 2 + p;
        const int vi = (j * 2 + p) * 4;
        if (n < L) {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            if (ch + c < C)
              out[(int64_t)n * C + ch + c] = quant_int8(v[vi + c], sc);
          }
        }
      }
    }
  }
}
#pragma nv_diag_default 1056

// ---------------------------------------------------------------------------
// Fused Q+K kernel – single launch, 128 threads, no shared memory.
// blockIdx.x < q_oblk_count  →  Q path
// blockIdx.x >= q_oblk_count →  K path
// ---------------------------------------------------------------------------
template <typename T, int NR, int NL>
__global__ __launch_bounds__(128) void quant_qk_fused(
    const T *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const T *__restrict__ k_in,
    int8_t *__restrict__ k_out, float *__restrict__ k_sb, const int Lq,
    const int Lk, const int C, const int BLKQ, const int WARPQ, const int BLKK,
    const int WARPK, const int q_oblk_count, const int H_q, const int H_kv,
    const int q_sc_per_h, const int k_sc_per_h) {
  const int h = blockIdx.y, b = blockIdx.z;

  if (blockIdx.x < (unsigned)q_oblk_count) {
    if (h >= H_q)
      return;
    const int64_t bh = ((int64_t)b * H_q + h) * Lq * C;
    const int64_t sbh = ((int64_t)b * H_q + h) * q_sc_per_h;
    process_q<T, NR>(q_in + bh, q_out + bh, q_sb + sbh, blockIdx.x, Lq, C, BLKQ,
                     WARPQ);
  } else {
    if (h >= H_kv)
      return;
    const int64_t bh = ((int64_t)b * H_kv + h) * Lk * C;
    const int64_t sbh = ((int64_t)b * H_kv + h) * k_sc_per_h;
    process_k<T, NL>(k_in + bh, k_out + bh, k_sb + sbh,
                     (int)blockIdx.x - q_oblk_count, Lk, C, BLKK, WARPK);
  }
}

} // namespace

extern "C" void launch_quant_qk_per_thread_int8(
    const void *q, void *q_int8, void *q_scale, const void *k, void *k_int8,
    void *k_scale, int B, int H_q, int Lq, int H_kv, int Lk, int C, int BLKQ,
    int WARPQ, int BLKK, int WARPK, int input_dtype_code, cudaStream_t stream) {
  const int q_oblk = (Lq + BLKQ - 1) / BLKQ * (BLKQ / WARPQ);
  const int k_oblk = (Lk + BLKK - 1) / BLKK * (BLKK / WARPK);
  const int nr = WARPQ >> 3;
  const int nl = WARPK >> 3;
  const int q_sc_per_h = q_oblk * 8;
  const int k_sc_per_h = k_oblk * 4;

#define LAUNCH_FUSED(T, NR, NL)                                                \
  do {                                                                         \
    const int H_max = H_q > H_kv ? H_q : H_kv;                                 \
    dim3 g(q_oblk + k_oblk, H_max, B);                                         \
    quant_qk_fused<T, NR, NL><<<g, 128, 0, stream>>>(                          \
        (const T *)q, (int8_t *)q_int8, (float *)q_scale, (const T *)k,        \
        (int8_t *)k_int8, (float *)k_scale, Lq, Lk, C, BLKQ, WARPQ, BLKK,      \
        WARPK, q_oblk, H_q, H_kv, q_sc_per_h, k_sc_per_h);                     \
  } while (0)

#define DISPATCH(T, NR, NL) LAUNCH_FUSED(T, NR, NL)

#define DO(T)                                                                  \
  switch (nr * 100 + nl) {                                                     \
  case 402:                                                                    \
    DISPATCH(T, 4, 2);                                                         \
    break;                                                                     \
  case 404:                                                                    \
    DISPATCH(T, 4, 4);                                                         \
    break;                                                                     \
  case 408:                                                                    \
    DISPATCH(T, 4, 8);                                                         \
    break;                                                                     \
  case 208:                                                                    \
    DISPATCH(T, 2, 8);                                                         \
    break;                                                                     \
  case 808:                                                                    \
    DISPATCH(T, 8, 8);                                                         \
    break;                                                                     \
  default:                                                                     \
    DISPATCH(T, 4, 8);                                                         \
    break;                                                                     \
  }

  DISPATCH_HALF_DTYPE(input_dtype_code, T, [&] { DO(T); });

#undef LAUNCH_FUSED
#undef DISPATCH
#undef DO
}
