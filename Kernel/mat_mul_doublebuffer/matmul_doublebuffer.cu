#include <cuda_runtime.h>
#include "../../Header/matmul_kernels.cuh"

// Double-buffered (software-pipelined) version of the tuned kernel. Each iteration
// prefetches the next K-block from global memory into registers while the current
// block's FMAs run, overlapping global-load latency with compute. Geometry is the
// tuned winner: 128x128, BK=16, 16x8 register tile (128 threads). Square N, mult of 128.
#define BM 128
#define BN 128
#define BK 16
#define TM 16
#define TN 8

__global__ void matmul_doublebuffer(const float *A, const float *B, float *C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[2][BK * BM];   // transposed, double-buffered
    __shared__ float Bs[2][BK * BN];

    A += cRow * BM * N;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // constexpr so the loops below fully unroll and the prefetch arrays size at
    // compile time (this is what makes the kernel fast — see matmul_tuned.cu).
    constexpr int numThreads = (BM * BN) / (TM * TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    const int irA = threadIdx.x / (BK / 4), icA = threadIdx.x % (BK / 4);
    constexpr int strideA = numThreads / (BK / 4);
    const int irB = threadIdx.x / (BN / 4), icB = threadIdx.x % (BN / 4);
    constexpr int strideB = numThreads / (BN / 4);
    constexpr int nA = BM / strideA;
    constexpr int nB = BK / strideB;

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    // prologue: load block 0 into buffer 0
#pragma unroll
    for (int o = 0; o < BM; o += strideA) {
        float4 t = reinterpret_cast<const float4 *>(&A[(irA + o) * N + icA * 4])[0];
        As[0][(icA * 4 + 0) * BM + irA + o] = t.x;
        As[0][(icA * 4 + 1) * BM + irA + o] = t.y;
        As[0][(icA * 4 + 2) * BM + irA + o] = t.z;
        As[0][(icA * 4 + 3) * BM + irA + o] = t.w;
    }
#pragma unroll
    for (int o = 0; o < BK; o += strideB)
        reinterpret_cast<float4 *>(&Bs[0][(irB + o) * BN + icB * 4])[0] =
            reinterpret_cast<const float4 *>(&B[(irB + o) * N + icB * 4])[0];
    __syncthreads();
    A += BK;
    B += BK * N;

    const int numTiles = N / BK;
    int buf = 0;
    for (int t = 0; t < numTiles; ++t) {
        const bool has_next = (t + 1 < numTiles);
        float4 rA[nA];
        float4 rB[nB];
        if (has_next) {
#pragma unroll
            for (int o = 0, idx = 0; o < BM; o += strideA, ++idx)
                rA[idx] = reinterpret_cast<const float4 *>(&A[(irA + o) * N + icA * 4])[0];
#pragma unroll
            for (int o = 0, idx = 0; o < BK; o += strideB, ++idx)
                rB[idx] = reinterpret_cast<const float4 *>(&B[(irB + o) * N + icB * 4])[0];
            A += BK;
            B += BK * N;
        }
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[buf][k * BM + threadRow * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[buf][k * BN + threadCol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        if (has_next) {
            const int nbuf = buf ^ 1;
#pragma unroll
            for (int o = 0, idx = 0; o < BM; o += strideA, ++idx) {
                As[nbuf][(icA * 4 + 0) * BM + irA + o] = rA[idx].x;
                As[nbuf][(icA * 4 + 1) * BM + irA + o] = rA[idx].y;
                As[nbuf][(icA * 4 + 2) * BM + irA + o] = rA[idx].z;
                As[nbuf][(icA * 4 + 3) * BM + irA + o] = rA[idx].w;
            }
#pragma unroll
            for (int o = 0, idx = 0; o < BK; o += strideB, ++idx)
                reinterpret_cast<float4 *>(&Bs[nbuf][(irB + o) * BN + icB * 4])[0] = rB[idx];
            __syncthreads();
            buf = nbuf;
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 v;
            v.x = acc[i * TN + j + 0];
            v.y = acc[i * TN + j + 1];
            v.z = acc[i * TN + j + 2];
            v.w = acc[i * TN + j + 3];
            reinterpret_cast<float4 *>(&C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = v;
        }
}

#undef BM
#undef BN
#undef BK
#undef TM
#undef TN
