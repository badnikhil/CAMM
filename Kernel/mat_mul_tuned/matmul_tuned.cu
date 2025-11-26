#include <cuda_runtime.h>
#include "../../Header/matmul_kernels.cuh"

// Autotuned register-blocked kernel: 128x128 block tile, BK=16, asymmetric
// 16x8 thread register tile (128 threads/block). Shared A is stored transposed
// and all global<->shared traffic is float4-vectorized. The 16x8 (not 8x8 or
// 8x16) register tile is the autotune winner on Ampere. Square N, multiple of 128.
#define BM 128
#define BN 128
#define BK 16
#define TM 16
#define TN 8

__global__ void matmul_tuned(const float *A, const float *B, float *C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[BK * BM];   // transposed: As[k*BM + m]
    __shared__ float Bs[BK * BN];

    A += cRow * BM * N;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const int numThreads = (BM * BN) / (TM * TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    const int irA = threadIdx.x / (BK / 4), icA = threadIdx.x % (BK / 4);
    const int strideA = numThreads / (BK / 4);
    const int irB = threadIdx.x / (BN / 4), icB = threadIdx.x % (BN / 4);
    const int strideB = numThreads / (BN / 4);

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int bk = 0; bk < N; bk += BK) {
        for (int o = 0; o < BM; o += strideA) {
            float4 t = reinterpret_cast<const float4 *>(&A[(irA + o) * N + icA * 4])[0];
            As[(icA * 4 + 0) * BM + irA + o] = t.x;
            As[(icA * 4 + 1) * BM + irA + o] = t.y;
            As[(icA * 4 + 2) * BM + irA + o] = t.z;
            As[(icA * 4 + 3) * BM + irA + o] = t.w;
        }
        for (int o = 0; o < BK; o += strideB)
            reinterpret_cast<float4 *>(&Bs[(irB + o) * BN + icB * 4])[0] =
                reinterpret_cast<const float4 *>(&B[(irB + o) * N + icB * 4])[0];
        __syncthreads();
        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) regM[i] = As[k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i)
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
