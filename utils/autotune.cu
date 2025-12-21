// Autotuner: sweeps register-tile geometries of the vectorized/tuned kernel and
// reports GFLOPS and % of cuBLAS at the given square sizes, to find the best
// (BM,BN,BK,TM,TN) config for the current GPU. The asymmetric 16x8 tile is the
// winner on Ampere (sm_86); re-run on a different GPU to re-tune.
//
//   build:  nvcc -O3 -arch=sm_86 -o autotune utils/autotune.cu -lcublas
//   run:    ./autotune 2048 4096        (defaults to 2048 4096 if no args)
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Templated tuned kernel (square N, multiple of the tile). Transposed shared A,
// float4 vectorized loads/stores.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void tuned_cfg(const float* A, const float* B, float* C, int N) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];
    A += cRow * BM * N; B += cCol * BN; C += cRow * BM * N + cCol * BN;
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
            float4 t = reinterpret_cast<const float4*>(&A[(irA + o) * N + icA * 4])[0];
            As[(icA * 4 + 0) * BM + irA + o] = t.x;
            As[(icA * 4 + 1) * BM + irA + o] = t.y;
            As[(icA * 4 + 2) * BM + irA + o] = t.z;
            As[(icA * 4 + 3) * BM + irA + o] = t.w;
        }
        for (int o = 0; o < BK; o += strideB)
            reinterpret_cast<float4*>(&Bs[(irB + o) * BN + icB * 4])[0] =
                reinterpret_cast<const float4*>(&B[(irB + o) * N + icB * 4])[0];
        __syncthreads();
        A += BK; B += BK * N;
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) regM[i] = As[k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; j += 4) {
            float4 v;
            v.x = acc[i * TN + j + 0]; v.y = acc[i * TN + j + 1];
            v.z = acc[i * TN + j + 2]; v.w = acc[i * TN + j + 3];
            reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = v;
        }
}

static cublasHandle_t g_h;
static float time_kernel(std::function<void()> launch, int warm = 5, int runs = 20) {
    for (int i = 0; i < warm; ++i) launch();
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < runs; ++i) launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / runs;
}

template <int BM, int BN, int BK, int TM, int TN>
void try_cfg(const float* dA, const float* dB, float* dC, int N, double g_cublas) {
    const int numThreads = (BM * BN) / (TM * TN);
    dim3 grid(N / BN, N / BM), block(numThreads);
    auto fn = [&] { tuned_cfg<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, N); };
    fn();
    if (cudaGetLastError() != cudaSuccess) {
        printf("  %3dx%-3d BK%-2d %dx%-2d : launch fail\n", BM, BN, BK, TM, TN);
        cudaGetLastError(); return;
    }
    cudaDeviceSynchronize();
    float ms = time_kernel(fn);
    double g = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;
    printf("  %3dx%-3d BK%-2d %dx%-2d nT%-4d : %8.1f GFLOP/s  %6.1f%% cuBLAS\n",
           BM, BN, BK, TM, TN, numThreads, g, 100.0 * g / g_cublas);
}

#define TRY(bm, bn, bk, tm, tn) try_cfg<bm, bn, bk, tm, tn>(dA, dB, dC, N, g_cublas)

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {2048, 4096};

    cublasCreate(&g_h);
    cublasSetMathMode(g_h, CUBLAS_DEFAULT_MATH);

    for (int N : sizes) {
        size_t bytes = (size_t)N * N * 4;
        std::vector<float> hA(N * N), hB(N * N);
        for (auto& x : hA) x = (float)rand() / RAND_MAX;
        for (auto& x : hB) x = (float)rand() / RAND_MAX;
        float *dA, *dB, *dC;
        cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
        cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

        const float alpha = 1.f, beta = 0.f;
        auto cub = [&] {
            cublasSgemm(g_h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
        };
        cub(); cudaDeviceSynchronize();
        float ms = time_kernel(cub);
        double g_cublas = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;

        printf("===== N=%d  (cuBLAS %.1f GFLOP/s) =====\n", N, g_cublas);
        TRY(128, 128, 8, 8, 8);
        TRY(128, 128, 16, 8, 8);
        TRY(128, 128, 16, 8, 16);
        TRY(128, 128, 16, 16, 8);   // the winner
        TRY(128, 128, 32, 8, 8);
        TRY(128, 64, 8, 8, 8);
        TRY(64, 64, 8, 8, 8);
        TRY(256, 128, 16, 8, 8);
        TRY(128, 256, 16, 8, 8);
        printf("\n");

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    cublasDestroy(g_h);
    return 0;
}
