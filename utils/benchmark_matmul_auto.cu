#include <iostream>
#include <cuda_runtime.h>
#include "../Header/matmul_kernels.cuh"
#include <iomanip>
#include <vector>
#include <string>

// Benchmarks launch_matmul_auto, which auto-selects the best kernel per shape.
//   ./auto              -> default square sweep (128 .. 8192)
//   ./auto 4096         -> single square N=4096
//   ./auto 1024 4096 2048  -> single shape M=1024 N=4096 K=2048 (rectangular)
// ROW-MAJOR: C(MxN) = A(MxK) * B(KxN).

void random_fill(float* a, size_t n) { for (size_t i = 0; i < n; ++i) a[i] = (float)rand() / RAND_MAX; }

double rel_fro_err_cpu(const float* A, const float* B, const float* C, int M, int N, int K) {
    std::vector<double> ref((size_t)M * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += (double)A[i * K + k] * B[k * N + j];
            ref[(size_t)i * N + j] = s;
        }
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        double d = (double)C[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
    }
    return sqrt(num / (den > 0 ? den : 1));
}

void bench(int M, int N, int K) {
    size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
    float *hA = (float*)malloc(aN * 4), *hB = (float*)malloc(bN * 4), *hC = (float*)malloc(cN * 4);
    random_fill(hA, aN); random_fill(hB, bN);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, aN * 4); cudaMalloc(&dB, bN * 4); cudaMalloc(&dC, cN * 4);
    cudaMemcpy(dA, hA, aN * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bN * 4, cudaMemcpyHostToDevice);

    launch_matmul_auto(dA, dB, dC, M, N, K); cudaDeviceSynchronize();  // warmup

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms = 0; int runs = 10;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(s); launch_matmul_auto(dA, dB, dC, M, N, K);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float t = 0; cudaEventElapsedTime(&t, s, e); ms += t;
    }
    ms /= runs;
    double gf = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;

    printf("auto  %5dx%-5dx%-5d : %8.3f ms  %8.1f GFLOP/s", M, N, K, ms, gf);
    if ((size_t)M * N * K <= (size_t)1024 * 1024 * 1024) {   // correctness check on small shapes
        cudaMemcpy(hC, dC, cN * 4, cudaMemcpyDeviceToHost);
        printf("   rel-Fro err = %.2e", rel_fro_err_cpu(hA, hB, hC, M, N, K));
    }
    printf("\n");

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
}

int main(int argc, char** argv) {
    if (argc == 2) {                       // single square N
        int N = atoi(argv[1]); bench(N, N, N);
    } else if (argc == 4) {                // explicit M N K
        bench(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
    } else {                               // default sweep
        int sizes[] = {128, 1024, 2048, 3072, 4096, 6144, 8192};
        for (int N : sizes) bench(N, N, N);
    }
    return 0;
}
