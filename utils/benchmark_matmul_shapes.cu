#include <iostream>
#include <cuda_runtime.h>
#include "../Header/matmul_kernels.cuh"
#include <iomanip>
#include <vector>

// Demonstrates the arbitrary-shape kernels (general + boundary) on a mix of
// square, rectangular, and non-128-aligned shapes that the tuned/doublebuffer
// kernels can't handle. ROW-MAJOR C(MxN) = A(MxK) * B(KxN).

void random_fill(float* a, size_t n) { for (size_t i = 0; i < n; ++i) a[i] = (float)rand() / RAND_MAX; }

// Relative Frobenius error vs a double-precision CPU reference (small shapes only).
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

void bench_shape(const char* which, int M, int N, int K, bool check) {
    size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
    float *hA = (float*)malloc(aN * 4), *hB = (float*)malloc(bN * 4), *hC = (float*)malloc(cN * 4);
    random_fill(hA, aN); random_fill(hB, bN);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, aN * 4); cudaMalloc(&dB, bN * 4); cudaMalloc(&dC, cN * 4);
    cudaMemcpy(dA, hA, aN * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bN * 4, cudaMemcpyHostToDevice);

    auto run = [&] {
        if (std::string(which) == "general") launch_matmul_general(dA, dB, dC, M, N, K);
        else                                 launch_matmul_boundary(dA, dB, dC, M, N, K);
    };

    run(); cudaDeviceSynchronize();  // warmup

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms = 0; int runs = 10;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(s); run(); cudaEventRecord(e); cudaEventSynchronize(e);
        float t = 0; cudaEventElapsedTime(&t, s, e); ms += t;
    }
    ms /= runs;
    double gf = (2.0 * M * N * K) / (ms / 1000.0) / 1e9;

    printf("[%-8s] %5dx%-5dx%-5d : %8.3f ms  %8.1f GFLOP/s", which, M, N, K, ms, gf);
    if (check) {
        cudaMemcpy(hC, dC, cN * 4, cudaMemcpyDeviceToHost);
        printf("   rel-Fro err = %.2e", rel_fro_err_cpu(hA, hB, hC, M, N, K));
    }
    printf("\n");

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
}

int main() {
    struct Shape { int M, N, K; bool check; };
    Shape shapes[] = {
        {512, 512, 512, true},        // aligned square
        {300, 300, 300, true},        // small non-aligned square
        {1000, 1000, 1000, true},     // non-aligned square
        {2000, 2000, 2000, false},    // large non-aligned (N,K %4==0 -> fast interior)
        {1024, 4096, 2048, false},    // rectangular
        {2048, 2048, 256, false},     // thin-K
        {1500, 900, 700, false},      // odd rectangular
        {4097, 4097, 4097, false},    // fully non-aligned (scalar edge path)
    };
    std::cout << "=== general kernel ===" << std::endl;
    for (auto& s : shapes) bench_shape("general", s.M, s.N, s.K, s.check);
    std::cout << "\n=== boundary kernel ===" << std::endl;
    for (auto& s : shapes) bench_shape("boundary", s.M, s.N, s.K, s.check);
    return 0;
}
