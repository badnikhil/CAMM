#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../Header/matmul_kernels.cuh"
#include <iomanip>
#include <vector>

// Side-by-side comparison: cuBLAS (pure FP32, default math) vs our auto kernel
// picker, on the same square sizes as benchmark_matmul_cublas.cu. Prints each
// kernel's time + GFLOPS, then a comparison row (% of cuBLAS, WIN/loss).
//
//   build: nvcc -O3 -arch=sm_86 -o compare utils/benchmark_compare.cu \
//                Kernel/mat_mul_auto/*.cu Kernel/mat_mul_tuned/*.cu \
//                Kernel/mat_mul_doublebuffer/*.cu Kernel/mat_mul_boundary/*.cu -lcublas
//   run:   ./compare
//
// NOTE: CPU checksum verification is commented out (too slow on CPU at large N;
// correctness already validated separately at small sizes).

void random_fill(float* arr, size_t n) {
    for (size_t i = 0; i < n; ++i) arr[i] = static_cast<float>(rand()) / RAND_MAX;
}

/*  // CPU checksum — disabled (O(N^3) on CPU is far too slow at large N).
void checksum(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    double cs = 0.0;
    for (int i = 0; i < n * n; ++i) cs += C[i];
    std::cout << "Checksum CPU: " << cs << std::endl;
}
*/

struct Result { double cublas_gflops; double auto_gflops; };

Result run_benchmark(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    if (!h_A || !h_B) { std::cerr << "Host alloc failed!\n"; return {0, 0}; }
    random_fill(h_A, (size_t)N * N);
    random_fill(h_B, (size_t)N * N);

    // checksum(h_A, h_B, h_C, N);   // disabled (slow CPU reference)

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);   // pure FP32, no TF32
    const float alpha = 1.0f, beta = 0.0f;

    auto run_cublas = [&] {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    &alpha, d_B, N, d_A, N, &beta, d_C, N);
    };
    auto run_auto = [&] { launch_matmul_auto(d_A, d_B, d_C, N, N, N); };

    // Light warmup only: get past cold-start clocks WITHOUT cooking the GPU.
    // (A heavy warmup throttles this 55W card before timing even begins.)
    for (int i = 0; i < 3; ++i) { run_cublas(); run_auto(); }
    cudaDeviceSynchronize();

    // Interleaved best-of-N: alternate cuBLAS and auto so both see the same
    // thermal state, and keep each one's FASTEST (least-throttled, peak-clock)
    // time. Keep N modest so sustained heat doesn't dominate the samples.
    // NOTE: on a power-capped laptop GPU the margin is within thermal noise; for
    // a clean comparison lock the clock first:  sudo nvidia-smi -lgc 1500,1500
    const int runs = 12;
    float cublas_best = 1e30f, auto_best = 1e30f;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start); run_cublas(); cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0; cudaEventElapsedTime(&ms, start, stop);
        if (ms < cublas_best) cublas_best = ms;

        cudaEventRecord(start); run_auto(); cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        ms = 0; cudaEventElapsedTime(&ms, start, stop);
        if (ms < auto_best) auto_best = ms;
    }
    cublasDestroy(handle);

    double cublas_gflops = (2.0 * N * N * N) / (cublas_best / 1000.0) / 1e9;
    double auto_gflops   = (2.0 * N * N * N) / (auto_best / 1000.0) / 1e9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  cuBLAS : " << std::setw(9) << cublas_best << " ms   "
              << std::setw(9) << cublas_gflops << " GFLOP/s" << std::endl;
    std::cout << "  auto   : " << std::setw(9) << auto_best << " ms   "
              << std::setw(9) << auto_gflops << " GFLOP/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
    return {cublas_gflops, auto_gflops};
}

int main() {
    int sizes[] = {128, 512, 1024, 2048, 3072, 4096, 6144, 8192};
    const int nSizes = sizeof(sizes) / sizeof(int);
    std::vector<Result> results(nSizes);

    for (int i = 0; i < nSizes; ++i) {
        std::cout << "Running benchmark for N = " << sizes[i] << std::endl;
        results[i] = run_benchmark(sizes[i]);
        std::cout << std::endl;
    }

    // ---------------- comparison table ----------------
    std::cout << "==================== cuBLAS vs auto (FP32) ====================" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    int wins = 0;
    for (int i = 0; i < nSizes; ++i) {
        double c = results[i].cublas_gflops, a = results[i].auto_gflops;
        double pct = (c > 0) ? 100.0 * (a / c - 1.0) : 0.0;
        bool win = a > c;
        if (win) wins++;
        std::cout << "  N=" << std::setw(5) << sizes[i]
                  << "  cuBLAS " << std::setw(8) << c
                  << "  | auto " << std::setw(8) << a
                  << "  | " << std::showpos << std::setw(6) << pct << "%" << std::noshowpos
                  << "  " << (win ? "WIN" : "loss") << std::endl;
    }
    std::cout << "  ---- auto beats cuBLAS at " << wins << " / " << nSizes << " sizes ----" << std::endl;
    std::cout << "===============================================================" << std::endl;
    return 0;
}
