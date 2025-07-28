#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

// CUTLASS
#include "cutlass/gemm/device/gemm.h"

void random_fill(float* arr, int n) {
    for (int i = 0; i < n; ++i) arr[i] = static_cast<float>(rand()) / RAND_MAX;
}

void checksum(const float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    double cs = 0.0;
    for (int i = 0; i < n*n; ++i) cs += C[i];
    std::cout << "Checksum CPU: " << cs << std::endl;
}

int run_benchmark(int N) {
    using Gemm = cutlass::gemm::device::Gemm<float,                             // A element type
                                            cutlass::layout::RowMajor,         // A layout
                                            float,                             // B element type
                                            cutlass::layout::RowMajor,         // B layout
                                            float,                             // C element type
                                            cutlass::layout::RowMajor>;        // C layout

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    float *h_A = (float*)aligned_alloc(16, bytes);
    float *h_B = (float*)aligned_alloc(16, bytes);
    float *h_C = (float*)aligned_alloc(16, bytes);

    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        return 1;
    }

    random_fill(h_A, N*N);
    random_fill(h_B, N*N);

    checksum(h_A, h_B, h_C, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // CUTLASS GEMM arguments
    typename Gemm::Arguments args({N, N, N},          // problem size: m,n,k
                                   {d_A, N},          // A ptr, lda
                                   {d_B, N},          // B ptr, ldb
                                   {d_C, N},          // C ptr, ldc (beta * C)
                                   {d_C, N},          // D ptr, ldd (output)
                                   {1.0f, 0.0f});     // alpha, beta

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM cannot implement problem size" << std::endl;
        return 1;
    }

    // Warm-up
    gemm_op(args);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int runs = 10;
    float total_ms = 0.f;
    for(int i=0;i<runs;++i){
        cudaEventRecord(start);
        gemm_op(args);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms=0.f; cudaEventElapsedTime(&ms,start,stop);
        total_ms += ms;
    }
    float avg_ms = total_ms / runs;

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    std::cout << "CUTLASS execution time (avg of "<<runs<<"): "<< avg_ms << " ms" << std::endl;
    double gflops = (2.0 * N * N * N) / (avg_ms / 1000.0) / 1e9;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Kernel GFLOPS: " << gflops << std::endl;

    double cs = 0.0; for (int i = 0; i < N*N; ++i) cs += h_C[i];
    std::cout << "Checksum GPU: " << cs << std::endl << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

int main(){
    int sizes[] = {128,512,1024,2048,3072,4096};
    for(int N: sizes){
        std::cout << "Running CUTLASS benchmark N="<<N<< std::endl;
        run_benchmark(N);
    }
    return 0;
}
