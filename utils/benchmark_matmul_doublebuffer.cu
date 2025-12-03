#include <iostream>
#include <cuda_runtime.h>
#include "../Header/matmul_kernels.cuh"
#include <iomanip>
#include <vector>

#define BM 128
#define BN 128
#define TM 16
#define TN 8

void random_fill(float* arr, int n) {
    for (int i = 0; i < n; ++i) arr[i] = static_cast<float>(rand()) / RAND_MAX;
}

void checksum(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    double checksum_val = 0.0;
    for (int i = 0; i < n * n; ++i) checksum_val += C[i];
    printf(" C[0] =  %f\n", C[0]);
    std::cout << "Checksum CPU: " << std::fixed << std::setprecision(8) << checksum_val << std::endl;
}

int run_benchmark(int N) {
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        return 1;
    }

    random_fill(h_A, N * N);
    random_fill(h_B, N * N);
    if (N <= 2048) checksum(h_A, h_B, h_C_cpu, N);  // CPU ref only at small N

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads((BM * BN) / (TM * TN));        // 128
    dim3 blocks(N / BN, N / BM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    matmul_doublebuffer<<<blocks, threads>>>(d_A, d_B, d_C, N);  // warmup
    cudaDeviceSynchronize();

    float kernel_ms = 0;
    int runs = 10;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start);
        matmul_doublebuffer<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        kernel_ms += ms;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }
    }
    kernel_ms /= runs;

    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Kernel execution time (avg of " << runs << "): " << kernel_ms << " ms" << std::endl;
    double gflops = (2.0 * N * N * N) / (kernel_ms / 1000.0) / 1e9;
    std::cout << "Kernel GFLOPS: " << gflops << std::endl;

    double checksum_gpu = 0.0;
    for (int i = 0; i < N * N; ++i) checksum_gpu += h_C_gpu[i];
    std::cout << "Checksum GPU: " << checksum_gpu << std::endl << std::endl << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    return 0;
}

int main() {
    int sizes[] = {128, 1024, 2048, 3072, 4096, 6144, 8192};
    for (int N : sizes) {
        std::cout << "Running benchmark for N = " << N << std::endl;
        run_benchmark(N);
    }
    return 0;
}
