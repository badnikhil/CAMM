#include <iostream>
#include <cuda_runtime.h>
#include "../Header/matmul_kernels.cuh"
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <algorithm>

#define TILE_SIZE 64
#define COARSE_FACTOR 4

void random_fill(float* arr, int n) {
    for (int i = 0; i < n; ++i) arr[i] = static_cast<float>(rand()) / RAND_MAX;
}

// Function to transpose a matrix
void transpose_matrix(const float* input, float* output, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            output[j * n + i] = input[i * n + j];
        }
    }
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
    double checksum_val = 0.0;
    for (int i = 0; i < n*n; ++i) checksum_val += C[i];
    printf(" C[0] =  %f\n", C[0]);
    std::cout << "Checksum CPU: " << std::fixed << std::setprecision(8) << checksum_val << std::endl;
}

int run_benchmark(int N) {
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_A_T = (float*)malloc(bytes); // For transposed A
    float *h_B = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes); // Renamed to avoid confusion
    float *h_C_cpu = (float*)malloc(bytes); // For CPU result

    if (!h_A || !h_A_T || !h_B || !h_C_gpu || !h_C_cpu) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        if (h_A) free(h_A);
        if (h_A_T) free(h_A_T);
        if (h_B) free(h_B);
        if (h_C_gpu) free(h_C_gpu);
        if (h_C_cpu) free(h_C_cpu);
        return 1;
    }
    
    random_fill(h_A, N*N);
    random_fill(h_B, N*N);
    
    // Transpose matrix A on the CPU
    transpose_matrix(h_A, h_A_T, N);
    
    // Calculate checksum on CPU using original matrices
    checksum(h_A, h_B, h_C_cpu, N);

    float *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for d_A: " << cudaGetErrorString(err) << std::endl;
        // cleanup...
        return 1;
    }
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for d_B: " << cudaGetErrorString(err) << std::endl;
        // cleanup...
        return 1;
    }
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for d_C: " << cudaGetErrorString(err) << std::endl;
        // cleanup...
        return 1;
    }

    dim3 threads(TILE_SIZE * TILE_SIZE / (COARSE_FACTOR * COARSE_FACTOR));
    dim3 blocks(ceil((float)N / TILE_SIZE), ceil((float)N / TILE_SIZE));

    cudaEvent_t start, stop, h2d_start, h2d_stop, d2h_start, d2h_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);

    // Host to Device timing
    cudaEventRecord(h2d_start);
    // **Copy TRANSPOSED A to device**
    cudaMemcpy(d_A, h_A_T, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);
    float h2d_ms = 0;
    cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);

    // Warmup kernel run
    matmul_vectorized<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Kernel timing (average over 10 runs)
    float kernel_ms = 0;
    int runs = 10;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start);
        matmul_vectorized<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        kernel_ms += ms;
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }
    }
    kernel_ms /= runs;

    // Device to Host timing
    cudaEventRecord(d2h_start);
    err = cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Memcpy D2H failed for C: " << cudaGetErrorString(err) << std::endl;
        // cleanup...
        return 1;
    }
    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);
    float d2h_ms = 0;
    cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop);

    std::cout << "Host to Device memcpy time: " << h2d_ms << " ms" << std::endl;
    std::cout << "Kernel execution time (avg of " << runs << "): " << kernel_ms << " ms" << std::endl;
    std::cout << "Device to Host memcpy time: " << d2h_ms << " ms" << std::endl;

    double gflops = (2.0 * N * N * N) / (kernel_ms / 1000.0) / 1e9;
    std::cout << "Kernel GFLOPS: " << gflops << std::endl;

    // GPU Checksum calculation
    double checksum_gpu = 0.0;
    for (int i = 0; i < N*N; ++i) checksum_gpu += h_C_gpu[i];
    std::cout << "Checksum GPU: " << checksum_gpu << std::endl << std::endl << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_A_T); free(h_B); free(h_C_gpu); free(h_C_cpu);
    return 0;
} 

int main() {
    int sizes[] = {128, 1024, 2048, 3072, 4096, 6144, 8192};
    for (int N : sizes) {
        std::cout  << "Running benchmark for N = " << N << std::endl;
        run_benchmark(N);
    }
    return 0;
}