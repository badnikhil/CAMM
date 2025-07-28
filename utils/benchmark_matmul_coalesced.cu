#include <iostream>
#include <cuda_runtime.h>
#include "../Header/matmul_kernels.cuh"
#include <iomanip>
#include <stdexcept>

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
    double checksum = 0.0;
    for (int i = 0; i < n*n; ++i) checksum += C[i];
    std::cout << "Checksum CPU: " << checksum << std::endl;
}

int run_benchmark(int N) {
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        if (h_C) free(h_C);
        return 1;
    }
    random_fill(h_A, N*N);
    random_fill(h_B, N*N);
    

    checksum(h_A , h_B , h_C , N);
    
    // when accessing B matrix the memory around it is also copied..
    //  this will increase cache hits because for multiplying a(0.0)
    //  full row 0 of B will be faster loaded
    float *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for d_A: " << cudaGetErrorString(err) << std::endl;
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for d_B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Device memory allocation failed for d_C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Memcpy H2D failed for A: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Memcpy H2D failed for B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    int x = 16;
    int y = 16;
    int z = 1;
    dim3 threads(x, y, z);
    printf("dimensions of threads: %d, %d, %d\n", x, y, z);
    dim3 blocks(ceil(N/(float)(x)), ceil(N/(float)(y)) , 1);

    // CUDA events for timing
    cudaEvent_t start, stop, h2d_start, h2d_stop, d2h_start, d2h_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);

    // Host to Device timing
    cudaEventRecord(h2d_start);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);
    float h2d_ms = 0;
    cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);

    // Warmup kernel run
    matmul_coalesced<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Kernel timing (average over 10 runs)
    float kernel_ms = 0;
    int runs = 10;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start);
        matmul_coalesced<<<blocks, threads>>>(d_A, d_B, d_C, N);
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
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Memcpy D2H failed for C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
        return 1;
    }
    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);
    float d2h_ms = 0;
    cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop);

    std::cout << "C[0] = " << h_C[0] << std::endl;
    std::cout << "Host to Device memcpy time: " << h2d_ms << " ms" << std::endl;
    std::cout << "Kernel execution time (avg of " << runs << "): " << kernel_ms << " ms" << std::endl;
    std::cout << "Device to Host memcpy time: " << d2h_ms << " ms" << std::endl;

    // GFLOPS calculation
    double gflops = (2.0 * N * N * N) / (kernel_ms / 1000.0) / 1e9;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Kernel GFLOPS: " << gflops << std::endl;

    // Checksum calculation
    double checksum = 0.0;
    for (int i = 0; i < N*N; ++i) checksum += h_C[i];
    std::cout << "Checksum: " << checksum << std::endl<<std::endl << std::endl<<std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
} 


int main() {
    int sizes[] = {512, 1024, 2048, 3072, 4096, 6144, 8192};
    for (int N : sizes) {
        std::cout  << "Running benchmark for N = " << N << std::endl ;
        run_benchmark(N);
    }
    return 0;
}