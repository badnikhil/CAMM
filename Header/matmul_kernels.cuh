#ifndef MATMUL_KERNELS_H
#define MATMUL_KERNELS_H

// Naive matrix multiplication kernel declaration
__global__ void matmul_naive(const float* A, const float* B, float* C, int N);

// Coalesced matrix multiplication kernel declaration
__global__ void matmul_coalesced(const float* A, const float* B, float* C, int N);

// Shared Memory 
__global__ void matMulShared(float *A, float *B, float *C, int width);

// register tiling
__global__ void matmul_register_tiling(const float *A , const float *B ,  float *C , const int N);

#endif // MATMUL_KERNELS_H 