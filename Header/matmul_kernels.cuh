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
// vectorized access
__global__ void matmul_vectorized(float *A, float *B, float *C, int N);

// autotuned 128x128 / BK16 / 16x8 register tile
__global__ void matmul_tuned(const float *A, const float *B, float *C, int N);

// double-buffered (software-pipelined) tuned kernel
__global__ void matmul_doublebuffer(const float *A, const float *B, float *C, int N);

// --- arbitrary-shape kernels: host launchers, C(MxN)=A(MxK)*B(KxN) ---
// general: bounds-checked, runs any M,N,K correctly
void launch_matmul_general(const float *A, const float *B, float *C, int M, int N, int K);
// boundary: fast float4 interior tiles + masked edges, any M,N,K
void launch_matmul_boundary(const float *A, const float *B, float *C, int M, int N, int K);
#endif // MATMUL_KERNELS_H