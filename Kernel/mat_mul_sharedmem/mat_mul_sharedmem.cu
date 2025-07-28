#include "../../Header/matmul_kernels.cuh"
#define TILE_WIDTH 16

__global__ void matMulShared(float *A, float *B, float *C, int N) {
    // Allocate shared memory for A and B tiles
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    // Thread and block coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Cvalue = 0.0;

    // Loop over tiles
    for (int ph = 0; ph < N / TILE_WIDTH; ++ph) {
        // Load one tile of A and B into shared memory
        ds_A[ty][tx] = A[Row * N + (ph * TILE_WIDTH + tx)];
        ds_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + Col];

        __syncthreads();

        // Multiply the two tiles
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    // Write result to global memory
    C[Row * N + Col] = Cvalue;
}
