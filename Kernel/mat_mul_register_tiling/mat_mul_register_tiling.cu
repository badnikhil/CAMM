#include<cuda_runtime.h>

#include "../../Header/matmul_kernels.cuh"

#include<iomanip>

#define TILE_SIZE 64

#define COARSE_FACTOR 4

__global__ void matmul_register_tiling(const float *A, const float *B, float *C, const int N) {

    const int threads_per_block = TILE_SIZE * TILE_SIZE / (COARSE_FACTOR * COARSE_FACTOR);
    const int block_y = blockIdx.y;
    const int block_x = blockIdx.x;

    const int thread_id = threadIdx.x;

    // 1D -> 2D index mapping for shared load from A
    const int shared_A_row = thread_id / TILE_SIZE;
    const int shared_A_col = thread_id % TILE_SIZE;
    const int shared_A_stride = threads_per_block / TILE_SIZE;

    // 1D -> 2D index mapping for shared load from B
    const int shared_B_row = thread_id / TILE_SIZE;
    const int shared_B_col = thread_id % TILE_SIZE;
    const int shared_B_stride = threads_per_block / TILE_SIZE;

    const int C_row = COARSE_FACTOR * (thread_id / (TILE_SIZE / COARSE_FACTOR));
    const int C_col = COARSE_FACTOR * (thread_id % (TILE_SIZE / COARSE_FACTOR));

    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    float accum[COARSE_FACTOR][COARSE_FACTOR] = {0.0f};
    float reg_A[COARSE_FACTOR] = {0.0f};
    float reg_B[COARSE_FACTOR] = {0.0f};

    const int num_phases = ceil((float)N / TILE_SIZE);

    for (int phase = 0; phase < num_phases; phase++) {
        for (int offset = 0; offset < TILE_SIZE; offset += shared_A_stride) {
            if ((block_y * TILE_SIZE + offset + shared_A_row < N) && ((phase * TILE_SIZE + shared_A_col) < N))
                tile_A[offset + shared_A_row][shared_A_col] = A[(block_y * TILE_SIZE + offset + shared_A_row) * N + (phase * TILE_SIZE + shared_A_col)];
            else
                tile_A[offset + shared_A_row][shared_A_col] = 0.0f;
        }

        for (int offset = 0; offset < TILE_SIZE; offset += shared_B_stride) {
            if (((phase * TILE_SIZE + shared_B_row + offset) < N) && (block_x * TILE_SIZE + shared_B_col < N))
                tile_B[shared_B_row + offset][shared_B_col] = B[(phase * TILE_SIZE + shared_B_row + offset) * N + (block_x * TILE_SIZE + shared_B_col)];
            else
                tile_B[shared_B_row + offset][shared_B_col] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            for (int i = 0; i < COARSE_FACTOR; ++i)
                reg_A[i] = tile_A[C_row + i][k];

            for (int i = 0; i < COARSE_FACTOR; ++i)
                reg_B[i] = tile_B[k][C_col + i];

            for (int y = 0; y < COARSE_FACTOR; ++y) {
                for (int x = 0; x < COARSE_FACTOR; ++x)
                    accum[y][x] += reg_A[y] * reg_B[x];
            }
        }
        __syncthreads();
    }

    for (int y = 0; y < COARSE_FACTOR; ++y) {
        for (int x = 0; x < COARSE_FACTOR; x++) {
            if ((block_y * TILE_SIZE + C_row + y < N) && (block_x * TILE_SIZE + C_col + x < N))
                C[(block_y * TILE_SIZE + C_row + y) * N + (block_x * TILE_SIZE + C_col + x)] = accum[y][x];
        }
    } 
}
