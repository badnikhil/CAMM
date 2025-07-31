#include<cuda_runtime.h>
#include "../../Header/matmul_kernels.cuh"
#include<iomanip>


__global__ void matmul_register_tiling(const float *A, const float *B, float *C, const int N) {
    // Define the block and thread indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Define the number of registers to use for tiling
    const int REG_TILE_SIZE = 4;

    // Define the shared memory size
    const int SHARED_MEM_SIZE = 16;

    // Declare the shared memory
    __shared__ float sharedA[SHARED_MEM_SIZE][SHARED_MEM_SIZE];
    __shared__ float sharedB[SHARED_MEM_SIZE][SHARED_MEM_SIZE];

    // Initialize the registers for the A and B matrices
    float regA[REG_TILE_SIZE];
    float regB[REG_TILE_SIZE];

    // Initialize the register for the C matrix
    float regC = 0.0f;

    // Calculate the global thread indices
    int globalRow = blockRow * blockDim.y + row;
    int globalCol = blockCol * blockDim.x + col;

    // Check if the thread is within the matrix bounds
    if (globalRow < N && globalCol < N) {
        // Loop through the tiles
        for (int tile = 0; tile < N / SHARED_MEM_SIZE; tile++) {
            // Load the A and B matrices into shared memory
            if (row < SHARED_MEM_SIZE && col < SHARED_MEM_SIZE) {
                sharedA[row][col] = A[globalRow * N + tile * SHARED_MEM_SIZE + col];
                sharedB[row][col] = B[(tile * SHARED_MEM_SIZE + row) * N + globalCol];
            }

            // Synchronize the threads
            __syncthreads();

            // Load the A and B matrices into registers
            for (int i = 0; i < REG_TILE_SIZE; i++) {
                regA[i] = sharedA[row][i];
                regB[i] = sharedB[i][col];
            }

            // Perform the matrix multiplication using the registers
            for (int i = 0; i < REG_TILE_SIZE; i++) {
                regC += regA[i] * regB[i];
            }
        }

        // Store the result in the C matrix
        C[globalRow * N + globalCol] = regC;
    }
}