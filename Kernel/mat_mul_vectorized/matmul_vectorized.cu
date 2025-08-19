#include<cuda_runtime.h>

#include "../../Header/matmul_kernels.cuh"



#define COARSE_FACTOR 4

#define TILE_SIZE 64

__global__ void matmul_vectorized(float *A, float *B, float *C, int N)
{
    // Number of threads per block
    const int n_threads_per_block = TILE_SIZE * TILE_SIZE / (COARSE_FACTOR*COARSE_FACTOR);
    static_assert(n_threads_per_block % TILE_SIZE == 0);
    static_assert(TILE_SIZE % 4 == 0);

    // Details regarding this thread
    const int by = blockIdx.y;
    const int bx = blockIdx.x; 

    const int tx = threadIdx.x;

    // 1D -> 2D while loading A
    const int A_view_ty = tx / (TILE_SIZE / 4);
    const int A_view_tx = tx % (TILE_SIZE / 4);
    const int stride_A = n_threads_per_block/(TILE_SIZE/4);

    // 1D -> 2D while loading B
    const int B_view_ty = tx / (TILE_SIZE / 4);
    const int B_view_tx = tx % (TILE_SIZE / 4);
    const int stride_B = n_threads_per_block/(TILE_SIZE / 4);

    // Working on C[row, col]
    const int row = COARSE_FACTOR * (tx / (TILE_SIZE/COARSE_FACTOR));
    const int col = COARSE_FACTOR * (tx % (TILE_SIZE/COARSE_FACTOR));
    
    // Allocating shared memory
    __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
    __shared__ float sh_B[TILE_SIZE][TILE_SIZE];

    // Parallel mat mul
    float value[COARSE_FACTOR][COARSE_FACTOR] = {0.0f};
    float register_A[COARSE_FACTOR] = {0.0f};
    float register_B[COARSE_FACTOR] = {0.0f};

    // Phases
    const int phases = ceil((float)N/TILE_SIZE);

    for (int phase = 0; phase < phases; phase++)
    {
        // Load Tiles into shared memory
        for (int load_offset = 0; load_offset < TILE_SIZE; load_offset+=stride_A)
        {
            if ((by*TILE_SIZE + load_offset+A_view_ty < N) && (((phase*TILE_SIZE+A_view_tx*4)) < N))
            {
                float4 A_tmp = reinterpret_cast<float4 *>(&A[(by*TILE_SIZE + load_offset+A_view_ty)*N + ((phase*TILE_SIZE+A_view_tx*4))])[0];
                sh_A[A_view_tx*4+0][load_offset+A_view_ty] = A_tmp.x;
                sh_A[A_view_tx*4+1][load_offset+A_view_ty] = A_tmp.y;
                sh_A[A_view_tx*4+2][load_offset+A_view_ty] = A_tmp.z;
                sh_A[A_view_tx*4+3][load_offset+A_view_ty] = A_tmp.w;
            }
            else
            {
                sh_A[A_view_tx*4+0][load_offset+A_view_ty] = 0.0f;
                sh_A[A_view_tx*4+1][load_offset+A_view_ty] = 0.0f;
                sh_A[A_view_tx*4+2][load_offset+A_view_ty] = 0.0f;
                sh_A[A_view_tx*4+3][load_offset+A_view_ty] = 0.0f;
            }
            
        }
        
        for (int load_offset = 0; load_offset < TILE_SIZE; load_offset+=stride_B)
        {
            if (((phase*TILE_SIZE + B_view_ty+load_offset) < N) && (((bx*TILE_SIZE + B_view_tx*4)) < N))
            {
                float4 B_tmp = reinterpret_cast<float4 *>(&B[(phase*TILE_SIZE + B_view_ty+load_offset)*N + ((bx*TILE_SIZE + B_view_tx*4))])[0];
                sh_B[B_view_ty+load_offset][B_view_tx*4+0] = B_tmp.x;
                sh_B[B_view_ty+load_offset][B_view_tx*4+1] = B_tmp.y;
                sh_B[B_view_ty+load_offset][B_view_tx*4+2] = B_tmp.z;
                sh_B[B_view_ty+load_offset][B_view_tx*4+3] = B_tmp.w;
            }
            else
            {
                sh_B[B_view_ty+load_offset][B_view_tx*4+0] = 0.0f;
                sh_B[B_view_ty+load_offset][B_view_tx*4+1] = 0.0f;
                sh_B[B_view_ty+load_offset][B_view_tx*4+2] = 0.0f;
                sh_B[B_view_ty+load_offset][B_view_tx*4+3] = 0.0f;
            }
            
        }
        __syncthreads();

        // calculate per-thread results
        for (int k = 0; k < TILE_SIZE; ++k) 
        {
            // block into registers
            for (int i = 0; i < COARSE_FACTOR; ++i)
                register_A[i] = sh_A[k][row+i];
            
            for (int i = 0; i < COARSE_FACTOR; ++i)
                register_B[i] = sh_B[k][col+i];
            
            for (int cy = 0; cy < COARSE_FACTOR; ++cy) 
            {
                for (int cx = 0; cx < COARSE_FACTOR; ++cx) 
                    value[cy][cx] += register_A[cy] * register_B[cx];
            }
        }
        __syncthreads();
    }

    // Assigning calculated value
    for (int cy = 0; cy < COARSE_FACTOR; ++cy)
    {
        for (int cx = 0; cx < COARSE_FACTOR; cx++)
        {
            if ((by*TILE_SIZE+row+cy < N) && (bx*TILE_SIZE+col+cx < N))
                C[(by*TILE_SIZE+row+cy)*N + (bx*TILE_SIZE+col+cx)] = 1*value[cy][cx] + 0*C[(by*TILE_SIZE+row+cy)*N + (bx*TILE_SIZE+col+cx)];
        }
    } 
}