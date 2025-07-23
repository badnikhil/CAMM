#include <stdio.h>
#define N 10
#define TILE_WIDTH 5

__global__ void tiledMatMul(int *C, int *A, int *B) {
    __shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int temp = 0;

    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < N && t * TILE_WIDTH + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if (col < N && t * TILE_WIDTH + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            temp += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = temp;
}

int main() {
    int a[N][N], b[N][N], c[N][N];

    // Initialize matrices a and b
    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
        a[i][j] = j + 1;
        b[i][j] = j + 11;
    }

    int *aptr, *bptr, *cptr;
    cudaMalloc(&aptr, N * N * sizeof(int));
    cudaMalloc(&bptr, N * N * sizeof(int));
    cudaMalloc(&cptr, N * N * sizeof(int));

    cudaMemcpy(aptr, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bptr, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    tiledMatMul<<<blocksPerGrid, threadsPerBlock>>>(cptr, aptr, bptr);
    cudaDeviceSynchronize();

    cudaMemcpy(c, cptr, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        printf("%d\n", c[i][j]);

    cudaFree(aptr);
    cudaFree(bptr);
    cudaFree(cptr);

    return 0;
}
