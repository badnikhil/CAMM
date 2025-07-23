// Automatically uses CUDA's built-in variables
__device__
int getIndex1D() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__
int getIndex2D() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int width = blockDim.x * gridDim.x;
    return x + y * width;
}

__device__
int getIndex3D() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    int width  = blockDim.x * gridDim.x;
    int height = blockDim.y * gridDim.y;

    return x + y * width + z * width * height;
}


void fill(int *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = rand() % 100;
}