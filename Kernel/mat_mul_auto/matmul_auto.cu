#include <cuda_runtime.h>
#include "../../Header/matmul_kernels.cuh"

// Auto-dispatcher: picks the best available kernel for the given shape.
//   - square, multiple of 128:
//       * K in [5120, 8192)  -> tuned (single-buffered wins at deep K)
//       * otherwise          -> doublebuffer (best general default)
//   - any other shape (rectangular / non-128-aligned / non-square)
//       -> boundary (fast float4 interior tiles + masked edges, runs any M,N,K)
// ROW-MAJOR: C(MxN) = A(MxK) * B(KxN).
void launch_matmul_auto(const float* A, const float* B, float* C, int M, int N, int K) {
    const bool squareAligned =
        (M == N && N == K) && (N % 128 == 0) && (N % 4 == 0);

    if (squareAligned) {
        constexpr int BM = 128, BN = 128, TM = 16, TN = 8;
        dim3 grid(N / BN, N / BM);
        dim3 block((BM * BN) / (TM * TN));   // 128 threads
        if (K >= 5120 && K < 8192)
            matmul_tuned<<<grid, block>>>(A, B, C, N);
        else
            matmul_doublebuffer<<<grid, block>>>(A, B, C, N);
        return;
    }
    // Everything else: the boundary kernel handles arbitrary M, N, K.
    launch_matmul_boundary(A, B, C, M, N, K);
}
