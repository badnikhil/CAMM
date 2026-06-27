#include <cuda_runtime.h>
#include "../../Header/matmul_kernels.cuh"

// General matmul for ARBITRARY M, N, K. Same 2D register-blocking compute as the
// tuned kernel, but every global access is bounds-checked (out-of-range reads
// contribute 0, out-of-range writes are skipped), so any shape runs correctly on
// this path. Loads use a thread-linear sweep that handles partial edge tiles with
// no special cases. ROW-MAJOR: C(MxN) = A(MxK) * B(KxN).

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_general_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                      float* __restrict__ C, int M, int N, int K) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    __shared__ float As[BK * BM];  // transposed: As[k*BM + m]
    __shared__ float Bs[BK * BN];

    const int numThreads = (BM * BN) / (TM * TN);
    const int tRow = threadIdx.x / (BN / TN);
    const int tCol = threadIdx.x % (BN / TN);
    const int blockRow = cRow * BM, blockCol = cCol * BN;

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    const int A_elems = BM * BK;
    const int B_elems = BK * BN;

    for (int bk = 0; bk < K; bk += BK) {
        for (int idx = threadIdx.x; idx < A_elems; idx += numThreads) {
            int r = idx / BK, c = idx % BK;
            int gr = blockRow + r, gc = bk + c;
            As[c * BM + r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int idx = threadIdx.x; idx < B_elems; idx += numThreads) {
            int r = idx / BN, c = idx % BN;
            int gr = bk + r, gc = blockCol + c;
            Bs[r * BN + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) regM[i] = As[k * BM + tRow * TM + i];
            for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + tCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        int gr = blockRow + tRow * TM + i;
        if (gr >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int gc = blockCol + tCol * TN + j;
            if (gc < N) C[gr * N + gc] = acc[i * TN + j];
        }
    }
}

// Host launcher: small tile (64x64) when a dimension is tiny so all SMs fill,
// otherwise the 128x128 / 16x8 config.
void launch_matmul_general(const float* A, const float* B, float* C, int M, int N, int K) {
    if (M <= 256 || N <= 256) {
        constexpr int BM = 64, BN = 64, BK = 16, TM = 8, TN = 8;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block((BM * BN) / (TM * TN));
        matmul_general_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
    } else {
        constexpr int BM = 128, BN = 128, BK = 16, TM = 16, TN = 8;
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block((BM * BN) / (TM * TN));
        matmul_general_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
    }
}
