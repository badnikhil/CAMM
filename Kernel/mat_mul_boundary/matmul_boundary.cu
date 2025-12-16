#include <cuda_runtime.h>
#include "../../Header/matmul_kernels.cuh"

// Boundary kernel: the double-buffered float4 winner with predicated global
// boundary handling. Runs directly on ARBITRARY M, N, K with no padding buffers
// and no extra copies. Interior 128x128 tiles take the fast float4 double-buffered
// path; only the partial edge row/col tiles and the K-remainder take a scalar
// bounds-checked path. ROW-MAJOR: C(MxN) = A(MxK) * B(KxN). As transposed As[k*BM+m].
//
// ALIGNED=true  : every block is interior (M,N exact tile multiples; K%BK==0;
//                 N%4==0; K%4==0) -> pure fast path.
// ALIGNED=false : interior blocks still take the float4 path at runtime; edge
//                 blocks take the scalar masked path. vecOK = (N%4==0 && K%4==0).

template <int BM, int BN, int BK, int TM, int TN, bool ALIGNED>
__global__ void matmul_boundary_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                       float* __restrict__ C, int M, int N, int K, int vecOK) {
    const int cRow = blockIdx.y, cCol = blockIdx.x;
    const int blockRow = cRow * BM, blockCol = cCol * BN;

    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    const int numThreads = (BM * BN) / (TM * TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int threadCol = threadIdx.x % (BN / TN);

    const int irA = threadIdx.x / (BK / 4), icA = threadIdx.x % (BK / 4);
    const int strideA = numThreads / (BK / 4);
    const int irB = threadIdx.x / (BN / 4), icB = threadIdx.x % (BN / 4);
    const int strideB = numThreads / (BN / 4);
    const int nA = BM / strideA;
    const int nB = BK / strideB;

    const int A_elems = BM * BK;
    const int B_elems = BK * BN;

    const bool fullM = (blockRow + BM) <= M;
    const bool fullN = (blockCol + BN) <= N;
    const bool interior = ALIGNED || (fullM && fullN && vecOK);

    const int numFull = K / BK;
    const int kRem = K - numFull * BK;

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    const float* Ai = A + (size_t)blockRow * K;
    const float* Bi = B + blockCol;

    // ===================== INTERIOR (fast float4) =====================
    if (interior) {
        const float* Ap = Ai;
        const float* Bp = Bi;
        for (int o = 0; o < BM; o += strideA) {
            float4 t = reinterpret_cast<const float4*>(&Ap[(irA + o) * K + icA * 4])[0];
            As[0][(icA * 4 + 0) * BM + irA + o] = t.x;
            As[0][(icA * 4 + 1) * BM + irA + o] = t.y;
            As[0][(icA * 4 + 2) * BM + irA + o] = t.z;
            As[0][(icA * 4 + 3) * BM + irA + o] = t.w;
        }
        for (int o = 0; o < BK; o += strideB)
            reinterpret_cast<float4*>(&Bs[0][(irB + o) * BN + icB * 4])[0] =
                reinterpret_cast<const float4*>(&Bp[(irB + o) * N + icB * 4])[0];
        __syncthreads();
        Ap += BK;
        Bp += BK * N;

        int buf = 0;
        for (int t = 0; t < numFull; ++t) {
            const bool has_next = (t + 1 < numFull);
            float4 rA[8];   // nA <= 8 for the configs used here
            float4 rB[8];   // nB <= 8
            if (has_next) {
                for (int o = 0, idx = 0; o < BM; o += strideA, ++idx)
                    rA[idx] = reinterpret_cast<const float4*>(&Ap[(irA + o) * K + icA * 4])[0];
                for (int o = 0, idx = 0; o < BK; o += strideB, ++idx)
                    rB[idx] = reinterpret_cast<const float4*>(&Bp[(irB + o) * N + icB * 4])[0];
                Ap += BK;
                Bp += BK * N;
            }
            for (int k = 0; k < BK; ++k) {
                for (int i = 0; i < TM; ++i) regM[i] = As[buf][k * BM + threadRow * TM + i];
                for (int j = 0; j < TN; ++j) regN[j] = Bs[buf][k * BN + threadCol * TN + j];
                for (int i = 0; i < TM; ++i)
                    for (int j = 0; j < TN; ++j)
                        acc[i * TN + j] += regM[i] * regN[j];
            }
            if (has_next) {
                const int nbuf = buf ^ 1;
                for (int o = 0, idx = 0; o < BM; o += strideA, ++idx) {
                    As[nbuf][(icA * 4 + 0) * BM + irA + o] = rA[idx].x;
                    As[nbuf][(icA * 4 + 1) * BM + irA + o] = rA[idx].y;
                    As[nbuf][(icA * 4 + 2) * BM + irA + o] = rA[idx].z;
                    As[nbuf][(icA * 4 + 3) * BM + irA + o] = rA[idx].w;
                }
                for (int o = 0, idx = 0; o < BK; o += strideB, ++idx)
                    reinterpret_cast<float4*>(&Bs[nbuf][(irB + o) * BN + icB * 4])[0] = rB[idx];
                __syncthreads();
                buf = nbuf;
            }
        }
        // K-remainder (only when !ALIGNED). fullM & fullN guaranteed here.
        if (!ALIGNED && kRem) {
            const int kbase = numFull * BK;
            __syncthreads();
            for (int idx = threadIdx.x; idx < A_elems; idx += numThreads) {
                int r = idx / BK, c = idx % BK;
                As[0][c * BM + r] = (c < kRem) ? Ai[(size_t)r * K + kbase + c] : 0.0f;
            }
            for (int idx = threadIdx.x; idx < B_elems; idx += numThreads) {
                int r = idx / BN, c = idx % BN;
                Bs[0][r * BN + c] = (r < kRem) ? Bi[(size_t)(kbase + r) * N + c] : 0.0f;
            }
            __syncthreads();
            for (int k = 0; k < BK; ++k) {
                for (int i = 0; i < TM; ++i) regM[i] = As[0][k * BM + threadRow * TM + i];
                for (int j = 0; j < TN; ++j) regN[j] = Bs[0][k * BN + threadCol * TN + j];
                for (int i = 0; i < TM; ++i)
                    for (int j = 0; j < TN; ++j)
                        acc[i * TN + j] += regM[i] * regN[j];
            }
        }
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; j += 4) {
                float4 v;
                v.x = acc[i * TN + j + 0];
                v.y = acc[i * TN + j + 1];
                v.z = acc[i * TN + j + 2];
                v.w = acc[i * TN + j + 3];
                reinterpret_cast<float4*>(
                    &C[(size_t)(blockRow + threadRow * TM + i) * N + blockCol + threadCol * TN + j])[0] = v;
            }
        return;
    }

    // ===================== EDGE (scalar masked) =====================
    const int numTiles = (K + BK - 1) / BK;
    int buf = 0;
    // initial tile
    for (int idx = threadIdx.x; idx < A_elems; idx += numThreads) {
        int r = idx / BK, c = idx % BK;
        int gr = blockRow + r, gc = c;
        As[0][c * BM + r] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
    }
    for (int idx = threadIdx.x; idx < B_elems; idx += numThreads) {
        int r = idx / BN, c = idx % BN;
        int gr = r, gc = blockCol + c;
        Bs[0][r * BN + c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
    }
    __syncthreads();
    for (int t = 0; t < numTiles; ++t) {
        const bool has_next = (t + 1 < numTiles);
        const int nbuf = buf ^ 1;
        if (has_next) {
            int kbase = (t + 1) * BK;
            for (int idx = threadIdx.x; idx < A_elems; idx += numThreads) {
                int r = idx / BK, c = idx % BK;
                int gr = blockRow + r, gc = kbase + c;
                As[nbuf][c * BM + r] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.0f;
            }
            for (int idx = threadIdx.x; idx < B_elems; idx += numThreads) {
                int r = idx / BN, c = idx % BN;
                int gr = kbase + r, gc = blockCol + c;
                Bs[nbuf][r * BN + c] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.0f;
            }
        }
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) regM[i] = As[buf][k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; ++j) regN[j] = Bs[buf][k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
        buf = nbuf;
    }
    for (int i = 0; i < TM; ++i) {
        int gr = blockRow + threadRow * TM + i;
        if (gr >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int gc = blockCol + threadCol * TN + j;
            if (gc < N) C[(size_t)gr * N + gc] = acc[i * TN + j];
        }
    }
}

// Host launcher: 128x128 / BK16 / 16x8. Picks the ALIGNED fast instantiation when
// the shape is an exact tile multiple with float4-safe N,K; otherwise the general
// boundary-handled instantiation.
void launch_matmul_boundary(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 16, TM = 16, TN = 8;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block((BM * BN) / (TM * TN));
    const bool tileExact = (M % BM == 0) && (N % BN == 0) && (K % BK == 0);
    const bool vecOK = (N % 4 == 0) && (K % 4 == 0);
    if (tileExact && vecOK)
        matmul_boundary_kernel<BM, BN, BK, TM, TN, true><<<grid, block>>>(A, B, C, M, N, K, 1);
    else
        matmul_boundary_kernel<BM, BN, BK, TM, TN, false><<<grid, block>>>(A, B, C, M, N, K, vecOK ? 1 : 0);
}
