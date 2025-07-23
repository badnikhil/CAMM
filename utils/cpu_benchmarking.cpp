#include <iostream>
#include <vector>
#include <chrono>

#define N 1024

void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void random_fill(std::vector<float>& arr) {
    for (auto& x : arr) x = static_cast<float>(rand()) / RAND_MAX;
}

int main() {
    std::vector<float> A(N*N), B(N*N), C(N*N);
    random_fill(A);
    random_fill(B);

    auto start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Checksum calculation
    double checksum = 0.0;
    for (int i = 0; i < N*N; ++i) checksum += C[i];
    std::cout << "Checksum: " << checksum << std::endl;

    std::cout << "C[0] = " << C[0] << std::endl;
    std::cout << "CPU matmul time: " << elapsed.count() << " ms" << std::endl;
    return 0;
} 