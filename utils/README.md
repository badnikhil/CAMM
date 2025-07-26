# CUDA Matrix Multiplication Benchmark (main.cu)

This file contains the main function for benchmarking matrix multiplication on the GPU using CUDA.

## What Happens in main.cu (Step-by-Step)
1. **Host Memory Allocation:**

   - Allocates memory for matrices A, B, and C on the host (CPU).
   - If allocation fails, prints an error and exits.
2. **Matrix Initialization:**
   - Fills matrices A and B with random floating-point values.
3. **Device Memory Allocation:**
   - Allocates memory for matrices A, B, and C on the device (GPU).
   - If allocation fails, prints an error and exits.
4. **Host-to-Device Memory Copy:**
   - Copies matrices A and B from host to device memory.
   - If copy fails, prints an error and exits.
5. **Kernel Warmup:**
   - Launches the matrix multiplication kernel once to warm up the GPU and avoid first-run overhead in timing.
6. **Benchmarking Loop:**
   - Runs the matrix multiplication kernel multiple times (default: 10) and measures the average execution time using CUDA events.
   - Checks for kernel launch errors after each run.
7. **Device-to-Host Memory Copy:**
   - Copies the result matrix C from device to host memory.
   - If copy fails, prints an error and exits.
8. **Timing and Performance Reporting:**
   - Prints the following to the console:
     - Host-to-device memory transfer time
     - Device-to-host memory transfer time
     - Average kernel execution time
     - Kernel GFLOPS (Giga Floating Point Operations Per Second)
9. **Cleanup:**
   - Frees all allocated host and device memory.
   - Destroys CUDA events.

## Features
- Allocates and initializes matrices on the host and device
- Runs a warmup kernel execution
- Measures and reports:
  - Host-to-device memory transfer time
  - Device-to-host memory transfer time
  - Average kernel execution time (over multiple runs)
  - Kernel GFLOPS (Giga Floating Point Operations Per Second)
- Includes robust error checking for memory allocation and CUDA API calls

## Usage
1. **Compile:**
   ```bash
   nvcc utils/main.cu Kernel/matmul_naive.cu -o cuda_benchmark
   ```
2. **Run:**
   ```bash
   ./cuda_benchmark
   ```

## Output
The program prints timing results and GFLOPS to the console. Adjust the matrix size by changing the `N` macro in `main.cu`.

---
For CPU benchmarking, see `cpu_benchmarking.cpp` in this folder. 