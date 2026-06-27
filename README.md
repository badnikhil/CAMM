# CAMM - CUDA Accelerated Matrix Multiplication

A comprehensive CUDA implementation showcasing various matrix multiplication optimization techniques, from naive approaches to highly optimized kernels with register tiling and vectorization.

## 🚀 Features

- **9 Different Kernel Implementations** with progressive optimizations
- **Auto-dispatch** entry point that selects the best kernel per shape
- **Arbitrary-shape kernels** (any M, N, K — square, rectangular, non-aligned)
- **Autotuner** that sweeps tile geometries to find the best config per GPU
- **Comprehensive Benchmarking** against cuBLAS and CUTLASS
- **Performance Analysis** with detailed metrics
- **Modular Architecture** for easy experimentation

## 📁 Project Structure

```
CAMM/
├── Kernel/                          # CUDA kernel implementations
│   ├── matmul_naive/               # Basic matrix multiplication
│   ├── mat_mul_coalesced/          # Memory coalescing optimization
│   ├── mat_mul_sharedmem/          # Shared memory optimization
│   ├── mat_mul_register_tiling/    # Register tiling with specialization
│   ├── mat_mul_vectorized/         # float4 vectorized memory access
│   ├── mat_mul_tuned/              # Autotuned 128x128 / BK16 / 16x8 reg tile
│   ├── mat_mul_doublebuffer/       # Software-pipelined double buffering
│   ├── mat_mul_general/            # Arbitrary M,N,K (bounds-checked)
│   ├── mat_mul_boundary/           # Arbitrary M,N,K (fast interior + masked edges)
│   └── mat_mul_auto/               # Auto-dispatch: best kernel per shape
├── Header/
│   └── matmul_kernels.cuh          # Kernel function declarations
├── utils/                          # Benchmarking and utility functions
│   ├── benchmark_matmul_*.cu       # Individual kernel benchmarks
│   ├── benchmark_matmul_shapes.cu  # Arbitrary-shape (general/boundary) benchmark
│   ├── autotune.cu                 # Tile-geometry sweep vs cuBLAS
│   ├── main.cu                     # Main benchmarking suite
│   └── cpu_benchmarking.cpp        # CPU reference implementation
├── Benchmarks/                     # Performance results (ignored by git)
└── cutlass/                        # NVIDIA CUTLASS library integration
```

## 🔧 Kernel Implementations

### 1. Naive Implementation (`matmul_naive`)
- **Description**: Basic matrix multiplication without optimizations
- **Grid/Block**: Standard 2D grid configuration
- **Use Case**: Baseline performance reference

### 2. Coalesced Memory Access (`mat_mul_coalesced`)
- **Description**: Optimized memory access patterns for better bandwidth utilization
- **Optimization**: Ensures coalesced global memory access
- **Performance**: ~2-3x improvement over naive implementation

### 3. Shared Memory (`mat_mul_sharedmem`)
- **Description**: Utilizes shared memory to reduce global memory accesses
- **Optimization**: Tile-based computation with shared memory blocking
- **Performance**: ~4-6x improvement over naive implementation

### 4. Register Tiling (`mat_mul_register_tiling`)
- **Description**: Advanced optimization using register-level tiling
- **Features**:
  - Base register tiling implementation
  - **Size-specialized kernels** for 128x128 and 512x512 matrices
  - Optimized grid dimensions: `gridDim(16,16)`, `blockDim(16,16)`
- **Performance**: ~8-12x improvement over naive implementation

### 5. Vectorized Operations (`matmul_vectorized`)
- **Description**: Specialized kernels with vectorized memory operations
- **Performance**: Highest performance Achieved Yet
- **⚠️ Note**: Currently experiencing efficiency loss that requires optimization fixes

### 6. Tuned Register Tiling (`matmul_tuned`)
- **Description**: Autotuned 128x128 block tile, BK=16, asymmetric **16x8** thread
  register tile (128 threads/block), transposed shared A, float4-vectorized loads/stores
- **Key finding**: the asymmetric 16x8 register tile beats both 8x8 and 8x16
- **Constraint**: square N, multiple of 128

### 7. Double Buffering (`matmul_doublebuffer`)
- **Description**: Software-pipelined version of the tuned kernel — prefetches the
  next K-block into registers while the current block's FMAs run, overlapping
  global-load latency with compute
- **Constraint**: square N, multiple of 128

### 8. General (`launch_matmul_general`)
- **Description**: Arbitrary **M, N, K** matmul. Same register-blocking compute as
  the tuned kernel, but every global access is bounds-checked, so any shape runs
  correctly (square, rectangular, non-128-aligned). Uses a 64x64 tile for small
  dimensions (fills all SMs) and 128x128 otherwise.
- **Signature**: `launch_matmul_general(A, B, C, M, N, K)` — `C(MxN)=A(MxK)*B(KxN)`

### 9. Boundary (`launch_matmul_boundary`)
- **Description**: Arbitrary **M, N, K**, but interior 128x128 tiles take the fast
  float4 double-buffered path while only the partial edge tiles + K-remainder take
  the scalar masked path — no padding buffers, no extra copies. The fast interior
  path needs `N%4==0 && K%4==0`; otherwise it falls back to the scalar path.
- **Signature**: `launch_matmul_boundary(A, B, C, M, N, K)`

### ⭐ Auto (`launch_matmul_auto`) — recommended entry point
- **Description**: Picks the best kernel for the given shape automatically:
  - square & multiple of 128, K in [5120, 8192) → **tuned** (single-buffer wins at deep K)
  - square & multiple of 128, otherwise → **doublebuffer** (best general default)
  - any other shape → **boundary** (handles arbitrary M, N, K)
- **Signature**: `launch_matmul_auto(A, B, C, M, N, K)`
- **Use this** if you just want the fastest correct result for any shape.

## 🏗️ Build Instructions

### Prerequisites
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.0+
- GCC/G++ compiler
- CMake (optional)

### Compilation

#### Individual Kernels
```bash
# Naive implementation
nvcc -o naive utils/benchmark_matmul_naive.cu Kernel/matmul_naive/*.cu

# Coalesced memory access
nvcc -o coalesced utils/benchmark_matmul_coalesced.cu Kernel/mat_mul_coalesced/*.cu

# Shared memory optimization
nvcc -o shared utils/benchmark_matmul_sharedmem.cu Kernel/mat_mul_sharedmem/*.cu

# Register tiling
nvcc -o register utils/benchmark_matmul_register_tiling.cu Kernel/mat_mul_register_tiling/*.cu

# Vectorized
nvcc -o vectorized utils/benchmark_matmul_vectorized.cu Kernel/mat_mul_vectorized/*.cu

# Tuned (128x128, BK=16, 16x8 register tile)
nvcc -O3 -arch=sm_86 -o tuned utils/benchmark_matmul_tuned.cu Kernel/mat_mul_tuned/*.cu

# Double-buffered
nvcc -O3 -arch=sm_86 -o doublebuffer utils/benchmark_matmul_doublebuffer.cu Kernel/mat_mul_doublebuffer/*.cu

# Arbitrary-shape kernels (general + boundary), square/rectangular/non-aligned
nvcc -O3 -arch=sm_86 -o shapes utils/benchmark_matmul_shapes.cu \
     Kernel/mat_mul_general/*.cu Kernel/mat_mul_boundary/*.cu

# Autotuner: sweep tile geometries vs cuBLAS (needs -lcublas)
nvcc -O3 -arch=sm_86 -o autotune utils/autotune.cu -lcublas

# ⭐ Auto kernel — picks the best kernel per shape automatically
nvcc -O3 -arch=sm_86 -o auto utils/benchmark_matmul_auto.cu \
     Kernel/mat_mul_auto/*.cu Kernel/mat_mul_tuned/*.cu \
     Kernel/mat_mul_doublebuffer/*.cu Kernel/mat_mul_boundary/*.cu

# Head-to-head: cuBLAS vs our auto kernel on matched sizes (needs -lcublas)
nvcc -O3 -arch=sm_86 -o compare utils/benchmark_compare.cu \
     Kernel/mat_mul_auto/*.cu Kernel/mat_mul_tuned/*.cu \
     Kernel/mat_mul_doublebuffer/*.cu Kernel/mat_mul_boundary/*.cu -lcublas
```

#### Complete Benchmarking Suite
```bash
# Compile main benchmarking application
nvcc -o benchmark utils/main.cu Kernel/*/*.cu -I./Header

# Compare against cuBLAS
nvcc -o cublas_bench utils/benchmark_matmul_cublas.cu -lcublas

# Compare against CUTLASS
nvcc -o cutlass_bench utils/benchmark_matmul_cutlass.cu -I./cutlass/include
```

### Compilation Flags (Recommended)
```bash
nvcc -O3 -arch=sm_75 -use_fast_math -Xptxas -O3 -o <output> <source_files>
```

## 📊 Performance Benchmarking

### Running Benchmarks
```bash
# Run individual kernel benchmark
./benchmark

# ⭐ Auto kernel — best kernel per shape, the recommended entry point
./auto              # default square sweep (128 .. 8192)
./auto 4096         # single square N=4096
./auto 1024 4096 2048   # rectangular M=1024 N=4096 K=2048

# Head-to-head: cuBLAS vs our auto kernel (matched sizes, prints comparison table)
./compare

# Tuned kernel sweep (N = 128 .. 8192)
./tuned

# Double-buffered kernel sweep
./doublebuffer

# Arbitrary-shape kernels: square, rectangular, non-aligned (with correctness check)
./shapes

# Autotuner — sweep tile geometries at the given sizes (defaults to 2048 4096)
./autotune 2048 4096

# Compare with cuBLAS
./cublas_bench

# Compare with CUTLASS
./cutlass_bench
```

> **Note on `-arch`**: use `sm_86` for Ampere consumer GPUs (e.g. RTX 30-series, RTX 2050),
> `sm_80` for A100, `sm_75` for Turing. The tuned/double-buffer/boundary kernels are
> square-tile-aligned to **multiples of 128** for the fast path; the general and boundary
> kernels accept any `M, N, K`. Re-run `./autotune` on a new GPU to re-find the best tile.

### Expected Performance Characteristics

| Kernel Type | Relative Performance | Memory Efficiency |
|-------------|---------------------|-------------------|
| Naive | 1x (baseline) | Low |
| Coalesced | 2-3x | Medium |
| Shared Memory | 4-6x | High |
| Register Tiling | 8-12x | Very High |
| Specialized | 10-15x | Very High |


## 🔬 Optimization Techniques Demonstrated

1. **Memory Coalescing**: Ensuring aligned memory access patterns
2. **Shared Memory Utilization**: Reducing global memory bandwidth requirements
3. **Register Tiling**: Maximizing register usage and reducing memory latency
4. **Thread Block Optimization**: Optimal thread block dimensions
5. **Vectorized Operations**: Using float4 vector load/store instructions
6. **Asymmetric Register Tiles**: 16x8 thread tile (autotuned winner on Ampere)
7. **Double Buffering**: Software-pipelined prefetch overlapping load with compute
8. **Predicated Boundary Handling**: fast interior tiles + masked edges for any shape
9. **Autotuning**: sweeping tile geometry to find the GPU-specific optimum


## 📈 Development and Testing

### Adding New Kernels
1. Create kernel implementation in `Kernel/<kernel_name>/`
2. Add declaration to `Header/matmul_kernels.cuh`
3. Create benchmark in `utils/benchmark_<kernel_name>.cu`
4. Update main benchmarking suite

### Performance Testing
- Benchmark results are automatically saved to `Benchmarks/` (git-ignored)
- Use consistent matrix sizes for fair comparisons
- Run multiple iterations for statistical significance

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive benchmarks for new implementations
3. Document optimization techniques used
4. Ensure compatibility with existing build system

## 📝 License

This project is for educational and research purposes, demonstrating CUDA optimization techniques for matrix multiplication.

## 🔗 References

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass)
- [cuBLAS Library Documentation](https://docs.nvidia.com/cuda/cublas/)

---

**Note**: Performance results may vary based on GPU architecture, CUDA version, and system configuration. Benchmark on your target hardware for accurate performance characteristics.
