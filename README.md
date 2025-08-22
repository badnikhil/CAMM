# CAMM - CUDA Accelerated Matrix Multiplication

A comprehensive CUDA implementation showcasing various matrix multiplication optimization techniques, from naive approaches to highly optimized kernels with register tiling and vectorization.

## 🚀 Features

- **5 Different Kernel Implementations** with progressive optimizations
- **Comprehensive Benchmarking** against cuBLAS and CUTLASS
- **Performance Analysis** with detailed metrics
- **Modular Architecture** for easy experimentation
- **Size-Specialized Kernels** for optimal performance

## 📁 Project Structure

```
CAMM/
├── Kernel/                          # CUDA kernel implementations
│   ├── matmul_naive/               # Basic matrix multiplication
│   ├── mat_mul_coalesced/          # Memory coalescing optimization
│   ├── mat_mul_sharedmem/          # Shared memory optimization
│   └── mat_mul_register_tiling/    # Register tiling with specialization
├── Header/
│   └── matmul_kernels.cuh          # Kernel function declarations
├── utils/                          # Benchmarking and utility functions
│   ├── benchmark_matmul_*.cu       # Individual kernel benchmarks
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

# Compare with cuBLAS
./cublas_bench

# Compare with CUTLASS
./cutlass_bench
```

### Expected Performance Characteristics

| Kernel Type | Relative Performance | Memory Efficiency | Best Use Case |
|-------------|---------------------|-------------------|---------------|
| Naive | 1x (baseline) | Low | Learning/Reference |
| Coalesced | 2-3x | Medium | Small matrices |
| Shared Memory | 4-6x | High | Medium matrices |
| Register Tiling | 8-12x | Very High | Large matrices |

### Matrix Size Recommendations
- **General sizes**: Use register tiling implementation
- **Very large matrices**: Consider cuBLAS integration

## 🔬 Optimization Techniques Demonstrated

1. **Memory Coalescing**: Ensuring aligned memory access patterns
2. **Shared Memory Utilization**: Reducing global memory bandwidth requirements
3. **Register Tiling**: Maximizing register usage and reducing memory latency
4. **Thread Block Optimization**: Optimal thread block dimensions
5. **Vectorized Operations**: Using vector load/store instructions
6. **Size Specialization**: Kernel variants optimized for specific matrix dimensions


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
- adding more soon
---

**Note**: Performance results may vary based on GPU architecture, CUDA version, and system configuration. Benchmark on your target hardware for accurate performance characteristics.
