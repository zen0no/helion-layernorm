# Helion LayerNorm

A high-performance LayerNorm implementation using [Helion](https://github.com/your-org/helion), optimized for GPU acceleration.

## Overview

This project provides a GPU-accelerated LayerNorm implementation that leverages Helion's kernel compilation capabilities. The implementation uses a tiled approach with online statistics computation for efficient memory usage and performance.

## Features

- ✅ GPU-accelerated with CUDA support
- ✅ Compatible with PyTorch tensors
- ✅ Numerical correctness verified against PyTorch's native implementation
- ✅ Optimized for various input shapes and dimensions

## Installation

```bash
pip install .
```

## Usage

To benchmark this implementation, use following command:

```bash
HELION_SKIP_CACHE=1 python3 benchmark.py --full-search
```

Full-search option enables full compilation optimization.

Results will be logged to `logs/benchmark_run_<timestamp>.log`.

## Benchmarking Results

Benchmark results comparing Helion LayerNorm against PyTorch's native `F.layer_norm` implementation. All benchmarks were run on CUDA with 100 iterations and 10 warmup runs.

### Performance Summary

| Input Shape | Helion ⚡ (ms) | PyTorch 🔥 (ms) | Speedup |
|-------------|----------------|-----------------|---------|
| **1 × 16** | 0.09 | 0.02 | 0.19x |
| **1 × 512** | 0.09 | 0.02 | 0.20x |
| **1 × 768** | 0.08 | 0.02 | 0.22x |
| **1 × 1024** | 0.08 | 0.02 | 0.21x |
| **1 × 2048** | 0.09 | 0.02 | 0.19x |
| **128 × 16** | 0.09 | 0.02 | 0.20x |
| **128 × 512** | 0.09 | 0.02 | 0.19x |
| **128 × 768** | 0.09 | 0.02 | 0.20x |
| **128 × 1024** | 0.09 | 0.02 | 0.19x |
| **128 × 2048** | 0.08 | 0.02 | 0.21x |
| **1024 × 16** | 0.09 | 0.02 | 0.19x |
| **1024 × 512** | 0.08 | 0.03 | 0.32x |
| **1024 × 768** | 0.09 | 0.03 | 0.39x |
| **1024 × 1024** | 0.09 | 0.04 | 0.44x |
| **1024 × 2048** | 0.09 | 0.07 | **0.80x** |
| **2561 × 16** | 0.08 | 0.04 | 0.43x |
| **2561 × 512** | 0.08 | 0.06 | 0.69x |
| **2561 × 768** | 0.09 | 0.07 | 0.81x |
| **2561 × 1024** | 0.08 | 0.08 | **1.09x** 🚀 |
| **2561 × 2048** | 0.18 | 0.16 | 0.90x |

## License

MIT License - see [LICENSE](LICENSE) file for details.
