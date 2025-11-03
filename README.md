# Helion LayerNorm

A high-performance LayerNorm implementation using [Helion](https://github.com/your-org/helion), optimized for GPU acceleration.

## Overview

This project provides a GPU-accelerated LayerNorm implementation that leverages Helion's kernel compilation capabilities. The implementation uses a tiled approach with online statistics computation for efficient memory usage and performance.

## Installation

```bash
pip install .
```

## Usage

### Benchmarking

#### Basic Benchmark

Run the benchmark with default parameters:

```bash
python3 benchmark.py
```

Results will be logged to `logs/benchmark_run_<timestamp>.log`.

#### Custom Benchmark

Customize benchmark dimensions and iterations:

```bash
python3 benchmark.py --outer-dims 128 512 --inner-dims 256 512 --niters 200
```

#### Benchmark Options

- `--outer-dims`: Batch sizes to test (default: `128 1024 2561 25512`)
- `--inner-dims`: Sequence lengths to test (default: `16 512 768 1024 2048`)
- `--niters`: Number of iterations per benchmark (default: `100`)
- `--warmup`: Number of warmup iterations (default: `10`)
- `--atol`: Absolute tolerance for correctness check (default: `1e-6`)
- `--rtol`: Relative tolerance for correctness check (default: `1e-6`)

#### Example Output

The benchmark will output timing results and log them:

```
Results. Helion: 0.64 ms, Torch: 0.59 ms, Speedup: 0.93x
```

## Benchmarking Results

Benchmark results, which were run on Quadro RTX 4000. As you can see, at big tensors performance comparable with Pytorch's native `F.layer_norm` implemenation, and even outpeforms it. Time were averaged across `niters=100` runs.


### Performance Summary

| Input Shape | Helion ⚡ (ms) | PyTorch 🔥 (ms) | Speedup |
|-------------|----------------|-----------------|---------|
| **128 × 16** | 0.08 | 0.02 | 0.19x |
| **128 × 512** | 0.08 | 0.02 | 0.20x |
| **128 × 768** | 0.08 | 0.02 | 0.19x |
| **128 × 1024** | 0.07 | 0.02 | 0.21x |
| **128 × 2048** | 0.07 | 0.02 | 0.20x |
| **1024 × 16** | 0.08 | 0.02 | 0.21x |
| **1024 × 512** | 0.08 | 0.02 | 0.20x |
| **1024 × 768** | 0.08 | 0.02 | 0.27x |
| **1024 × 1024** | 0.08 | 0.03 | 0.31x |
| **1024 × 2048** | 0.08 | 0.06 | 0.73x |
| **2561 × 16** | 0.08 | 0.02 | 0.24x |
| **2561 × 512** | 0.09 | 0.03 | 0.36x |
| **2561 × 768** | 0.09 | 0.05 | 0.54x |
| **2561 × 1024** | 0.10 | 0.06 | 0.63x |
| **2561 × 2048** | 0.18 | 0.15 | 0.87x |
| **25512 × 16** | 0.08 | 0.18 | **2.10x** 🚀 |
| **25512 × 512** | 0.28 | 0.38 | **1.33x** 🚀 |
| **25512 × 768** | 0.43 | 0.48 | **1.13x** 🚀 |
| **25512 × 1024** | 0.64 | 0.59 | 0.93x |
| **25512 × 2048** | 1.65 | 1.48 | 0.90x |

## License

MIT License - see [LICENSE](LICENSE) file for details.
