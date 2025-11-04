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
python3 benchmark.py --niters 100 --warmup 10
```

#### Benchmark Options

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
| **2048 × 1024** | 0.10 | 0.05 | 0.52x |
| **2048 × 1536** | 0.10 | 0.08 | 0.81x |
| **2048 × 2048** | 0.14 | 0.12 | 0.87x |
| **2048 × 7168** | 0.47 | 0.47 | 1.00x |
| **3100 × 1536** | 0.11 | 0.12 | **1.12x** 🚀 |
| **3100 × 2048** | 0.18 | 0.18 | 1.03x |
| **3100 × 7168** | 0.71 | 0.70 | 0.99x |

## License

MIT License - see [LICENSE](LICENSE) file for details.
