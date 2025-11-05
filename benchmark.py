import functools
import logging
import random
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.layer_norm import layer_norm as layer_norm_helion


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# Set up logging

base_path = Path(__file__).parent / "logs"
base_path.mkdir(parents=True, exist_ok=True)

run_name = f"benchmark_run_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"

logging.basicConfig(
    filename=base_path / f"{run_name}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

logger = getLogger(__name__)


def layer_norm_torch(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float
) -> torch.Tensor:
    return F.layer_norm(x, [x.shape[-1]], gamma, beta, eps)


# Benchmarking logic


def time_fn(
    fn: Callable[[], Any], start: torch.cuda.Event, end: torch.cuda.Event
) -> float:
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


@dataclass
class BenchCase:
    name: str
    fn: Callable[[], Any]
    baseline_fn: Callable[[], Any]
    args: Any


def bench(fn: Callable[[], Any], warmup: int = 10, niters=100) -> float:
    for _ in range(warmup):
        fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    for i in range(niters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / niters


def run_bench_case(
    case: BenchCase,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    niters: int = 100,
    warmup: int = 10,
):
    logger.info(f"Running benchmark for {case.name}")
    # Assert correctness
    fn_args = functools.partial(case.fn, *case.args)
    baseline_fn_args = functools.partial(case.baseline_fn, *case.args)

    torch.testing.assert_close(fn_args(), baseline_fn_args(), atol=atol, rtol=rtol)

    result = bench(fn=fn_args, niters=niters, warmup=warmup)
    result_baseline = bench(fn=baseline_fn_args, niters=niters, warmup=warmup)

    logger.info(
        f" Results. Helion: {result:.2f} ms, Torch: {result_baseline:.2f} ms, Speedup: {result_baseline / result:.2f}x"
    )


@torch.no_grad()
def main(test_dims: list[tuple[int, int]], args: Namespace): 
    for outer_dim, inner_dim in test_dims:
        x = torch.randn(outer_dim, inner_dim, device="cuda", dtype=torch.float32)
        gamma = torch.randn(inner_dim, device="cuda", dtype=torch.float32)
        beta = torch.randn(inner_dim, device="cuda", dtype=torch.float32)
        eps = 1e-5

        case = BenchCase(
            f"input_shape_{outer_dim}_{inner_dim}",
            layer_norm_helion,
            layer_norm_torch,
            (x, gamma, beta, eps),
        )
        run_bench_case(
            case=case,
            atol=args.atol,
            rtol=args.rtol,
            niters=args.niters,
            warmup=args.warmup,
        )
    logger.info("Benchmarking completed successfully")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--niters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1.3e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    test_dims = [
        (2048, 1024),
        (2048, 1536),
        (2048, 2048),
        (2048, 7168),
        (3100, 1536),
        (3100, 2048),
        (3100, 7168),
    ]
    args = parse_args()
    main(test_dims=test_dims, args=args)
