import functools
import logging
import random
from argparse import ArgumentParser
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
    niters: int = 100


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


def run_bench_case(case: BenchCase):
    logger.info(f"Running benchmark for {case.name}")
    # Assert correctness
    fn_args = functools.partial(case.fn, *case.args)
    baseline_fn_args = functools.partial(case.baseline_fn, *case.args)

    torch.testing.assert_close(fn_args(), baseline_fn_args())

    result = bench(fn_args, case.niters)
    result_baseline = bench(baseline_fn_args, case.niters)

    logger.info(
        f" Results. Helion: {result:.2f} ms, Torch: {result_baseline:.2f} ms, Speedup: {result_baseline / result:.2f}x"
    )


@torch.no_grad()
def main():
    inner_dims = [16, 512, 768, 1024, 2048]
    outer_dims = [1, 128, 1024, 2561]

    for outer_dim in outer_dims:
        for inner_dim in inner_dims:
            x = torch.randn(outer_dim, inner_dim, device="cuda")
            gamma = torch.randn(inner_dim, device="cuda")
            beta = torch.randn(inner_dim, device="cuda")
            eps = 1e-5

            case = BenchCase(
                f"input_shape_{outer_dim}_{inner_dim}",
                layer_norm_helion,
                layer_norm_torch,
                (x, gamma, beta, eps),
            )
            run_bench_case(case)
    logger.info("Benchmarking completed successfully")


if __name__ == "__main__":
    set_seed(42)

    parser = ArgumentParser()
    parser.add_argument("--full-search", action="store_true")

    args = parser.parse_args()

    if args.full_search:
        layer_norm_helion.settings.autotune_effort = "full"
    else:
        layer_norm_helion.settings.autotune_effort = "quick"

    main()
