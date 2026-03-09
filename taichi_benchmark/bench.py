"""Benchmark runner for Taichi — n-body and spectral-norm.

Taichi compiles @ti.kernel decorated functions to native machine code via LLVM.
Similar to Numba but with explicit kernel/func distinction and auto-parallelization.

Usage: /tmp/taichi-venv/bin/python taichi_benchmark/bench.py

Requires: taichi (Python 3.13 — no 3.14 wheels available).
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path (needed when running from external venv)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import taichi as ti

# CPU backend with default thread count (Taichi auto-parallelizes eligible loops).
# Must be called before importing nbody/spectral_norm (they create fields at import time).
ti.init(arch=ti.cpu, default_fp=ti.f64)

from taichi_benchmark.nbody import DEFAULT_N as NBODY_N  # noqa: E402
from taichi_benchmark.nbody import run_nbody  # noqa: E402
from taichi_benchmark.spectral_norm import DEFAULT_N as SPECTRAL_N  # noqa: E402
from taichi_benchmark.spectral_norm import run_spectral  # noqa: E402

RUNS = 5


def main() -> None:
    print("Taichi 1.7.4 (CPU)")
    print(f"Runs: {RUNS}")
    print()

    # Warmup (triggers JIT compilation)
    run_nbody()
    run_spectral()

    # N-body
    print(f"  n-body ({NBODY_N} iterations)...", end=" ", flush=True)
    times_nb: list[float] = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        run_nbody()
        t1 = time.perf_counter()
        times_nb.append((t1 - t0) * 1000)
    med_nb = statistics.median(times_nb)
    print(f"{med_nb:.1f}ms")

    # Spectral-norm
    print(f"  spectral-norm (N={SPECTRAL_N})...", end=" ", flush=True)
    times_sn: list[float] = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        run_spectral()
        t1 = time.perf_counter()
        times_sn.append((t1 - t0) * 1000)
    med_sn = statistics.median(times_sn)
    print(f"{med_sn:.1f}ms")

    # Correctness
    e_before, e_after = run_nbody()
    sn = run_spectral()
    print()
    print("Correctness:")
    print(f"  nbody energy_before: {e_before}")
    print(f"  nbody energy_after:  {e_after}")
    print(f"  spectral_norm:       {sn}")


if __name__ == "__main__":
    main()
