"""Cython landmine demo: ** 0.5 vs sqrt() with typed doubles.

Both compute the same thing (square root). Cython's ** operator goes through
a slow dispatch path instead of compiling to a single fsqrt instruction.

Build:
    uv run --extra cython python cython_benchmark/setup_pow_vs_sqrt.py build_ext --inplace

Run:
    uv run python cython_benchmark/bench_pow_vs_sqrt.py
"""

from __future__ import annotations

import cython
from cython.cimports.libc.math import sqrt  # type: ignore[import-not-found]

N: int = 50_000_000


@cython.cfunc
@cython.cdivision(True)
def _bench_pow(n: cython.int) -> cython.double:
    total: cython.double = 0.0
    x: cython.double
    i: cython.int
    for i in range(1, n):
        x = cython.cast(cython.double, i)
        total += x**0.5
    return total


@cython.cfunc
@cython.cdivision(True)
def _bench_sqrt(n: cython.int) -> cython.double:
    total: cython.double = 0.0
    x: cython.double
    i: cython.int
    for i in range(1, n):
        x = cython.cast(cython.double, i)
        total += sqrt(x)
    return total


@cython.ccall
def run_pow(n: cython.int = N) -> cython.double:
    return _bench_pow(n)


@cython.ccall
def run_sqrt(n: cython.int = N) -> cython.double:
    return _bench_sqrt(n)


def main() -> None:
    import time

    # warmup
    run_pow(1000)
    run_sqrt(1000)

    times_pow: list[float] = []
    times_sqrt: list[float] = []

    for _ in range(5):
        t0 = time.perf_counter()
        run_pow(N)
        t1 = time.perf_counter()
        times_pow.append(t1 - t0)

    for _ in range(5):
        t0 = time.perf_counter()
        run_sqrt(N)
        t1 = time.perf_counter()
        times_sqrt.append(t1 - t0)

    min_pow = min(times_pow)
    min_sqrt = min(times_sqrt)

    print(f"x ** 0.5:  {min_pow:.4f}s  (50M iterations)")
    print(f"sqrt(x):   {min_sqrt:.4f}s  (50M iterations)")
    print(f"ratio:     {min_pow / min_sqrt:.1f}x")
    print()
    print("Both compute the same value. In pure CPython they run at the same speed.")
    print("Cython's ** operator goes through a slow dispatch path.")


if __name__ == "__main__":
    main()
