"""Spectral-norm — Numba JIT implementation.

Algorithm: Same spectral-norm problem from the Benchmarks Game
(https://benchmarksgame-team.pages.debian.net/benchmarksgame/).

Implementation: @njit-compiled loops that compute A[i][j] on the fly
(same approach as the baseline, but compiled to machine code via LLVM).
Uses O(N) memory unlike the NumPy version's O(N^2).

No community Numba spectral-norm exists (the Benchmarks Game only accepts
pure-language submissions), so this was written as a mechanical translation
of the baseline with Numba decorators.

Correctness: Verified against official expected outputs (verify_correctness.py):
  N=2:   1.183350177  [PASS]
  N=100: 1.274219991  [PASS]
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from baseline.spectral_norm import DEFAULT_N


@njit(cache=True)
def _eval_A(i: int, j: int) -> float:
    """Element A[i][j] of the infinite matrix. Compiled by Numba."""
    ij = i + j
    return 1.0 / (ij * (ij + 1) / 2 + i + 1)


@njit(cache=True)
def _eval_AtA_times_u(n: int, u: np.ndarray, out: np.ndarray) -> None:
    """Compute out = A^T * A * u in two passes. No temporary array allocation."""
    # First: tmp = A * u
    tmp = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += _eval_A(i, j) * u[j]
        tmp[i] = s

    # Second: out = A^T * tmp
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += _eval_A(j, i) * tmp[j]
        out[i] = s


@njit(cache=True)
def _run(n: int) -> float:
    """Run the full spectral-norm computation. Compiled by Numba."""
    u = np.ones(n, dtype=np.float64)
    v = np.empty(n, dtype=np.float64)

    for _ in range(10):
        _eval_AtA_times_u(n, u, v)
        _eval_AtA_times_u(n, v, u)

    vBv = 0.0
    vv = 0.0
    for i in range(n):
        vBv += u[i] * v[i]
        vv += v[i] * v[i]

    return math.sqrt(vBv / vv)


def warmup() -> None:
    """Trigger Numba compilation before timing."""
    _run(10)


def run_benchmark(n: int = DEFAULT_N) -> dict[str, object]:
    """Run spectral-norm with Numba JIT."""
    result = _run(n)
    return {
        "n": n,
        "spectral_norm": round(float(result), 9),
    }


def main() -> None:
    warmup()
    result = run_benchmark()
    print(f"Spectral-norm Numba (N={result['n']})")
    print(f"  Result: {result['spectral_norm']}")


if __name__ == "__main__":
    main()
