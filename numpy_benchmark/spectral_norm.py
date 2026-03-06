"""Spectral-norm — NumPy vectorized implementation.

Algorithm: Same spectral-norm problem from the Benchmarks Game
(https://benchmarksgame-team.pages.debian.net/benchmarksgame/).

Implementation: Pre-computes the full N*N matrix A using numpy broadcasting,
then delegates matrix-vector multiplies to BLAS (Apple Accelerate on macOS,
OpenBLAS on Linux). This trades O(N^2) memory for O(1) Python calls per
iteration — the "just use NumPy" approach.

The Benchmarks Game does not accept NumPy submissions (it tests the language,
not its libraries), so no official NumPy spectral-norm exists. This was
written as a vectorized translation of the baseline algorithm.

Correctness: Verified against official expected outputs (verify_correctness.py):
  N=2:   1.183350177  [PASS]
  N=100: 1.274219991  [PASS]
"""

from __future__ import annotations

import math

import numpy as np

from baseline.spectral_norm import DEFAULT_N


def build_matrix(n: int) -> np.ndarray:
    """Build the N×N matrix A where A[i][j] = 1/((i+j)(i+j+1)/2 + i + 1).

    Uses numpy broadcasting — no Python loops.
    """
    i = np.arange(n, dtype=np.float64).reshape(-1, 1)  # column vector
    j = np.arange(n, dtype=np.float64).reshape(1, -1)  # row vector
    ij = i + j
    return 1.0 / (ij * (ij + 1) / 2 + i + 1)


def run_benchmark(n: int = DEFAULT_N) -> dict[str, object]:
    """Run spectral-norm using NumPy matrix-vector multiply."""
    # Build the matrix once — this is O(N^2) but runs in vectorized C
    a = build_matrix(n)
    at = a.T  # A-transpose, also pre-computed

    u = np.ones(n, dtype=np.float64)

    for _ in range(10):
        v = at @ (a @ u)  # A^T * A * u in one step
        u = at @ (a @ v)  # A^T * A * v

    vBv = float(np.dot(u, v))
    vv = float(np.dot(v, v))

    result = math.sqrt(vBv / vv)
    return {
        "n": n,
        "spectral_norm": round(result, 9),
    }


def main() -> None:
    result = run_benchmark()
    print(f"Spectral-norm NumPy (N={result['n']})")
    print(f"  Result: {result['spectral_norm']}")


if __name__ == "__main__":
    main()
