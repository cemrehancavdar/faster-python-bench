"""Mypyc Spectral-norm: type-annotated Python compiled to C.

Key mypyc optimizations vs baseline:
- Explicit loop-based implementation (no list comprehensions with generators,
  which mypyc can't optimize as well as explicit loops)
- All locals explicitly typed for C primitive operations
- Pre-declared loop variables

Build: uv run python -c "from mypyc.build import mypycify; from setuptools import setup; setup(packages=[], ext_modules=mypycify(['mypyc_benchmark/spectral_norm.py']))" build_ext --inplace
"""

from __future__ import annotations

import math

DEFAULT_N: int = 2000


def eval_A(i: int, j: int) -> float:
    """Element A[i][j] of the infinite matrix."""
    ij: int = i + j
    return 1.0 / (ij * (ij + 1) / 2 + i + 1)


def eval_A_times_u(u: list[float], n: int) -> list[float]:
    """Multiply vector u by matrix A."""
    result: list[float] = [0.0] * n
    i: int
    j: int
    s: float
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += eval_A(i, j) * u[j]
        result[i] = s
    return result


def eval_At_times_u(u: list[float], n: int) -> list[float]:
    """Multiply vector u by A-transpose."""
    result: list[float] = [0.0] * n
    i: int
    j: int
    s: float
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += eval_A(j, i) * u[j]
        result[i] = s
    return result


def eval_AtA_times_u(u: list[float], n: int) -> list[float]:
    """Multiply vector u by A^T * A (one power iteration step)."""
    return eval_At_times_u(eval_A_times_u(u, n), n)


def run_benchmark(n: int = DEFAULT_N) -> dict[str, object]:
    """Run the spectral-norm benchmark."""
    u: list[float] = [1.0] * n
    v: list[float] = [0.0] * n

    _: int
    for _ in range(10):
        v = eval_AtA_times_u(u, n)
        u = eval_AtA_times_u(v, n)

    vBv: float = 0.0
    vv: float = 0.0
    i: int
    for i in range(n):
        vBv += u[i] * v[i]
        vv += v[i] * v[i]

    result: float = math.sqrt(vBv / vv)
    return {
        "n": n,
        "spectral_norm": round(result, 9),
    }


EXPECTED_NORM_5500: float = 1.274224153
EXPECTED_NORM_2000: float = 1.274224153


def main() -> None:
    result = run_benchmark()
    print(f"Spectral-norm (N={result['n']})")
    print(f"  Result: {result['spectral_norm']}")


if __name__ == "__main__":
    main()
