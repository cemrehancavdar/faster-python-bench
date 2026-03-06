"""Spectral-norm — Cython pure Python mode implementation.

Algorithm: Same spectral-norm problem from the Benchmarks Game
(https://benchmarksgame-team.pages.debian.net/benchmarksgame/).

Implementation: Cython pure Python mode with C-typed variables, malloc'd
arrays, and inlined eval_A function. All inner-loop arithmetic runs as
pure C with zero Python object overhead.

No community Cython spectral-norm exists, so this was written as a
translation of the baseline with Cython type annotations and C-level
memory management.

Correctness: Verified against official expected outputs (verify_correctness.py):
  N=2:   1.183350177  [PASS]
  N=100: 1.274219991  [PASS]

Build: uv run --extra cython python cython_benchmark/setup_spectral_norm.py build_ext --inplace
"""

from __future__ import annotations

import math

import cython
from cython.cimports.libc.stdlib import free, malloc  # type: ignore[import-not-found]

from baseline.spectral_norm import DEFAULT_N


@cython.cfunc
@cython.inline
@cython.exceptval(-1.0, check=False)
@cython.cdivision(True)
def _eval_A(i: cython.int, j: cython.int) -> cython.double:
    """Element A[i][j]. Pure C — no Python objects."""
    ij: cython.int = i + j
    return 1.0 / (ij * (ij + 1) / 2 + i + 1)


@cython.cfunc
@cython.cdivision(True)
def _eval_AtA_times_u(
    n: cython.int,
    u: cython.p_double,
    out: cython.p_double,
    tmp: cython.p_double,
) -> None:
    """Compute out = A^T * A * u. All C arrays — zero Python overhead."""
    i: cython.int
    j: cython.int
    s: cython.double

    # tmp = A * u
    for i in range(n):
        s = 0.0
        for j in range(n):
            s = s + _eval_A(i, j) * u[j]
        tmp[i] = s

    # out = A^T * tmp
    for i in range(n):
        s = 0.0
        for j in range(n):
            s = s + _eval_A(j, i) * tmp[j]
        out[i] = s


@cython.ccall
def run_benchmark(n: cython.int = DEFAULT_N) -> dict:
    """Run the spectral-norm benchmark with Cython compilation."""
    # Allocate C arrays via malloc — no Python objects in the hot path
    u: cython.p_double = cython.cast(cython.p_double, malloc(n * cython.sizeof(cython.double)))
    v: cython.p_double = cython.cast(cython.p_double, malloc(n * cython.sizeof(cython.double)))
    tmp: cython.p_double = cython.cast(cython.p_double, malloc(n * cython.sizeof(cython.double)))

    i: cython.int

    # Initialize u = [1.0, 1.0, ...]
    for i in range(n):
        u[i] = 1.0
        v[i] = 0.0
        tmp[i] = 0.0

    _dummy: cython.int
    for _dummy in range(10):
        _eval_AtA_times_u(n, u, v, tmp)
        _eval_AtA_times_u(n, v, u, tmp)

    vBv: cython.double = 0.0
    vv: cython.double = 0.0
    for i in range(n):
        vBv += u[i] * v[i]
        vv += v[i] * v[i]

    free(u)
    free(v)
    free(tmp)

    result = math.sqrt(vBv / vv)
    return {
        "n": n,
        "spectral_norm": round(result, 9),
    }


if __name__ == "__main__":
    result = run_benchmark()
    print(f"Spectral-norm Cython (N={result['n']})")
    print(f"  Result: {result['spectral_norm']}")
