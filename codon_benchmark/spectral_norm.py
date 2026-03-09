"""Spectral-norm — Codon implementation.

Direct translation of the Benchmarks Game spectral-norm problem for the Codon compiler.
Algorithm is identical to the CPython baseline.

Codon constraints: no f-strings, no json, no sys.implementation.

Build: ~/.codon/bin/codon build -release -o bench_codon codon_benchmark/bench.py
Run:   DYLD_LIBRARY_PATH=~/.codon/lib/codon ./bench_codon
"""

import math

# Default N — 2000 for local benchmarking.
DEFAULT_N = 2000


def eval_A(i: int, j: int) -> float:
    """Element A[i][j] of the infinite matrix."""
    ij = i + j
    return 1.0 / (ij * (ij + 1) / 2 + i + 1)


def eval_A_times_u(u: list[float]) -> list[float]:
    n = len(u)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += eval_A(i, j) * u[j]
        result[i] = s
    return result


def eval_At_times_u(u: list[float]) -> list[float]:
    n = len(u)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += eval_A(j, i) * u[j]
        result[i] = s
    return result


def eval_AtA_times_u(u: list[float]) -> list[float]:
    return eval_At_times_u(eval_A_times_u(u))


def run_spectral(n: int = DEFAULT_N) -> float:
    """Run spectral-norm computation. Returns the spectral norm value."""
    u = [1.0] * n
    v = [0.0] * n
    for _ in range(10):
        v = eval_AtA_times_u(u)
        u = eval_AtA_times_u(v)
    vBv = 0.0
    vv = 0.0
    for i in range(len(u)):
        vBv += u[i] * v[i]
        vv += v[i] * v[i]
    return round(math.sqrt(vBv / vv), 9)
