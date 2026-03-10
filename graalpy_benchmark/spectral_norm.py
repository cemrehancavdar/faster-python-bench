"""Spectral-norm — GraalPy (identical to baseline).

GraalPy's JIT compiler (GraalVM Truffle) accelerates the same pure Python
code with zero changes. Requires warmup runs for the JIT to kick in.

Install: uv python install graalpy-3.12
Venv:    uv venv /tmp/graalpy-venv --python graalpy-3.12
Run:     /tmp/graalpy-venv/bin/python -c "from graalpy_benchmark.spectral_norm import run_benchmark; ..."
"""

from __future__ import annotations

import math

DEFAULT_N: int = 2000


def eval_A(i: int, j: int) -> float:
    """Element A[i][j] of the infinite matrix."""
    ij = i + j
    return 1.0 / (ij * (ij + 1) / 2 + i + 1)


def eval_A_times_u(u: list[float]) -> list[float]:
    """Multiply vector u by matrix A."""
    local_eval_A = eval_A
    return [sum(local_eval_A(i, j) * u_j for j, u_j in enumerate(u)) for i in range(len(u))]


def eval_At_times_u(u: list[float]) -> list[float]:
    """Multiply vector u by A-transpose."""
    local_eval_A = eval_A
    return [sum(local_eval_A(j, i) * u_j for j, u_j in enumerate(u)) for i in range(len(u))]


def eval_AtA_times_u(u: list[float]) -> list[float]:
    """Multiply vector u by A^T * A (one power iteration step)."""
    return eval_At_times_u(eval_A_times_u(u))


def run_benchmark(n: int = DEFAULT_N) -> dict[str, object]:
    """Run the spectral-norm benchmark."""
    u: list[float] = [1.0] * n
    v: list[float] = [0.0] * n

    for _ in range(10):
        v = eval_AtA_times_u(u)
        u = eval_AtA_times_u(v)

    vBv = 0.0
    vv = 0.0
    for ue, ve in zip(u, v):
        vBv += ue * ve
        vv += ve * ve

    result = math.sqrt(vBv / vv)
    return {
        "n": n,
        "spectral_norm": round(result, 9),
    }


if __name__ == "__main__":
    result = run_benchmark()
    print(f"Spectral-norm (N={result['n']})")
    print(f"  Result: {result['spectral_norm']}")
