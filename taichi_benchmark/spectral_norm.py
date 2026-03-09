"""Spectral-norm — Taichi implementation.

Taichi compiles @ti.kernel decorated functions to native machine code via LLVM.
The entire spectral-norm computation runs inside a single kernel to avoid
Python-to-Taichi dispatch overhead on each power iteration.

Algorithm is identical to the CPython baseline.

Usage: /tmp/taichi-venv/bin/python taichi_benchmark/bench.py

Requires: taichi (Python 3.13 — no 3.14 wheels available).
ti.init() must be called before importing this module.
"""

import taichi as ti

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N = 2000

# ---------------------------------------------------------------------------
# Taichi fields
# ---------------------------------------------------------------------------

sn_u = ti.field(dtype=ti.f64, shape=DEFAULT_N)
sn_v = ti.field(dtype=ti.f64, shape=DEFAULT_N)
sn_tmp = ti.field(dtype=ti.f64, shape=DEFAULT_N)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@ti.func
def sn_eval_A(i: ti.i32, j: ti.i32) -> ti.f64:
    ij = i + j
    return 1.0 / (ij * (ij + 1) / 2 + i + 1)


@ti.kernel
def sn_full_run() -> ti.f64:
    """Run the entire spectral-norm computation inside one kernel."""
    n: ti.i32 = DEFAULT_N

    # Initialize u = [1, 1, ...], v = [0, 0, ...]
    ti.loop_config(serialize=True)
    for i in range(n):
        sn_u[i] = 1.0
        sn_v[i] = 0.0

    # 10 power iterations (must be serial — each depends on the previous)
    ti.loop_config(serialize=True)
    for _iter in range(10):
        # v = A^T * A * u
        # tmp = A * u
        ti.loop_config(serialize=True)
        for i in range(n):
            s: ti.f64 = 0.0
            for j in range(n):
                s += sn_eval_A(i, j) * sn_u[j]
            sn_tmp[i] = s
        # v = A^T * tmp
        ti.loop_config(serialize=True)
        for i in range(n):
            s: ti.f64 = 0.0
            for j in range(n):
                s += sn_eval_A(j, i) * sn_tmp[j]
            sn_v[i] = s

        # u = A^T * A * v
        # tmp = A * v
        ti.loop_config(serialize=True)
        for i in range(n):
            s: ti.f64 = 0.0
            for j in range(n):
                s += sn_eval_A(i, j) * sn_v[j]
            sn_tmp[i] = s
        # u = A^T * tmp
        ti.loop_config(serialize=True)
        for i in range(n):
            s: ti.f64 = 0.0
            for j in range(n):
                s += sn_eval_A(j, i) * sn_tmp[j]
            sn_u[i] = s

    # Compute result
    vBv: ti.f64 = 0.0
    vv: ti.f64 = 0.0
    ti.loop_config(serialize=True)
    for i in range(n):
        vBv += sn_u[i] * sn_v[i]
        vv += sn_v[i] * sn_v[i]

    return ti.sqrt(vBv / vv)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def run_spectral(n: int = DEFAULT_N) -> float:
    """Run spectral-norm computation. Returns the spectral norm value.

    Note: n parameter is accepted for interface consistency but the kernel
    uses DEFAULT_N (field sizes are fixed at module load time).
    """
    result = sn_full_run()
    return round(result, 9)
