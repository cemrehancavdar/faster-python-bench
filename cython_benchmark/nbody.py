"""N-body simulation — Cython pure Python mode implementation.

Algorithm: Direct translation of the Benchmarks Game n-body problem.
The physics and constants are identical to the official CPython baseline
(https://benchmarksgame-team.pages.debian.net/benchmarksgame/).

Implementation: Uses Cython's pure Python mode with C-typed variables,
stack-allocated C arrays, libc sqrt, and nested i/j loops (not pair-index
arrays — the compiler can unroll the constant-bound loops). No community
Cython n-body implementations exist, so this was written as a translation
of the baseline with Cython type annotations.

Key insights discovered during development: decomposing ** (-1.5) into
sqrt() + arithmetic gives ~7x speedup (true in any language, not Cython-specific).
Using pair-index arrays instead of nested for loops costs 2x silently.
See blog post for the full DX story.

Correctness: Verified against official expected outputs (verify_correctness.py):
  N=1000:  energy_before=-0.169075164, energy_after=-0.169087605  [PASS]
  N=10000: energy_before=-0.169075164, energy_after=-0.169016441  [PASS]

Build: uv run --extra cython python cython_benchmark/setup_nbody.py build_ext --inplace
"""

from __future__ import annotations

import cython
from cython.cimports.libc.math import sqrt  # type: ignore[import-not-found]

from baseline.nbody import DEFAULT_N, SOLAR_MASS, make_system

# Max bodies — fixed at 5 for the Jovian system
MAX_BODIES: int = 5
MAX_PAIRS: int = 10


@cython.cfunc
@cython.cdivision(True)
def _run_simulation(n: cython.int) -> cython.double:
    """Initialize, advance, and return final energy — all in C.

    All body state lives in C arrays on the stack. This function does
    everything: initialize from Python data, run the simulation, compute
    energy. This avoids any Python object access in the hot loop.
    """
    # Body state as flat C arrays
    x: cython.double[5]
    y: cython.double[5]
    z: cython.double[5]
    vx: cython.double[5]
    vy: cython.double[5]
    vz: cython.double[5]
    mass: cython.double[5]

    # Pair indices
    pi: cython.int[10]
    pj: cython.int[10]

    # Initialize from Python data
    bodies, _pairs = make_system()

    # Offset momentum
    opx: cython.double = 0.0
    opy: cython.double = 0.0
    opz: cython.double = 0.0
    idx: cython.int
    for idx in range(5):
        r, v, m = bodies[idx]
        x[idx] = r[0]
        y[idx] = r[1]
        z[idx] = r[2]
        vx[idx] = v[0]
        vy[idx] = v[1]
        vz[idx] = v[2]
        mass[idx] = m
        opx -= v[0] * m
        opy -= v[1] * m
        opz -= v[2] * m

    sm: cython.double = SOLAR_MASS
    vx[0] = opx / sm
    vy[0] = opy / sm
    vz[0] = opz / sm

    # Build pair indices
    k: cython.int = 0
    i: cython.int
    j: cython.int
    for i in range(5):
        for j in range(i + 1, 5):
            pi[k] = i
            pj[k] = j
            k += 1

    # --- Hot loop: pure C arithmetic, no Python objects ---
    # Match Rust's structure: nested i/j loops with constant bounds.
    # This lets the compiler unroll and eliminate pair-index indirection.
    dt: cython.double = 0.01
    step: cython.int
    dx: cython.double
    dy: cython.double
    dz: cython.double
    dsq: cython.double
    mag: cython.double
    mi: cython.double
    mj: cython.double

    for step in range(n):
        for i in range(5):
            for j in range(i + 1, 5):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dz = z[i] - z[j]
                dsq = dx * dx + dy * dy + dz * dz
                mag = dt / (dsq * sqrt(dsq))
                mi = mass[i] * mag
                mj = mass[j] * mag
                vx[i] -= dx * mj
                vy[i] -= dy * mj
                vz[i] -= dz * mj
                vx[j] += dx * mi
                vy[j] += dy * mi
                vz[j] += dz * mi

        for i in range(5):
            x[i] += dt * vx[i]
            y[i] += dt * vy[i]
            z[i] += dt * vz[i]

    # Compute final energy
    e: cython.double = 0.0
    dist: cython.double
    for p in range(10):
        bi = pi[p]
        bj = pj[p]
        dx = x[bi] - x[bj]
        dy = y[bi] - y[bj]
        dz = z[bi] - z[bj]
        e -= (mass[bi] * mass[bj]) / sqrt(dx * dx + dy * dy + dz * dz)

    for i in range(5):
        e += mass[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]) / 2.0

    return e


@cython.ccall
def run_benchmark(n: cython.int = DEFAULT_N) -> dict:
    """Run the n-body benchmark with Cython compilation."""
    # Compute initial energy separately (before advance modifies state)
    bodies_init, pairs_init = make_system()

    # Offset momentum for energy_before calculation
    opx: cython.double = 0.0
    opy: cython.double = 0.0
    opz: cython.double = 0.0
    for _r, v, m in bodies_init:
        opx -= v[0] * m
        opy -= v[1] * m
        opz -= v[2] * m
    sm: cython.double = SOLAR_MASS
    bodies_init[0][1][0] = opx / sm
    bodies_init[0][1][1] = opy / sm
    bodies_init[0][1][2] = opz / sm

    e_before: cython.double = 0.0
    for (r1, _v1, m1), (r2, _v2, m2) in pairs_init:
        dx = r1[0] - r2[0]
        dy = r1[1] - r2[1]
        dz = r1[2] - r2[2]
        e_before -= (m1 * m2) / sqrt(dx * dx + dy * dy + dz * dz)
    for _r, v, m in bodies_init:
        e_before += m * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) / 2.0

    # Run advance + compute final energy — all in C
    e_after = _run_simulation(n)

    return {
        "n": n,
        "energy_before": round(e_before, 9),
        "energy_after": round(e_after, 9),
    }


if __name__ == "__main__":
    result = run_benchmark()
    print(f"N-body Cython ({result['n']} iterations)")
    print(f"  Energy before: {result['energy_before']}")
    print(f"  Energy after:  {result['energy_after']}")
