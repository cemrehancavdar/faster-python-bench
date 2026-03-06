"""N-body simulation — Numba JIT implementation.

Algorithm: Direct translation of the Benchmarks Game n-body problem.
The physics and constants are identical to the official CPython baseline
(https://benchmarksgame-team.pages.debian.net/benchmarksgame/).

Implementation: Restructured for Numba — body state stored in contiguous
numpy arrays instead of Python lists, @njit decorator for LLVM compilation.
No community Numba n-body implementations exist (the Benchmarks Game only
accepts pure-language submissions), so this was written as a mechanical
translation of the baseline algorithm.

Correctness: Verified against official expected outputs (verify_correctness.py):
  N=1000:  energy_before=-0.169075164, energy_after=-0.169087605  [PASS]
  N=10000: energy_before=-0.169075164, energy_after=-0.169016441  [PASS]
"""

from __future__ import annotations

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from baseline.nbody import DEFAULT_N, SOLAR_MASS, make_system

# Number of bodies in the Jovian system
NUM_BODIES: int = 5

# Number of unique pairs: C(5,2) = 10
NUM_PAIRS: int = 10


def system_to_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert the Python body representation to flat numpy arrays.

    Returns:
        x, y, z: position arrays (NUM_BODIES,)
        vx, vy, vz packed into vel: velocity array (NUM_BODIES, 3)
        mass: mass array (NUM_BODIES,)
        pairs_i, pairs_j: pair index arrays (NUM_PAIRS,)
    """
    bodies, _pairs = make_system()

    # Offset momentum first (modifies sun's velocity)
    px = py = pz = 0.0
    for _r, v, m in bodies:
        px -= v[0] * m
        py -= v[1] * m
        pz -= v[2] * m
    bodies[0][1][0] = px / SOLAR_MASS
    bodies[0][1][1] = py / SOLAR_MASS
    bodies[0][1][2] = pz / SOLAR_MASS

    pos = np.zeros((NUM_BODIES, 3), dtype=np.float64)
    vel = np.zeros((NUM_BODIES, 3), dtype=np.float64)
    mass = np.zeros(NUM_BODIES, dtype=np.float64)

    for i, (r, v, m) in enumerate(bodies):
        pos[i] = r
        vel[i] = v
        mass[i] = m

    # Pre-compute pair indices
    pairs_i = np.zeros(NUM_PAIRS, dtype=np.int32)
    pairs_j = np.zeros(NUM_PAIRS, dtype=np.int32)
    k = 0
    for i in range(NUM_BODIES):
        for j in range(i + 1, NUM_BODIES):
            pairs_i[k] = i
            pairs_j[k] = j
            k += 1

    return pos, vel, mass, pairs_i, pairs_j


@njit(cache=True)
def _advance(
    dt: float,
    n: int,
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
) -> None:
    """Advance the simulation by n timesteps. Compiled to machine code by Numba."""
    num_pairs = len(pairs_i)
    num_bodies = len(mass)

    for _ in range(n):
        for k in range(num_pairs):
            i = pairs_i[k]
            j = pairs_j[k]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dsq = dx * dx + dy * dy + dz * dz
            mag = dt / (dsq * np.sqrt(dsq))
            b1m = mass[i] * mag
            b2m = mass[j] * mag
            vel[i, 0] -= dx * b2m
            vel[i, 1] -= dy * b2m
            vel[i, 2] -= dz * b2m
            vel[j, 0] += dx * b1m
            vel[j, 1] += dy * b1m
            vel[j, 2] += dz * b1m

        for i in range(num_bodies):
            pos[i, 0] += dt * vel[i, 0]
            pos[i, 1] += dt * vel[i, 1]
            pos[i, 2] += dt * vel[i, 2]


@njit(cache=True)
def _energy(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
) -> float:
    """Compute total energy. Compiled by Numba."""
    e = 0.0
    num_pairs = len(pairs_i)
    num_bodies = len(mass)

    for k in range(num_pairs):
        i = pairs_i[k]
        j = pairs_j[k]
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dz = pos[i, 2] - pos[j, 2]
        e -= (mass[i] * mass[j]) / np.sqrt(dx * dx + dy * dy + dz * dz)

    for i in range(num_bodies):
        e += mass[i] * (vel[i, 0] ** 2 + vel[i, 1] ** 2 + vel[i, 2] ** 2) / 2.0

    return e


def warmup() -> None:
    """Trigger Numba compilation before timing."""
    pos, vel, mass, pi, pj = system_to_arrays()
    _advance(0.01, 1, pos, vel, mass, pi, pj)
    _energy(pos, vel, mass, pi, pj)


def run_benchmark(n: int = DEFAULT_N) -> dict[str, object]:
    """Run the n-body benchmark with Numba JIT."""
    pos, vel, mass, pairs_i, pairs_j = system_to_arrays()
    e_before = _energy(pos, vel, mass, pairs_i, pairs_j)
    _advance(0.01, n, pos, vel, mass, pairs_i, pairs_j)
    e_after = _energy(pos, vel, mass, pairs_i, pairs_j)
    return {
        "n": n,
        "energy_before": round(float(e_before), 9),
        "energy_after": round(float(e_after), 9),
    }


def main() -> None:
    warmup()
    result = run_benchmark()
    print(f"N-body Numba ({result['n']} iterations)")
    print(f"  Energy before: {result['energy_before']}")
    print(f"  Energy after:  {result['energy_after']}")


if __name__ == "__main__":
    main()
