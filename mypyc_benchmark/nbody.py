"""Mypyc N-body: type-annotated Python compiled to C.

The baseline is already well-typed. Key mypyc optimizations:
- Explicit type annotations on all locals for C primitive operations
- sqrt() + arithmetic instead of ** (-1.5) (standard numerical optimization)
- Pre-declared loop variables

Build: uv run python -c "from mypyc.build import mypycify; from setuptools import setup; setup(packages=[], ext_modules=mypycify(['mypyc_benchmark/nbody.py']))" build_ext --inplace
"""

from __future__ import annotations

import math

PI: float = 3.14159265358979323
SOLAR_MASS: float = 4 * PI * PI
DAYS_PER_YEAR: float = 365.24

DEFAULT_N: int = 500_000

Body = tuple[list[float], list[float], float]
BodyPair = tuple[Body, Body]


def make_system() -> tuple[list[Body], list[BodyPair]]:
    """Create a fresh copy of the initial solar system state."""
    sun: Body = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS)

    jupiter: Body = (
        [4.84143144246472090e00, -1.16032004402742839e00, -1.03622044471123109e-01],
        [
            1.66007664274403694e-03 * DAYS_PER_YEAR,
            7.69901118419740425e-03 * DAYS_PER_YEAR,
            -6.90460016972063023e-05 * DAYS_PER_YEAR,
        ],
        9.54791938424326609e-04 * SOLAR_MASS,
    )

    saturn: Body = (
        [8.34336671824457987e00, 4.12479856412430479e00, -4.03523417114321381e-01],
        [
            -2.76742510726862411e-03 * DAYS_PER_YEAR,
            4.99852801234917238e-03 * DAYS_PER_YEAR,
            2.30417297573763929e-05 * DAYS_PER_YEAR,
        ],
        2.85885980666130812e-04 * SOLAR_MASS,
    )

    uranus: Body = (
        [1.28943695621391310e01, -1.51111514016986312e01, -2.23307578892655734e-01],
        [
            2.96460137564761618e-03 * DAYS_PER_YEAR,
            2.37847173959480950e-03 * DAYS_PER_YEAR,
            -2.96589568540237556e-05 * DAYS_PER_YEAR,
        ],
        4.36624404335156298e-05 * SOLAR_MASS,
    )

    neptune: Body = (
        [1.53796971148509165e01, -2.59193146099879641e01, 1.79258772950371181e-01],
        [
            2.68067772490389322e-03 * DAYS_PER_YEAR,
            1.62824170038242295e-03 * DAYS_PER_YEAR,
            -9.51592254519715870e-05 * DAYS_PER_YEAR,
        ],
        5.15138902046611451e-05 * SOLAR_MASS,
    )

    bodies: list[Body] = [sun, jupiter, saturn, uranus, neptune]

    pairs: list[BodyPair] = []
    x: int
    y: int
    for x in range(len(bodies) - 1):
        for y in range(x + 1, len(bodies)):
            pairs.append((bodies[x], bodies[y]))

    return bodies, pairs


def advance(dt: float, n: int, bodies: list[Body], pairs: list[BodyPair]) -> None:
    """Advance the simulation by n timesteps of size dt."""
    _: int
    dx: float
    dy: float
    dz: float
    dist_sq: float
    dist: float
    mag: float
    b1m: float
    b2m: float

    for _ in range(n):
        for (r1, v1, m1), (r2, v2, m2) in pairs:
            dx = r1[0] - r2[0]
            dy = r1[1] - r2[1]
            dz = r1[2] - r2[2]
            dist_sq = dx * dx + dy * dy + dz * dz
            dist = math.sqrt(dist_sq)
            mag = dt / (dist_sq * dist)
            b1m = m1 * mag
            b2m = m2 * mag
            v1[0] -= dx * b2m
            v1[1] -= dy * b2m
            v1[2] -= dz * b2m
            v2[0] += dx * b1m
            v2[1] += dy * b1m
            v2[2] += dz * b1m
        for r, (vx, vy, vz), _m in bodies:
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz


def energy(bodies: list[Body], pairs: list[BodyPair]) -> float:
    """Compute total energy of the system."""
    e: float = 0.0
    dx: float
    dy: float
    dz: float

    for (r1, _v1, m1), (r2, _v2, m2) in pairs:
        dx = r1[0] - r2[0]
        dy = r1[1] - r2[1]
        dz = r1[2] - r2[2]
        e -= (m1 * m2) / math.sqrt(dx * dx + dy * dy + dz * dz)
    for _r, v, m in bodies:
        e += m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / 2.0
    return e


def offset_momentum(bodies: list[Body]) -> None:
    """Adjust sun's velocity so total momentum is zero."""
    px: float = 0.0
    py: float = 0.0
    pz: float = 0.0
    for _r, v, m in bodies:
        px -= v[0] * m
        py -= v[1] * m
        pz -= v[2] * m
    sun: Body = bodies[0]
    sun[1][0] = px / SOLAR_MASS
    sun[1][1] = py / SOLAR_MASS
    sun[1][2] = pz / SOLAR_MASS


def run_benchmark(n: int = DEFAULT_N) -> dict[str, object]:
    """Run the n-body benchmark. Returns energy before and after."""
    bodies: list[Body]
    pairs: list[BodyPair]
    bodies, pairs = make_system()
    offset_momentum(bodies)
    e_before: float = energy(bodies, pairs)
    advance(0.01, n, bodies, pairs)
    e_after: float = energy(bodies, pairs)
    return {
        "n": n,
        "energy_before": round(e_before, 9),
        "energy_after": round(e_after, 9),
    }


EXPECTED_ENERGY_BEFORE: float = -0.169075164
EXPECTED_ENERGY_AFTER_500K: float = -0.169078071


def main() -> None:
    result = run_benchmark()
    print(f"N-body ({result['n']} iterations)")
    print(f"  Energy before: {result['energy_before']}")
    print(f"  Energy after:  {result['energy_after']}")


if __name__ == "__main__":
    main()
