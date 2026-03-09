"""N-body simulation — Codon implementation.

Direct translation of the Benchmarks Game n-body problem for the Codon compiler.
Algorithm and constants are identical to the CPython baseline.

Codon constraints: no f-strings, no json, no sys.implementation.

Build: ~/.codon/bin/codon build -release -o bench_codon codon_benchmark/bench.py
Run:   DYLD_LIBRARY_PATH=~/.codon/lib/codon ./bench_codon
"""

import math

# ---------------------------------------------------------------------------
# Constants — same as the Benchmarks Game specification
# ---------------------------------------------------------------------------

PI = 3.14159265358979323
SOLAR_MASS = 4.0 * PI * PI
DAYS_PER_YEAR = 365.24
DEFAULT_N = 500_000


# ---------------------------------------------------------------------------
# System setup
# ---------------------------------------------------------------------------


def make_system():
    sun = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS)
    jupiter = (
        [4.84143144246472090e00, -1.16032004402742839e00, -1.03622044471123109e-01],
        [
            1.66007664274403694e-03 * DAYS_PER_YEAR,
            7.69901118419740425e-03 * DAYS_PER_YEAR,
            -6.90460016972063023e-05 * DAYS_PER_YEAR,
        ],
        9.54791938424326609e-04 * SOLAR_MASS,
    )
    saturn = (
        [8.34336671824457987e00, 4.12479856412430479e00, -4.03523417114321381e-01],
        [
            -2.76742510726862411e-03 * DAYS_PER_YEAR,
            4.99852801234917238e-03 * DAYS_PER_YEAR,
            2.30417297573763929e-05 * DAYS_PER_YEAR,
        ],
        2.85885980666130812e-04 * SOLAR_MASS,
    )
    uranus = (
        [1.28943695621391310e01, -1.51111514016986312e01, -2.23307578892655734e-01],
        [
            2.96460137564761618e-03 * DAYS_PER_YEAR,
            2.37847173959480950e-03 * DAYS_PER_YEAR,
            -2.96589568540237556e-05 * DAYS_PER_YEAR,
        ],
        4.36624404335156298e-05 * SOLAR_MASS,
    )
    neptune = (
        [1.53796971148509165e01, -2.59193146099879641e01, 1.79258772950371181e-01],
        [
            2.68067772490389322e-03 * DAYS_PER_YEAR,
            1.62824170038242295e-03 * DAYS_PER_YEAR,
            -9.51592254519715870e-05 * DAYS_PER_YEAR,
        ],
        5.15138902046611451e-05 * SOLAR_MASS,
    )
    bodies = [sun, jupiter, saturn, uranus, neptune]
    pairs = []
    for x in range(len(bodies) - 1):
        for y in range(x + 1, len(bodies)):
            pairs.append((bodies[x], bodies[y]))
    return bodies, pairs


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def advance(dt: float, n: int, bodies, pairs):
    for _ in range(n):
        for pair in pairs:
            r1, v1, m1 = pair[0]
            r2, v2, m2 = pair[1]
            dx = r1[0] - r2[0]
            dy = r1[1] - r2[1]
            dz = r1[2] - r2[2]
            mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
            b1m = m1 * mag
            b2m = m2 * mag
            v1[0] -= dx * b2m
            v1[1] -= dy * b2m
            v1[2] -= dz * b2m
            v2[0] += dx * b1m
            v2[1] += dy * b1m
            v2[2] += dz * b1m
        for body in bodies:
            r = body[0]
            v = body[1]
            r[0] += dt * v[0]
            r[1] += dt * v[1]
            r[2] += dt * v[2]


def energy(bodies, pairs) -> float:
    e = 0.0
    for pair in pairs:
        r1, _v1, m1 = pair[0]
        r2, _v2, m2 = pair[1]
        dx = r1[0] - r2[0]
        dy = r1[1] - r2[1]
        dz = r1[2] - r2[2]
        e -= (m1 * m2) / math.sqrt(dx * dx + dy * dy + dz * dz)
    for body in bodies:
        v = body[1]
        m = body[2]
        e += m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / 2.0
    return e


def run_nbody(n: int = DEFAULT_N) -> tuple[float, float]:
    """Run n-body simulation. Returns (energy_before, energy_after)."""
    bodies, pairs = make_system()
    px = 0.0
    py = 0.0
    pz = 0.0
    for body in bodies:
        v = body[1]
        m = body[2]
        px -= v[0] * m
        py -= v[1] * m
        pz -= v[2] * m
    bodies[0][1][0] = px / SOLAR_MASS
    bodies[0][1][1] = py / SOLAR_MASS
    bodies[0][1][2] = pz / SOLAR_MASS

    e_before = energy(bodies, pairs)
    advance(0.01, n, bodies, pairs)
    e_after = energy(bodies, pairs)
    return (round(e_before, 9), round(e_after, 9))
