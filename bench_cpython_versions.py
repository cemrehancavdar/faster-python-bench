"""Benchmark CPython versions on n-body and spectral-norm baselines.

Self-contained — no imports beyond stdlib. Runs with any CPython 3.10+.

Usage:
    uv run --python 3.10 bench_cpython_versions.py
    uv run --python 3.14 bench_cpython_versions.py
"""

import json
import math
import statistics
import sys
import time

# ---------------------------------------------------------------------------
# N-body (inline — no external imports)
# ---------------------------------------------------------------------------

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24
NBODY_N = 500_000


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


def nbody_advance(dt, n, bodies, pairs):
    for _ in range(n):
        for (r1, v1, m1), (r2, v2, m2) in pairs:
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
        for r, (vx, vy, vz), _m in bodies:
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz


def nbody_energy(bodies, pairs):
    e = 0.0
    for (r1, _v1, m1), (r2, _v2, m2) in pairs:
        dx = r1[0] - r2[0]
        dy = r1[1] - r2[1]
        dz = r1[2] - r2[2]
        e -= (m1 * m2) / math.sqrt(dx * dx + dy * dy + dz * dz)
    for _r, v, m in bodies:
        e += m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / 2.0
    return e


def run_nbody():
    bodies, pairs = make_system()
    # Offset momentum
    px = py = pz = 0.0
    for _r, v, m in bodies:
        px -= v[0] * m
        py -= v[1] * m
        pz -= v[2] * m
    bodies[0][1][0] = px / SOLAR_MASS
    bodies[0][1][1] = py / SOLAR_MASS
    bodies[0][1][2] = pz / SOLAR_MASS

    e_before = nbody_energy(bodies, pairs)
    nbody_advance(0.01, NBODY_N, bodies, pairs)
    e_after = nbody_energy(bodies, pairs)
    return round(e_before, 9), round(e_after, 9)


# ---------------------------------------------------------------------------
# Spectral-norm (inline — no external imports)
# ---------------------------------------------------------------------------

SPECTRAL_N = 2000


def eval_A(i, j):
    ij = i + j
    return 1.0 / (ij * (ij + 1) / 2 + i + 1)


def eval_A_times_u(u):
    local_eval_A = eval_A
    return [sum(local_eval_A(i, j) * u_j for j, u_j in enumerate(u)) for i in range(len(u))]


def eval_At_times_u(u):
    local_eval_A = eval_A
    return [sum(local_eval_A(j, i) * u_j for j, u_j in enumerate(u)) for i in range(len(u))]


def eval_AtA_times_u(u):
    return eval_At_times_u(eval_A_times_u(u))


def run_spectral():
    u = [1.0] * SPECTRAL_N
    v = [0.0] * SPECTRAL_N
    for _ in range(10):
        v = eval_AtA_times_u(u)
        u = eval_AtA_times_u(v)
    vBv = 0.0
    vv = 0.0
    for ue, ve in zip(u, v):
        vBv += ue * ve
        vv += ve * ve
    return round(math.sqrt(vBv / vv), 9)


# ---------------------------------------------------------------------------
# Main — run both benchmarks, report times
# ---------------------------------------------------------------------------

RUNS = 5


def bench(name, fn, expected_check):
    # Warmup
    result = fn()
    if not expected_check(result):
        print(f"  {name}: CORRECTNESS FAILURE: {result}")
        return None

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    med = statistics.median(times)
    return med


def main():
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    impl = sys.implementation.name
    print(f"Python {ver} ({impl})")
    print(f"Runs: {RUNS}")
    print()

    results = {"python_version": ver, "implementation": impl}

    print(f"  n-body ({NBODY_N} iterations)...", end=" ", flush=True)
    nbody_ms = bench(
        "nbody",
        run_nbody,
        lambda r: abs(r[0] - (-0.169075164)) < 1e-6,
    )
    if nbody_ms is not None:
        print(f"{nbody_ms:.1f}ms")
        results["nbody_ms"] = round(nbody_ms, 1)

    print(f"  spectral-norm (N={SPECTRAL_N})...", end=" ", flush=True)
    spectral_ms = bench(
        "spectral",
        run_spectral,
        lambda r: abs(r - 1.274224153) < 1e-6,
    )
    if spectral_ms is not None:
        print(f"{spectral_ms:.1f}ms")
        results["spectral_ms"] = round(spectral_ms, 1)

    print()
    print(json.dumps(results))


if __name__ == "__main__":
    main()
