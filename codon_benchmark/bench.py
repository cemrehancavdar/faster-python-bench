"""N-body and spectral-norm benchmarks for Codon.

Codon-compatible: no json, no sys.implementation, no f-strings with expressions.
Build: ~/.codon/bin/codon build -release -o bench_codon codon_benchmark/bench.py
Run:   DYLD_LIBRARY_PATH=~/.codon/lib/codon ./bench_codon
"""

import math
import time

# ---------------------------------------------------------------------------
# N-body
# ---------------------------------------------------------------------------

PI = 3.14159265358979323
SOLAR_MASS = 4.0 * PI * PI
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


def nbody_advance(dt: float, n: int, bodies, pairs):
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


def nbody_energy(bodies, pairs) -> float:
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


def run_nbody() -> tuple[float, float]:
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

    e_before = nbody_energy(bodies, pairs)
    nbody_advance(0.01, NBODY_N, bodies, pairs)
    e_after = nbody_energy(bodies, pairs)
    return (round(e_before, 9), round(e_after, 9))


# ---------------------------------------------------------------------------
# Spectral-norm
# ---------------------------------------------------------------------------

SPECTRAL_N = 2000


def eval_A(i: int, j: int) -> float:
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


def run_spectral() -> float:
    u = [1.0] * SPECTRAL_N
    v = [0.0] * SPECTRAL_N
    for _ in range(10):
        v = eval_AtA_times_u(u)
        u = eval_AtA_times_u(v)
    vBv = 0.0
    vv = 0.0
    for i in range(len(u)):
        vBv += u[i] * v[i]
        vv += v[i] * v[i]
    return round(math.sqrt(vBv / vv), 9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUNS = 5


def main():
    print("Codon 0.19.6")
    print("Runs: " + str(RUNS))
    print("")

    # N-body
    print("  n-body (" + str(NBODY_N) + " iterations)... ", end="")
    # warmup
    run_nbody()

    times_nb = []
    for _ in range(RUNS):
        t0 = time.time()
        run_nbody()
        t1 = time.time()
        times_nb.append((t1 - t0) * 1000.0)

    times_nb.sort()
    med_nb = times_nb[RUNS // 2]
    print(str(round(med_nb, 1)) + "ms")

    # Spectral-norm
    print("  spectral-norm (N=" + str(SPECTRAL_N) + ")... ", end="")
    # warmup
    run_spectral()

    times_sn = []
    for _ in range(RUNS):
        t0 = time.time()
        run_spectral()
        t1 = time.time()
        times_sn.append((t1 - t0) * 1000.0)

    times_sn.sort()
    med_sn = times_sn[RUNS // 2]
    print(str(round(med_sn, 1)) + "ms")

    # Correctness check
    e_before, e_after = run_nbody()
    sn = run_spectral()
    print("")
    print("Correctness:")
    print("  nbody energy_before: " + str(e_before))
    print("  spectral_norm: " + str(sn))


main()
