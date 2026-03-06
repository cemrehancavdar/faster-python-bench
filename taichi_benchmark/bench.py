"""N-body and spectral-norm benchmarks for Taichi.

Taichi compiles @ti.kernel decorated functions to native machine code via LLVM.
Similar to Numba but with explicit kernel/func distinction and auto-parallelization.

Usage: uv run bench_taichi.py
"""

import statistics
import time

import taichi as ti

# Single-threaded CPU to compare fairly with other single-threaded benchmarks
ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)

# ---------------------------------------------------------------------------
# N-body
# ---------------------------------------------------------------------------

NUM_BODIES = 5
NBODY_N = 500_000

# Body state as Taichi fields
nbody_x = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_y = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_z = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_vx = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_vy = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_vz = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_mass = ti.field(dtype=ti.f64, shape=NUM_BODIES)


@ti.kernel
def nbody_advance(n: ti.i32):
    dt: ti.f64 = 0.01
    ti.loop_config(serialize=True)
    for _step in range(n):
        for i in range(NUM_BODIES):
            for j in range(i + 1, NUM_BODIES):
                dx = nbody_x[i] - nbody_x[j]
                dy = nbody_y[i] - nbody_y[j]
                dz = nbody_z[i] - nbody_z[j]
                dsq = dx * dx + dy * dy + dz * dz
                mag = dt / (dsq * ti.sqrt(dsq))
                mi = nbody_mass[i] * mag
                mj = nbody_mass[j] * mag
                nbody_vx[i] -= dx * mj
                nbody_vy[i] -= dy * mj
                nbody_vz[i] -= dz * mj
                nbody_vx[j] += dx * mi
                nbody_vy[j] += dy * mi
                nbody_vz[j] += dz * mi
        for i in range(NUM_BODIES):
            nbody_x[i] += dt * nbody_vx[i]
            nbody_y[i] += dt * nbody_vy[i]
            nbody_z[i] += dt * nbody_vz[i]


@ti.kernel
def nbody_energy() -> ti.f64:
    e: ti.f64 = 0.0
    for i in range(NUM_BODIES):
        for j in range(i + 1, NUM_BODIES):
            dx = nbody_x[i] - nbody_x[j]
            dy = nbody_y[i] - nbody_y[j]
            dz = nbody_z[i] - nbody_z[j]
            e -= (nbody_mass[i] * nbody_mass[j]) / ti.sqrt(dx * dx + dy * dy + dz * dz)
    for i in range(NUM_BODIES):
        e += nbody_mass[i] * (nbody_vx[i] ** 2 + nbody_vy[i] ** 2 + nbody_vz[i] ** 2) / 2.0
    return e


PI = 3.14159265358979323
SOLAR_MASS = 4.0 * PI * PI
DAYS_PER_YEAR = 365.24

# Initial positions/velocities/masses for the Jovian system
INIT_DATA = [
    # Sun
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS),
    # Jupiter
    (
        4.84143144246472090e00,
        -1.16032004402742839e00,
        -1.03622044471123109e-01,
        1.66007664274403694e-03 * DAYS_PER_YEAR,
        7.69901118419740425e-03 * DAYS_PER_YEAR,
        -6.90460016972063023e-05 * DAYS_PER_YEAR,
        9.54791938424326609e-04 * SOLAR_MASS,
    ),
    # Saturn
    (
        8.34336671824457987e00,
        4.12479856412430479e00,
        -4.03523417114321381e-01,
        -2.76742510726862411e-03 * DAYS_PER_YEAR,
        4.99852801234917238e-03 * DAYS_PER_YEAR,
        2.30417297573763929e-05 * DAYS_PER_YEAR,
        2.85885980666130812e-04 * SOLAR_MASS,
    ),
    # Uranus
    (
        1.28943695621391310e01,
        -1.51111514016986312e01,
        -2.23307578892655734e-01,
        2.96460137564761618e-03 * DAYS_PER_YEAR,
        2.37847173959480950e-03 * DAYS_PER_YEAR,
        -2.96589568540237556e-05 * DAYS_PER_YEAR,
        4.36624404335156298e-05 * SOLAR_MASS,
    ),
    # Neptune
    (
        1.53796971148509165e01,
        -2.59193146099879641e01,
        1.79258772950371181e-01,
        2.68067772490389322e-03 * DAYS_PER_YEAR,
        1.62824170038242295e-03 * DAYS_PER_YEAR,
        -9.51592254519715870e-05 * DAYS_PER_YEAR,
        5.15138902046611451e-05 * SOLAR_MASS,
    ),
]


def init_nbody():
    """Load initial state into Taichi fields and offset momentum."""
    for i, (x, y, z, vx, vy, vz, m) in enumerate(INIT_DATA):
        nbody_x[i] = x
        nbody_y[i] = y
        nbody_z[i] = z
        nbody_vx[i] = vx
        nbody_vy[i] = vy
        nbody_vz[i] = vz
        nbody_mass[i] = m

    # Offset momentum
    px = py = pz = 0.0
    for i in range(NUM_BODIES):
        px -= nbody_vx[i] * nbody_mass[i]
        py -= nbody_vy[i] * nbody_mass[i]
        pz -= nbody_vz[i] * nbody_mass[i]
    nbody_vx[0] = px / SOLAR_MASS
    nbody_vy[0] = py / SOLAR_MASS
    nbody_vz[0] = pz / SOLAR_MASS


def run_nbody():
    init_nbody()
    e_before = nbody_energy()
    nbody_advance(NBODY_N)
    e_after = nbody_energy()
    return round(e_before, 9), round(e_after, 9)


# ---------------------------------------------------------------------------
# Spectral-norm
# ---------------------------------------------------------------------------

SPECTRAL_N = 2000

sn_u = ti.field(dtype=ti.f64, shape=SPECTRAL_N)
sn_v = ti.field(dtype=ti.f64, shape=SPECTRAL_N)
sn_tmp = ti.field(dtype=ti.f64, shape=SPECTRAL_N)


@ti.func
def sn_eval_A(i: ti.i32, j: ti.i32) -> ti.f64:
    ij = i + j
    return 1.0 / (ij * (ij + 1) / 2 + i + 1)


@ti.kernel
def sn_eval_AtA_times_u(u: ti.template(), out: ti.template(), tmp: ti.template()):
    """Compute out = A^T * A * u."""
    # tmp = A * u
    ti.loop_config(serialize=True)
    for i in range(SPECTRAL_N):
        s: ti.f64 = 0.0
        for j in range(SPECTRAL_N):
            s += sn_eval_A(i, j) * u[i, j]  # u[j] but need field access
        tmp[i] = s
    # out = A^T * tmp
    ti.loop_config(serialize=True)
    for i in range(SPECTRAL_N):
        s: ti.f64 = 0.0
        for j in range(SPECTRAL_N):
            s += sn_eval_A(j, i) * tmp[j]
        out[i] = s


@ti.kernel
def sn_full_run() -> ti.f64:
    """Run the entire spectral-norm computation inside one kernel."""
    n: ti.i32 = SPECTRAL_N

    # Initialize u = [1, 1, ...], v = [0, 0, ...]
    ti.loop_config(serialize=True)
    for i in range(n):
        sn_u[i] = 1.0
        sn_v[i] = 0.0

    # 10 power iterations
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


def run_spectral() -> float:
    result = sn_full_run()
    return round(result, 9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUNS = 5


def main():
    print("Taichi 1.7.4 (CPU, single-threaded)")
    print(f"Runs: {RUNS}")
    print()

    # Warmup (triggers JIT compilation)
    run_nbody()
    run_spectral()

    # N-body
    print(f"  n-body ({NBODY_N} iterations)...", end=" ", flush=True)
    times_nb: list[float] = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        run_nbody()
        t1 = time.perf_counter()
        times_nb.append((t1 - t0) * 1000)
    med_nb = statistics.median(times_nb)
    print(f"{med_nb:.1f}ms")

    # Spectral-norm
    print(f"  spectral-norm (N={SPECTRAL_N})...", end=" ", flush=True)
    times_sn: list[float] = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        run_spectral()
        t1 = time.perf_counter()
        times_sn.append((t1 - t0) * 1000)
    med_sn = statistics.median(times_sn)
    print(f"{med_sn:.1f}ms")

    # Correctness
    e_before, e_after = run_nbody()
    sn = run_spectral()
    print()
    print("Correctness:")
    print(f"  nbody energy_before: {e_before}")
    print(f"  nbody energy_after:  {e_after}")
    print(f"  spectral_norm:       {sn}")


if __name__ == "__main__":
    main()
