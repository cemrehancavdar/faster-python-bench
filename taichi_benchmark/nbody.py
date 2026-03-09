"""N-body simulation — Taichi implementation.

Taichi compiles @ti.kernel decorated functions to native machine code via LLVM.
Body state is stored in Taichi fields (GPU-compatible arrays).

Algorithm and constants are identical to the CPython baseline.

Usage: /tmp/taichi-venv/bin/python taichi_benchmark/bench.py

Requires: taichi (Python 3.13 — no 3.14 wheels available).
ti.init() must be called before importing this module.
"""

import taichi as ti

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_BODIES = 5
DEFAULT_N = 500_000

PI = 3.14159265358979323
SOLAR_MASS = 4.0 * PI * PI
DAYS_PER_YEAR = 365.24

# ---------------------------------------------------------------------------
# Body state as Taichi fields
# ---------------------------------------------------------------------------

nbody_x = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_y = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_z = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_vx = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_vy = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_vz = ti.field(dtype=ti.f64, shape=NUM_BODIES)
nbody_mass = ti.field(dtype=ti.f64, shape=NUM_BODIES)

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


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@ti.kernel
def advance(n: ti.i32):
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
def energy() -> ti.f64:
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


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


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


def run_nbody(n: int = DEFAULT_N) -> tuple[float, float]:
    """Run n-body simulation. Returns (energy_before, energy_after)."""
    init_nbody()
    e_before = energy()
    advance(n)
    e_after = energy()
    return round(e_before, 9), round(e_after, 9)
