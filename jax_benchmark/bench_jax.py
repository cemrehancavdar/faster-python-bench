"""JAX JIT benchmarks: n-body and spectral-norm.

Tests jax.jit on CPU for the same problems as the blog post.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# Force CPU
jax.config.update("jax_platform_name", "cpu")

# Use float64 for correctness comparison
jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI: float = 3.14159265358979323
SOLAR_MASS: float = 4 * PI * PI
DAYS_PER_YEAR: float = 365.24

# ---------------------------------------------------------------------------
# N-body
# ---------------------------------------------------------------------------


def make_nbody_system():
    """Create initial state as JAX arrays."""
    pos = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [4.84143144246472090e00, -1.16032004402742839e00, -1.03622044471123109e-01],
            [8.34336671824457987e00, 4.12479856412430479e00, -4.03523417114321381e-01],
            [1.28943695621391310e01, -1.51111514016986312e01, -2.23307578892655734e-01],
            [1.53796971148509165e01, -2.59193146099879641e01, 1.79258772950371181e-01],
        ],
        dtype=jnp.float64,
    )

    vel = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [
                1.66007664274403694e-03 * DAYS_PER_YEAR,
                7.69901118419740425e-03 * DAYS_PER_YEAR,
                -6.90460016972063023e-05 * DAYS_PER_YEAR,
            ],
            [
                -2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR,
            ],
            [
                2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR,
            ],
            [
                2.68067772490389322e-03 * DAYS_PER_YEAR,
                1.62824170038242295e-03 * DAYS_PER_YEAR,
                -9.51592254519715870e-05 * DAYS_PER_YEAR,
            ],
        ],
        dtype=jnp.float64,
    )

    mass = jnp.array(
        [
            SOLAR_MASS,
            9.54791938424326609e-04 * SOLAR_MASS,
            2.85885980666130812e-04 * SOLAR_MASS,
            4.36624404335156298e-05 * SOLAR_MASS,
            5.15138902046611451e-05 * SOLAR_MASS,
        ],
        dtype=jnp.float64,
    )

    return pos, vel, mass


def offset_momentum_jax(vel, mass):
    vel = vel.at[0].set(-jnp.sum(vel * mass[:, None], axis=0) / SOLAR_MASS)
    return vel


def energy_jax(pos, vel, mass):
    ke = 0.5 * jnp.sum(mass * jnp.sum(vel * vel, axis=1))
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=2)
    idx_i, idx_j = jnp.triu_indices(5, k=1)
    dist = jnp.sqrt(dist_sq[idx_i, idx_j])
    pe = -jnp.sum(mass[idx_i] * mass[idx_j] / dist)
    return ke + pe


@jit
def nbody_step(pos, vel, mass, dt):
    """Single n-body timestep, JIT compiled."""
    diff = pos[:, None, :] - pos[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=2)
    safe_dist_sq = dist_sq + jnp.eye(5) * 1.0
    mag = dt * safe_dist_sq ** (-1.5)
    mag = mag * (1.0 - jnp.eye(5))

    weighted_mag = mag * mass[None, :]
    accel = -jnp.einsum("ijk,ij->ik", diff, weighted_mag)

    vel = vel + accel
    pos = pos + dt * vel
    return pos, vel


def run_nbody_jax(n: int = 500_000) -> dict:
    pos, vel, mass = make_nbody_system()
    vel = offset_momentum_jax(vel, mass)
    e_before = float(energy_jax(pos, vel, mass))

    dt = 0.01

    @jit
    def run_loop(pos, vel):
        def body(_, carry):
            p, v = carry
            p, v = nbody_step(p, v, mass, dt)
            return (p, v)

        return jax.lax.fori_loop(0, n, body, (pos, vel))

    # Warmup (includes JIT compilation)
    pos_w, vel_w = run_loop(pos, vel)
    pos_w.block_until_ready()

    # Timed runs
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pos_final, vel_final = run_loop(pos, vel)
        pos_final.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    elapsed = times[1]

    e_after = float(energy_jax(pos_final, vel_final, mass))
    return {
        "time_ms": elapsed,
        "energy_before": round(e_before, 9),
        "energy_after": round(e_after, 9),
    }


# ---------------------------------------------------------------------------
# Spectral-norm
# ---------------------------------------------------------------------------


@partial(jit, static_argnums=(0,))
def spectral_norm_jax(n: int = 2000):
    """Spectral-norm using JAX JIT."""
    i = jnp.arange(n, dtype=jnp.float64).reshape(-1, 1)
    j = jnp.arange(n, dtype=jnp.float64).reshape(1, -1)
    ij = i + j
    a = 1.0 / (ij * (ij + 1) / 2 + i + 1)
    at = a.T

    u = jnp.ones(n, dtype=jnp.float64)

    def body(_, u_v):
        u, v = u_v
        v = at @ (a @ u)
        u = at @ (a @ v)
        return (u, v)

    v = jnp.ones(n, dtype=jnp.float64)
    u, v = jax.lax.fori_loop(0, 10, body, (u, v))

    vBv = jnp.dot(u, v)
    vv = jnp.dot(v, v)
    return jnp.sqrt(vBv / vv)


def run_spectral_jax(n: int = 2000) -> dict:
    # Warmup (includes JIT compilation)
    result_w = spectral_norm_jax(n)
    result_w.block_until_ready()

    # Timed runs
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = spectral_norm_jax(n)
        result.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    elapsed = times[1]

    return {
        "time_ms": elapsed,
        "result": round(float(result), 9),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("JAX JIT Benchmarks (CPU)")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    # N-body
    print("N-body (500K iterations)...")
    r = run_nbody_jax(500_000)
    print(f"  Time: {r['time_ms']:.0f} ms")
    print(f"  Energy before: {r['energy_before']}")
    print(f"  Energy after:  {r['energy_after']}")
    print()

    # Spectral-norm
    print("Spectral-norm (N=2000)...")
    r = run_spectral_jax(2000)
    print(f"  Time: {r['time_ms']:.1f} ms")
    print(f"  Result: {r['result']}")
    print()

    # Compare with baseline
    print("For reference:")
    print("  N-body baseline (CPython 3.14): ~1220 ms")
    print("  Spectral-norm baseline (CPython 3.14): ~14046 ms")


if __name__ == "__main__":
    main()
