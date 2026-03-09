"""Benchmark runner for Mojo — n-body and spectral-norm.

Mojo is a compiled language with Python-like syntax. These implementations
are direct translations of the official Benchmarks Game algorithms.

Build: pixi run mojo build -O3 -o /tmp/bench_mojo bench.mojo
Run:   /tmp/bench_mojo
  or:  pixi run mojo run bench.mojo
"""

from time import perf_counter_ns

from nbody import run_nbody, DEFAULT_N as NBODY_N
from spectral_norm import run_spectral, DEFAULT_N as SPECTRAL_N


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

comptime RUNS = 5


fn median(mut values: List[Float64]) -> Float64:
    """Simple selection sort to find median (only 5 elements)."""
    var n = len(values)
    for i in range(n):
        var min_idx = i
        for j in range(i + 1, n):
            if values[j] < values[min_idx]:
                min_idx = j
        if min_idx != i:
            var tmp = values[i]
            values[i] = values[min_idx]
            values[min_idx] = tmp
    return values[n // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

fn main():
    print("Mojo nightly (CPU, compiled with -O3)")
    print("Runs:", RUNS)
    print()

    # Warmup
    _ = run_nbody()
    _ = run_spectral()

    # N-body
    print("  n-body (500000 iterations)...", end=" ")
    var times_nb = List[Float64]()
    for _ in range(RUNS):
        var t0 = perf_counter_ns()
        _ = run_nbody()
        var t1 = perf_counter_ns()
        times_nb.append(Float64(t1 - t0) / 1_000_000.0)
    var med_nb = median(times_nb)
    print(med_nb, "ms")

    # Spectral-norm
    print("  spectral-norm (N=2000)...", end=" ")
    var times_sn = List[Float64]()
    for _ in range(RUNS):
        var t0 = perf_counter_ns()
        _ = run_spectral()
        var t1 = perf_counter_ns()
        times_sn.append(Float64(t1 - t0) / 1_000_000.0)
    var med_sn = median(times_sn)
    print(med_sn, "ms")

    # Correctness
    var result = run_nbody()
    var sn = run_spectral()
    print()
    print("Correctness:")
    print("  nbody energy_before:", result[0])
    print("  nbody energy_after: ", result[1])
    print("  spectral_norm:      ", sn)
