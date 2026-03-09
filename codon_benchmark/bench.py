"""Benchmark runner for Codon — n-body and spectral-norm.

Codon is a standalone Python compiler (not a CPython extension).
These benchmarks must be compiled and run as a standalone binary.

Build: ~/.codon/bin/codon build -release -o bench_codon codon_benchmark/bench.py
Run:   DYLD_LIBRARY_PATH=~/.codon/lib/codon ./bench_codon

Codon constraints: no f-strings, no json, no sys.implementation.
"""

import time

from nbody import DEFAULT_N as NBODY_N
from nbody import run_nbody
from spectral_norm import DEFAULT_N as SPECTRAL_N
from spectral_norm import run_spectral

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
    print("  nbody energy_after:  " + str(e_after))
    print("  spectral_norm:       " + str(sn))


main()
