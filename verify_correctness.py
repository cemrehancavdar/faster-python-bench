"""Correctness verification for all benchmark implementations.

Checks that every implementation produces results matching the official
expected outputs from the Computer Language Benchmarks Game.

Known expected values (from hanabi1224/Programming-Language-Benchmarks):
  N-body 1000:   energy_before = -0.169075164, energy_after = -0.169087605
  N-body 10000:  energy_before = -0.169075164, energy_after = -0.169016441
  Spectral-norm N=100: 1.274219991
  Spectral-norm N=2:   1.183350177

These are the canonical test vectors that all Benchmarks Game submissions
must produce. If our implementations match these, they are correct regardless
of who wrote them.

Usage:
    uv run verify_correctness.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# Tolerance: 9 decimal places in the expected output
ENERGY_TOL = 1e-7
SPECTRAL_TOL = 1e-7

# Official expected values from the Benchmarks Game
NBODY_EXPECTED: dict[int, tuple[float, float]] = {
    # n: (energy_before, energy_after)
    1000: (-0.169075164, -0.169087605),
    10000: (-0.169075164, -0.169016441),
}

SPECTRAL_EXPECTED: dict[int, float] = {
    # n: spectral_norm
    2: 1.183350177,
    100: 1.274219991,
}


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str


def verify_nbody_baseline() -> list[TestResult]:
    """Verify n-body CPython baseline against known outputs."""
    results: list[TestResult] = []
    from baseline.nbody import run_benchmark

    for n, (exp_before, exp_after) in NBODY_EXPECTED.items():
        result = run_benchmark(n)
        e_before = result["energy_before"]
        e_after = result["energy_after"]

        before_ok = abs(e_before - exp_before) < ENERGY_TOL
        after_ok = abs(e_after - exp_after) < ENERGY_TOL

        if before_ok and after_ok:
            results.append(
                TestResult(
                    f"nbody/baseline n={n}",
                    True,
                    f"before={e_before:.9f} after={e_after:.9f}",
                )
            )
        else:
            results.append(
                TestResult(
                    f"nbody/baseline n={n}",
                    False,
                    f"MISMATCH before={e_before:.9f} (exp {exp_before:.9f}) "
                    f"after={e_after:.9f} (exp {exp_after:.9f})",
                )
            )
    return results


def verify_nbody_numba() -> list[TestResult]:
    """Verify n-body Numba implementation against known outputs."""
    results: list[TestResult] = []
    try:
        from numba_benchmark.nbody import run_benchmark, warmup

        warmup()
    except ImportError as e:
        return [TestResult("nbody/numba", False, f"Import failed: {e}")]

    for n, (exp_before, exp_after) in NBODY_EXPECTED.items():
        result = run_benchmark(n)
        e_before = result["energy_before"]
        e_after = result["energy_after"]

        before_ok = abs(e_before - exp_before) < ENERGY_TOL
        after_ok = abs(e_after - exp_after) < ENERGY_TOL

        if before_ok and after_ok:
            results.append(
                TestResult(
                    f"nbody/numba n={n}",
                    True,
                    f"before={e_before:.9f} after={e_after:.9f}",
                )
            )
        else:
            results.append(
                TestResult(
                    f"nbody/numba n={n}",
                    False,
                    f"MISMATCH before={e_before:.9f} (exp {exp_before:.9f}) "
                    f"after={e_after:.9f} (exp {exp_after:.9f})",
                )
            )
    return results


def verify_nbody_cython() -> list[TestResult]:
    """Verify n-body Cython implementation against known outputs."""
    results: list[TestResult] = []
    try:
        from cython_benchmark.nbody_compiled import run_benchmark  # type: ignore[import-not-found]
    except ImportError:
        try:
            from cython_benchmark.nbody import run_benchmark
        except ImportError as e:
            return [TestResult("nbody/cython", False, f"Import failed: {e}")]

    for n, (exp_before, exp_after) in NBODY_EXPECTED.items():
        result = run_benchmark(n)
        e_before = result["energy_before"]
        e_after = result["energy_after"]

        before_ok = abs(e_before - exp_before) < ENERGY_TOL
        after_ok = abs(e_after - exp_after) < ENERGY_TOL

        if before_ok and after_ok:
            results.append(
                TestResult(
                    f"nbody/cython n={n}",
                    True,
                    f"before={e_before:.9f} after={e_after:.9f}",
                )
            )
        else:
            results.append(
                TestResult(
                    f"nbody/cython n={n}",
                    False,
                    f"MISMATCH before={e_before:.9f} (exp {exp_before:.9f}) "
                    f"after={e_after:.9f} (exp {exp_after:.9f})",
                )
            )
    return results


def verify_nbody_rust() -> list[TestResult]:
    """Verify n-body Rust/PyO3 implementation against known outputs."""
    results: list[TestResult] = []
    try:
        import pipeline_rust  # type: ignore[import-not-found]
    except ImportError:
        return [TestResult("nbody/rust", False, "pipeline_rust not built")]

    for n, (exp_before, exp_after) in NBODY_EXPECTED.items():
        e_before, e_after = pipeline_rust.nbody_benchmark(n)
        # Rust returns raw floats, round to 9 decimal places for comparison
        e_before = round(e_before, 9)
        e_after = round(e_after, 9)

        before_ok = abs(e_before - exp_before) < ENERGY_TOL
        after_ok = abs(e_after - exp_after) < ENERGY_TOL

        if before_ok and after_ok:
            results.append(
                TestResult(
                    f"nbody/rust n={n}",
                    True,
                    f"before={e_before:.9f} after={e_after:.9f}",
                )
            )
        else:
            results.append(
                TestResult(
                    f"nbody/rust n={n}",
                    False,
                    f"MISMATCH before={e_before:.9f} (exp {exp_before:.9f}) "
                    f"after={e_after:.9f} (exp {exp_after:.9f})",
                )
            )
    return results


def verify_spectral_baseline() -> list[TestResult]:
    """Verify spectral-norm CPython baseline against known outputs."""
    results: list[TestResult] = []
    from baseline.spectral_norm import run_benchmark

    for n, exp_value in SPECTRAL_EXPECTED.items():
        result = run_benchmark(n)
        actual = result["spectral_norm"]

        if abs(actual - exp_value) < SPECTRAL_TOL:
            results.append(
                TestResult(
                    f"spectral/baseline N={n}",
                    True,
                    f"result={actual:.9f}",
                )
            )
        else:
            results.append(
                TestResult(
                    f"spectral/baseline N={n}",
                    False,
                    f"MISMATCH result={actual:.9f} (exp {exp_value:.9f})",
                )
            )
    return results


def verify_spectral_numpy() -> list[TestResult]:
    """Verify spectral-norm NumPy implementation against known outputs."""
    results: list[TestResult] = []
    try:
        from numpy_benchmark.spectral_norm import run_benchmark
    except ImportError as e:
        return [TestResult("spectral/numpy", False, f"Import failed: {e}")]

    for n, exp_value in SPECTRAL_EXPECTED.items():
        result = run_benchmark(n)
        actual = result["spectral_norm"]

        if abs(actual - exp_value) < SPECTRAL_TOL:
            results.append(
                TestResult(
                    f"spectral/numpy N={n}",
                    True,
                    f"result={actual:.9f}",
                )
            )
        else:
            results.append(
                TestResult(
                    f"spectral/numpy N={n}",
                    False,
                    f"MISMATCH result={actual:.9f} (exp {exp_value:.9f})",
                )
            )
    return results


def verify_spectral_numba() -> list[TestResult]:
    """Verify spectral-norm Numba implementation against known outputs."""
    results: list[TestResult] = []
    try:
        from numba_benchmark.spectral_norm import run_benchmark, warmup

        warmup()
    except ImportError as e:
        return [TestResult("spectral/numba", False, f"Import failed: {e}")]

    for n, exp_value in SPECTRAL_EXPECTED.items():
        result = run_benchmark(n)
        actual = result["spectral_norm"]

        if abs(actual - exp_value) < SPECTRAL_TOL:
            results.append(
                TestResult(
                    f"spectral/numba N={n}",
                    True,
                    f"result={actual:.9f}",
                )
            )
        else:
            results.append(
                TestResult(
                    f"spectral/numba N={n}",
                    False,
                    f"MISMATCH result={actual:.9f} (exp {exp_value:.9f})",
                )
            )
    return results


def verify_spectral_cython() -> list[TestResult]:
    """Verify spectral-norm Cython implementation against known outputs."""
    results: list[TestResult] = []
    try:
        from cython_benchmark.spectral_norm_compiled import (
            run_benchmark,  # type: ignore[import-not-found]
        )
    except ImportError:
        try:
            from cython_benchmark.spectral_norm import run_benchmark
        except ImportError as e:
            return [TestResult("spectral/cython", False, f"Import failed: {e}")]

    for n, exp_value in SPECTRAL_EXPECTED.items():
        result = run_benchmark(n)
        actual = result["spectral_norm"]

        if abs(actual - exp_value) < SPECTRAL_TOL:
            results.append(
                TestResult(
                    f"spectral/cython N={n}",
                    True,
                    f"result={actual:.9f}",
                )
            )
        else:
            results.append(
                TestResult(
                    f"spectral/cython N={n}",
                    False,
                    f"MISMATCH result={actual:.9f} (exp {exp_value:.9f})",
                )
            )
    return results


def verify_spectral_rust() -> list[TestResult]:
    """Verify spectral-norm Rust/PyO3 implementation against known outputs."""
    results: list[TestResult] = []
    try:
        import pipeline_rust  # type: ignore[import-not-found]
    except ImportError:
        return [TestResult("spectral/rust", False, "pipeline_rust not built")]

    for n, exp_value in SPECTRAL_EXPECTED.items():
        actual = round(pipeline_rust.spectral_norm_benchmark(n), 9)

        if abs(actual - exp_value) < SPECTRAL_TOL:
            results.append(
                TestResult(
                    f"spectral/rust N={n}",
                    True,
                    f"result={actual:.9f}",
                )
            )
        else:
            results.append(
                TestResult(
                    f"spectral/rust N={n}",
                    False,
                    f"MISMATCH result={actual:.9f} (exp {exp_value:.9f})",
                )
            )
    return results


def main() -> None:
    print("=" * 70)
    print("  CORRECTNESS VERIFICATION")
    print("  Checking all implementations against Benchmarks Game test vectors")
    print("=" * 70)
    print()

    all_verifiers = [
        ("N-body baseline", verify_nbody_baseline),
        ("N-body Numba", verify_nbody_numba),
        ("N-body Cython", verify_nbody_cython),
        ("N-body Rust", verify_nbody_rust),
        ("Spectral-norm baseline", verify_spectral_baseline),
        ("Spectral-norm NumPy", verify_spectral_numpy),
        ("Spectral-norm Numba", verify_spectral_numba),
        ("Spectral-norm Cython", verify_spectral_cython),
        ("Spectral-norm Rust", verify_spectral_rust),
    ]

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for section_name, verifier in all_verifiers:
        print(f"--- {section_name} ---")
        try:
            test_results = verifier()
        except Exception as e:
            print(f"  ERROR: {e}")
            total_fail += 1
            continue

        for tr in test_results:
            if tr.passed:
                print(f"  PASS  {tr.name}: {tr.message}")
                total_pass += 1
            elif "Import failed" in tr.message or "not built" in tr.message:
                print(f"  SKIP  {tr.name}: {tr.message}")
                total_skip += 1
            else:
                print(f"  FAIL  {tr.name}: {tr.message}")
                total_fail += 1
        print()

    print("=" * 70)
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed, {total_skip} skipped")
    print("=" * 70)

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
