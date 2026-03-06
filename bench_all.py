"""Unified benchmark runner for all three benchmark suites.

Supports:
  - nbody:         N-body simulation (Benchmarks Game) — tight FP loops
  - spectral_norm: Spectral-norm (Benchmarks Game) — matrix-vector arithmetic
  - pipeline:      JSON transform pipeline — dict-heavy real-world code

Each suite has multiple implementations (baseline, numba, cython, rust, etc.).
The runner measures wall-clock time (median of N runs) and verifies correctness.

Usage:
    uv run bench_all.py                          # run all suites
    uv run bench_all.py --suite nbody            # run only n-body
    uv run bench_all.py --suite spectral_norm    # run only spectral-norm
    uv run bench_all.py --suite pipeline         # run only pipeline
    uv run bench_all.py --runs 10                # 10 runs per benchmark
    uv run bench_all.py --list                   # list all implementations
    uv run bench_all.py --json-out results.json  # save results to JSON
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "events.json"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class BenchResult:
    suite: str
    name: str
    times: list[float] = field(default_factory=list)
    median_ms: float = 0.0
    speedup: float = 0.0
    correct: bool = False
    error: str = ""
    skipped: bool = False
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def time_runs(fn: Any, runs: int) -> tuple[list[float], Any]:
    """Run fn() multiple times, return (list of times in ms, last result)."""
    times: list[float] = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times, result


# ---------------------------------------------------------------------------
# N-body benchmarks
# ---------------------------------------------------------------------------

NBODY_DEFAULT_N = 500_000  # 500K iterations for local benchmarking


def nbody_baseline(runs: int) -> BenchResult:
    """N-body: CPython baseline."""
    from baseline.nbody import run_benchmark

    res = BenchResult(suite="nbody", name="CPython baseline")
    times, result = time_runs(lambda: run_benchmark(NBODY_DEFAULT_N), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None and abs(result["energy_before"] - (-0.169075164)) < 1e-6
    return res


def nbody_numba(runs: int) -> BenchResult:
    """N-body: Numba JIT."""
    res = BenchResult(suite="nbody", name="Numba")
    try:
        from numba_benchmark.nbody import run_benchmark, warmup

        warmup()
    except ImportError as e:
        res.skipped = True
        res.skip_reason = f"Import failed: {e}. Install: uv sync --extra numba"
        return res

    times, result = time_runs(lambda: run_benchmark(NBODY_DEFAULT_N), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None and abs(result["energy_before"] - (-0.169075164)) < 1e-6
    return res


def nbody_cython(runs: int) -> BenchResult:
    """N-body: Cython compiled."""
    res = BenchResult(suite="nbody", name="Cython")
    try:
        from cython_benchmark.nbody_compiled import run_benchmark  # type: ignore[import-not-found]
    except ImportError:
        try:
            from cython_benchmark.nbody import run_benchmark

            res.name = "Cython (uncompiled)"
        except ImportError as e:
            res.skipped = True
            res.skip_reason = f"Import failed: {e}"
            return res

    times, result = time_runs(lambda: run_benchmark(NBODY_DEFAULT_N), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None and abs(result["energy_before"] - (-0.169075164)) < 1e-6
    return res


def nbody_pypy(runs: int) -> BenchResult:
    """N-body: PyPy (subprocess — needs pypy3 interpreter)."""
    res = BenchResult(suite="nbody", name="PyPy")

    pypy_check = subprocess.run(["which", "pypy3"], capture_output=True, text=True)
    if pypy_check.returncode != 0:
        res.skipped = True
        res.skip_reason = "pypy3 not found in PATH"
        return res

    script = (
        "import json, time, statistics, sys\n"
        "sys.path.insert(0, '.')\n"
        "from baseline.nbody import run_benchmark\n"
        "# warmup\n"
        "run_benchmark(100)\n"
        f"times = []\n"
        f"for _ in range({runs}):\n"
        "    t0 = time.perf_counter()\n"
        f"    result = run_benchmark({NBODY_DEFAULT_N})\n"
        "    t1 = time.perf_counter()\n"
        "    times.append((t1 - t0) * 1000)\n"
        "print(json.dumps({'times': times, 'median_ms': statistics.median(times),\n"
        "    'energy_before': result['energy_before']}))\n"
    )

    try:
        proc = subprocess.run(
            ["pypy3", "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            res.error = proc.stderr[:500]
            return res
        data = json.loads(proc.stdout.strip())
        res.times = data["times"]
        res.median_ms = data["median_ms"]
        res.correct = abs(data["energy_before"] - (-0.169075164)) < 1e-6
    except subprocess.TimeoutExpired:
        res.error = "Timed out (300s)"
    except Exception as e:
        res.error = str(e)

    return res


def nbody_rust(runs: int) -> BenchResult:
    """N-body: Rust/PyO3."""
    res = BenchResult(suite="nbody", name="Rust (PyO3)")
    try:
        import pipeline_rust  # type: ignore[import-not-found]
    except ImportError:
        res.skipped = True
        res.skip_reason = "Not built. Run: cd rust_benchmark && maturin develop --release"
        return res

    times, result = time_runs(lambda: pipeline_rust.nbody_benchmark(NBODY_DEFAULT_N), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    if result is not None:
        e_before, _e_after = result
        res.correct = abs(e_before - (-0.169075164)) < 1e-6
    return res


NBODY_BENCHMARKS: dict[str, Any] = {
    "baseline": nbody_baseline,
    "numba": nbody_numba,
    "cython": nbody_cython,
    "pypy": nbody_pypy,
    "rust": nbody_rust,
}


# ---------------------------------------------------------------------------
# Spectral-norm benchmarks
# ---------------------------------------------------------------------------

SPECTRAL_DEFAULT_N = 2000  # N=2000 for local benchmarking


def spectral_baseline(runs: int) -> BenchResult:
    """Spectral-norm: CPython baseline."""
    from baseline.spectral_norm import run_benchmark

    res = BenchResult(suite="spectral_norm", name="CPython baseline")
    times, result = time_runs(lambda: run_benchmark(SPECTRAL_DEFAULT_N), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None and abs(result["spectral_norm"] - 1.274224153) < 1e-6
    return res


def spectral_numpy(runs: int) -> BenchResult:
    """Spectral-norm: NumPy vectorized."""
    res = BenchResult(suite="spectral_norm", name="NumPy")
    try:
        from numpy_benchmark.spectral_norm import run_benchmark
    except ImportError as e:
        res.skipped = True
        res.skip_reason = f"Import failed: {e}. Install numpy."
        return res

    times, result = time_runs(lambda: run_benchmark(SPECTRAL_DEFAULT_N), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None and abs(result["spectral_norm"] - 1.274224153) < 1e-6
    return res


def spectral_numba(runs: int) -> BenchResult:
    """Spectral-norm: Numba JIT."""
    res = BenchResult(suite="spectral_norm", name="Numba")
    try:
        from numba_benchmark.spectral_norm import run_benchmark, warmup

        warmup()
    except ImportError as e:
        res.skipped = True
        res.skip_reason = f"Import failed: {e}. Install: uv sync --extra numba"
        return res

    times, result = time_runs(lambda: run_benchmark(SPECTRAL_DEFAULT_N), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None and abs(result["spectral_norm"] - 1.274224153) < 1e-6
    return res


def spectral_cython(runs: int) -> BenchResult:
    """Spectral-norm: Cython compiled."""
    res = BenchResult(suite="spectral_norm", name="Cython")
    try:
        from cython_benchmark.spectral_norm_compiled import (
            run_benchmark,  # type: ignore[import-not-found]
        )
    except ImportError:
        try:
            from cython_benchmark.spectral_norm import run_benchmark

            res.name = "Cython (uncompiled)"
        except ImportError as e:
            res.skipped = True
            res.skip_reason = f"Import failed: {e}"
            return res

    times, result = time_runs(lambda: run_benchmark(SPECTRAL_DEFAULT_N), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None and abs(result["spectral_norm"] - 1.274224153) < 1e-6
    return res


def spectral_pypy(runs: int) -> BenchResult:
    """Spectral-norm: PyPy (subprocess)."""
    res = BenchResult(suite="spectral_norm", name="PyPy")

    pypy_check = subprocess.run(["which", "pypy3"], capture_output=True, text=True)
    if pypy_check.returncode != 0:
        res.skipped = True
        res.skip_reason = "pypy3 not found in PATH"
        return res

    script = (
        "import json, time, statistics, sys\n"
        "sys.path.insert(0, '.')\n"
        "from baseline.spectral_norm import run_benchmark\n"
        "run_benchmark(10)\n"
        f"times = []\n"
        f"for _ in range({runs}):\n"
        "    t0 = time.perf_counter()\n"
        f"    result = run_benchmark({SPECTRAL_DEFAULT_N})\n"
        "    t1 = time.perf_counter()\n"
        "    times.append((t1 - t0) * 1000)\n"
        "print(json.dumps({'times': times, 'median_ms': statistics.median(times),\n"
        "    'spectral_norm': result['spectral_norm']}))\n"
    )

    try:
        proc = subprocess.run(
            ["pypy3", "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            res.error = proc.stderr[:500]
            return res
        data = json.loads(proc.stdout.strip())
        res.times = data["times"]
        res.median_ms = data["median_ms"]
        res.correct = abs(data["spectral_norm"] - 1.274224153) < 1e-6
    except subprocess.TimeoutExpired:
        res.error = "Timed out (300s)"
    except Exception as e:
        res.error = str(e)

    return res


def spectral_rust(runs: int) -> BenchResult:
    """Spectral-norm: Rust/PyO3."""
    res = BenchResult(suite="spectral_norm", name="Rust (PyO3)")
    try:
        import pipeline_rust  # type: ignore[import-not-found]
    except ImportError:
        res.skipped = True
        res.skip_reason = "Not built. Run: cd rust_benchmark && maturin develop --release"
        return res

    times, result = time_runs(
        lambda: pipeline_rust.spectral_norm_benchmark(SPECTRAL_DEFAULT_N), runs
    )
    res.times = times
    res.median_ms = statistics.median(times)
    if result is not None:
        res.correct = abs(result - 1.274224153) < 1e-6
    return res


SPECTRAL_BENCHMARKS: dict[str, Any] = {
    "baseline": spectral_baseline,
    "numpy": spectral_numpy,
    "numba": spectral_numba,
    "cython": spectral_cython,
    "pypy": spectral_pypy,
    "rust": spectral_rust,
}


# ---------------------------------------------------------------------------
# Pipeline benchmarks (existing — delegates to bench.py implementations)
# ---------------------------------------------------------------------------


def _load_events() -> list[dict[str, Any]]:
    """Load pipeline events once."""
    if not DATA_PATH.exists():
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "data" / "generate.py")],
            check=True,
        )
    with open(DATA_PATH) as f:
        return json.load(f)


_EVENTS_CACHE: list[dict[str, Any]] | None = None


def get_events() -> list[dict[str, Any]]:
    global _EVENTS_CACHE
    if _EVENTS_CACHE is None:
        _EVENTS_CACHE = _load_events()
    return _EVENTS_CACHE


def pipeline_baseline(runs: int) -> BenchResult:
    """Pipeline: CPython baseline."""
    from baseline.pipeline import run_pipeline

    events = get_events()
    res = BenchResult(suite="pipeline", name="CPython baseline")
    times, result = time_runs(lambda: run_pipeline(events), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


def pipeline_cython(runs: int) -> BenchResult:
    """Pipeline: Cython compiled."""
    res = BenchResult(suite="pipeline", name="Cython")
    try:
        from cython_benchmark.pipeline_compiled import (
            run_pipeline,  # type: ignore[import-not-found]
        )
    except ImportError:
        try:
            from cython_benchmark.pipeline import run_pipeline

            res.name = "Cython (uncompiled)"
        except ImportError as e:
            res.skipped = True
            res.skip_reason = f"Import failed: {e}"
            return res

    events = get_events()
    times, result = time_runs(lambda: run_pipeline(events), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


def pipeline_mypyc(runs: int) -> BenchResult:
    """Pipeline: Mypyc compiled."""
    res = BenchResult(suite="pipeline", name="Mypyc")
    try:
        from mypyc_benchmark.pipeline import run_pipeline
    except ImportError as e:
        res.skipped = True
        res.skip_reason = f"Import failed: {e}"
        return res

    events = get_events()
    times, result = time_runs(lambda: run_pipeline(events), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


def pipeline_rust_json(runs: int) -> BenchResult:
    """Pipeline: Rust/PyO3 from JSON bytes."""
    res = BenchResult(suite="pipeline", name="Rust (from JSON)")
    try:
        import pipeline_rust  # type: ignore[import-not-found]
    except ImportError:
        res.skipped = True
        res.skip_reason = "Not built. Run: cd rust_benchmark && maturin develop --release"
        return res

    json_bytes = DATA_PATH.read_bytes()
    times, result = time_runs(lambda: pipeline_rust.run_pipeline_from_json(json_bytes), runs)
    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


PIPELINE_BENCHMARKS: dict[str, Any] = {
    "baseline": pipeline_baseline,
    "cython": pipeline_cython,
    "mypyc": pipeline_mypyc,
    "rust_json": pipeline_rust_json,
}


# ---------------------------------------------------------------------------
# All suites
# ---------------------------------------------------------------------------

ALL_SUITES: dict[str, dict[str, Any]] = {
    "nbody": NBODY_BENCHMARKS,
    "spectral_norm": SPECTRAL_BENCHMARKS,
    "pipeline": PIPELINE_BENCHMARKS,
}


def format_suite_results(suite_name: str, results: list[BenchResult], baseline_ms: float) -> str:
    """Format results for one suite as a readable table."""
    lines = [f"\n{'=' * 70}", f"  {suite_name.upper()}", f"{'=' * 70}"]
    lines.append(f"  {'Implementation':<25} {'Median':>10} {'Speedup':>10} {'Status':>8}")
    lines.append(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 8}")

    for r in results:
        if r.skipped:
            lines.append(f"  {r.name:<25} {'—':>10} {'—':>10} {'SKIP':>8}")
            continue
        if r.error:
            lines.append(f"  {r.name:<25} {'—':>10} {'—':>10} {'ERROR':>8}")
            continue
        speedup = baseline_ms / r.median_ms if r.median_ms > 0 else 0
        status = "OK" if r.correct else "FAIL"
        lines.append(f"  {r.name:<25} {r.median_ms:>8.1f}ms {speedup:>9.1f}x {status:>8}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified benchmark runner")
    parser.add_argument("--suite", type=str, choices=list(ALL_SUITES) + ["all"], default="all")
    parser.add_argument("--runs", type=int, default=5, help="Runs per benchmark (default: 5)")
    parser.add_argument("--list", action="store_true", help="List all implementations")
    parser.add_argument("--json-out", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if args.list:
        for suite_name, benchmarks in ALL_SUITES.items():
            print(f"\n{suite_name}:")
            for name in benchmarks:
                print(f"  {name}")
        sys.exit(0)

    suites_to_run = ALL_SUITES if args.suite == "all" else {args.suite: ALL_SUITES[args.suite]}
    all_results: list[BenchResult] = []

    for suite_name, benchmarks in suites_to_run.items():
        print(f"\n--- {suite_name} ({args.runs} runs each) ---")
        suite_results: list[BenchResult] = []
        baseline_ms = 0.0

        for impl_name, bench_fn in benchmarks.items():
            print(f"  {impl_name}...", end=" ", flush=True)
            result = bench_fn(args.runs)
            suite_results.append(result)

            if result.skipped:
                print(f"SKIPPED ({result.skip_reason})")
            elif result.error:
                print(f"ERROR ({result.error[:60]})")
            else:
                print(f"{result.median_ms:.1f}ms")

            if impl_name == "baseline" and not result.skipped and not result.error:
                baseline_ms = result.median_ms

        if baseline_ms == 0 and suite_results:
            baseline_ms = next((r.median_ms for r in suite_results if r.median_ms > 0), 1.0)

        print(format_suite_results(suite_name, suite_results, baseline_ms))
        all_results.extend(suite_results)

        # Print skip/error details
        for r in suite_results:
            if r.skipped:
                print(f"    {r.name}: {r.skip_reason}")
            if r.error:
                print(f"    {r.name} error: {r.error}")

    if args.json_out:
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / args.json_out
        json_results = [
            {
                "suite": r.suite,
                "name": r.name,
                "times": r.times,
                "median_ms": r.median_ms,
                "correct": r.correct,
                "skipped": r.skipped,
                "error": r.error,
            }
            for r in all_results
        ]
        out_path.write_text(json.dumps(json_results, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
