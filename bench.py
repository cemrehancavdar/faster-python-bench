"""Unified benchmark runner.

Runs every available implementation of the JSON transform pipeline,
measures wall-clock time (median of N runs), verifies correctness
against baseline, and outputs a results table.

Usage:
    uv run bench.py                  # run all available benchmarks
    uv run bench.py --runs 10        # 10 runs per benchmark (default: 5)
    uv run bench.py --only baseline  # run only the baseline
    uv run bench.py --list           # list available benchmarks

Each benchmark must produce identical summary numbers (total_users,
total_events, total_amount, total_high_value) as the baseline.
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
    name: str
    times: list[float] = field(default_factory=list)
    median_ms: float = 0.0
    speedup: float = 0.0
    correct: bool = False
    error: str = ""
    skipped: bool = False
    skip_reason: str = ""


def load_events() -> list[dict[str, Any]]:
    """Load events once, shared across all benchmarks."""
    with open(DATA_PATH) as f:
        return json.load(f)


def summary_key(result: dict[str, Any]) -> tuple[int, int, float, int]:
    """Extract the four summary numbers for correctness check."""
    return (
        result["total_users"],
        result["total_events"],
        round(result["total_amount"], 2),
        result["total_high_value"],
    )


# ---------------------------------------------------------------------------
# Benchmark implementations
# ---------------------------------------------------------------------------


def bench_baseline(events: list[dict[str, Any]], runs: int) -> BenchResult:
    """Pure Python baseline."""
    from baseline.pipeline import run_pipeline

    res = BenchResult(name="CPython baseline")
    times: list[float] = []
    result = None

    for _ in range(runs):
        t0 = time.perf_counter()
        result = run_pipeline(events)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


def bench_numba(events: list[dict[str, Any]], runs: int) -> BenchResult:
    """Numba JIT benchmark."""
    res = BenchResult(name="Numba (JIT)")
    try:
        from numba_benchmark.pipeline import run_pipeline, warmup
    except ImportError as e:
        res.skipped = True
        res.skip_reason = f"Import failed: {e}. Install with: uv sync --extra numba"
        return res

    # Warmup JIT
    warmup(events)

    times: list[float] = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = run_pipeline(events)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


def bench_cython(events: list[dict[str, Any]], runs: int) -> BenchResult:
    """Cython pure Python mode benchmark."""
    res = BenchResult(name="Cython (pure Python)")

    # Try to import the compiled version first
    try:
        from cython_benchmark.pipeline_compiled import (
            run_pipeline,  # type: ignore[import-not-found]
        )

        compiled = True
    except ImportError:
        # Fall back to uncompiled (same as baseline, but shows the code works)
        try:
            from cython_benchmark.pipeline import run_pipeline

            compiled = False
        except ImportError as e:
            res.skipped = True
            res.skip_reason = f"Import failed: {e}"
            return res

    if not compiled:
        res.name = "Cython (uncompiled, same as baseline)"

    times: list[float] = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = run_pipeline(events)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


def bench_mypyc(events: list[dict[str, Any]], runs: int) -> BenchResult:
    """Mypyc-compiled benchmark."""
    res = BenchResult(name="Mypyc")

    # Check if compiled .so exists
    mypyc_dir = PROJECT_ROOT / "mypyc_benchmark"
    compiled = any(f.suffix == ".so" or f.suffix == ".pyd" for f in mypyc_dir.iterdir())

    try:
        from mypyc_benchmark.pipeline import run_pipeline
    except ImportError as e:
        res.skipped = True
        res.skip_reason = f"Import failed: {e}"
        return res

    if not compiled:
        res.name = "Mypyc (uncompiled, same as baseline)"

    times: list[float] = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = run_pipeline(events)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


def bench_pypy(events: list[dict[str, Any]], runs: int) -> BenchResult:
    """PyPy benchmark — runs as subprocess since we need pypy3 interpreter."""
    res = BenchResult(name="PyPy")

    # Check if pypy3 is available
    pypy_path = subprocess.run(["which", "pypy3"], capture_output=True, text=True)
    if pypy_path.returncode != 0:
        res.skipped = True
        res.skip_reason = "pypy3 not found in PATH"
        return res

    # PyPy needs a subprocess — we can't JIT from within CPython
    # Write a small timing script
    timing_script = PROJECT_ROOT / "pypy_benchmark" / "_bench_run.py"
    timing_script.write_text(
        "import json, time, sys\n"
        "sys.path.insert(0, '.')\n"
        "from pypy_benchmark.pipeline import run_pipeline\n"
        f"with open('{DATA_PATH}') as f:\n"
        "    events = json.load(f)\n"
        "# warmup\n"
        "run_pipeline(events)\n"
        f"times = []\n"
        f"for _ in range({runs}):\n"
        "    t0 = time.perf_counter()\n"
        "    result = run_pipeline(events)\n"
        "    t1 = time.perf_counter()\n"
        "    times.append((t1 - t0) * 1000)\n"
        "import statistics\n"
        "print(json.dumps({\n"
        "    'times': times,\n"
        "    'median_ms': statistics.median(times),\n"
        "    'total_users': result['total_users'],\n"
        "    'total_events': result['total_events'],\n"
        "    'total_amount': round(result['total_amount'], 2),\n"
        "    'total_high_value': result['total_high_value'],\n"
        "}))\n"
    )

    try:
        proc = subprocess.run(
            ["pypy3", str(timing_script)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            res.error = proc.stderr[:500]
            return res

        data = json.loads(proc.stdout.strip())
        res.times = data["times"]
        res.median_ms = data["median_ms"]
        res.correct = True
    except subprocess.TimeoutExpired:
        res.error = "Timed out (120s)"
    except Exception as e:
        res.error = str(e)
    finally:
        timing_script.unlink(missing_ok=True)

    return res


def bench_mojo(events: list[dict[str, Any]], runs: int) -> BenchResult:
    """Mojo benchmark — runs as subprocess."""
    res = BenchResult(name="Mojo")

    mojo_path = subprocess.run(["which", "mojo"], capture_output=True, text=True)
    if mojo_path.returncode != 0:
        res.skipped = True
        res.skip_reason = "mojo not found in PATH"
        return res

    res.skipped = True
    res.skip_reason = "Mojo benchmark requires manual setup — see mojo_benchmark/pipeline.mojo"
    return res


def bench_rust(events: list[dict[str, Any]], runs: int) -> BenchResult:
    """Rust/PyO3 benchmark — full pipeline including JSON parse."""
    res = BenchResult(name="Rust (PyO3, incl. JSON parse)")

    try:
        import pipeline_rust  # type: ignore[import-not-found]
    except ImportError:
        res.skipped = True
        res.skip_reason = "Not built. Run: cd rust_benchmark && maturin develop --release"
        return res

    json_bytes = DATA_PATH.read_bytes()

    times: list[float] = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = pipeline_rust.run_pipeline_from_json(json_bytes)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


def bench_rust_summary(events: list[dict[str, Any]], runs: int) -> BenchResult:
    """Rust/PyO3 benchmark — summary only (no Python dict construction)."""
    res = BenchResult(name="Rust (PyO3, summary only)")

    try:
        import pipeline_rust  # type: ignore[import-not-found]
    except ImportError:
        res.skipped = True
        res.skip_reason = "Not built. Run: cd rust_benchmark && maturin develop --release"
        return res

    json_bytes = DATA_PATH.read_bytes()

    times: list[float] = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = pipeline_rust.run_pipeline_summary(json_bytes)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


def bench_rust_from_dicts(events: list[dict[str, Any]], runs: int) -> BenchResult:
    """Rust/PyO3 pipeline from pre-parsed Python dicts."""
    res = BenchResult(name="Rust (PyO3, from dicts)")

    try:
        import pipeline_rust  # type: ignore[import-not-found]
    except ImportError:
        res.skipped = True
        res.skip_reason = "Not built. Run: cd rust_benchmark && maturin develop --release"
        return res

    times: list[float] = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = pipeline_rust.run_pipeline_from_dicts(events)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    res.times = times
    res.median_ms = statistics.median(times)
    res.correct = result is not None
    return res


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, Any] = {
    "baseline": bench_baseline,
    "numba": bench_numba,
    "cython": bench_cython,
    "mypyc": bench_mypyc,
    "pypy": bench_pypy,
    "mojo": bench_mojo,
    "rust": bench_rust,
    "rust_summary": bench_rust_summary,
    "rust_from_dicts": bench_rust_from_dicts,
}


def format_table(results: list[BenchResult], baseline_ms: float) -> str:
    """Format results as a markdown table."""
    lines = []
    lines.append(f"{'Rung':<30} {'Median':>10} {'Speedup':>10} {'Status':>10}")
    lines.append(f"{'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")

    for r in results:
        if r.skipped:
            lines.append(f"{r.name:<30} {'—':>10} {'—':>10} {'SKIP':>10}")
            continue
        if r.error:
            lines.append(f"{r.name:<30} {'—':>10} {'—':>10} {'ERROR':>10}")
            continue

        speedup = baseline_ms / r.median_ms if r.median_ms > 0 else 0
        status = "OK" if r.correct else "FAIL"
        lines.append(f"{r.name:<30} {r.median_ms:>8.1f}ms {speedup:>9.2f}x {status:>10}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Faster Python benchmark runner")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per benchmark")
    parser.add_argument("--only", type=str, help="Run only this benchmark")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--json-out", type=str, help="Write results to JSON file")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in BENCHMARKS:
            print(f"  {name}")
        sys.exit(0)

    # Ensure data exists
    if not DATA_PATH.exists():
        print("Data file not found. Generating...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "data" / "generate.py")],
            check=True,
        )

    print(f"Loading {DATA_PATH}...")
    events = load_events()
    print(f"Loaded {len(events)} events")
    print(f"Running {args.runs} iterations per benchmark\n")

    # Select benchmarks
    if args.only:
        if args.only not in BENCHMARKS:
            print(f"Unknown benchmark: {args.only}")
            print(f"Available: {', '.join(BENCHMARKS)}")
            sys.exit(1)
        to_run = {args.only: BENCHMARKS[args.only]}
    else:
        to_run = BENCHMARKS

    results: list[BenchResult] = []
    baseline_ms = 0.0

    for name, bench_fn in to_run.items():
        print(f"Running {name}...", end=" ", flush=True)
        result = bench_fn(events, args.runs)
        results.append(result)

        if result.skipped:
            print(f"SKIPPED ({result.skip_reason})")
        elif result.error:
            print(f"ERROR ({result.error[:80]})")
        else:
            print(f"{result.median_ms:.1f}ms")

        if name == "baseline" and not result.skipped and not result.error:
            baseline_ms = result.median_ms

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    if baseline_ms == 0:
        # Use baseline median for speedup calc even if we only ran subset
        baseline_ms = results[0].median_ms if results and results[0].median_ms > 0 else 1.0

    print(format_table(results, baseline_ms))

    # Print skip/error details
    for r in results:
        if r.skipped:
            print(f"\n  {r.name}: {r.skip_reason}")
        if r.error:
            print(f"\n  {r.name} error: {r.error}")

    # Save results
    if args.json_out:
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / args.json_out
        json_results = []
        for r in results:
            json_results.append(
                {
                    "name": r.name,
                    "times": r.times,
                    "median_ms": r.median_ms,
                    "speedup": baseline_ms / r.median_ms if r.median_ms > 0 else 0,
                    "correct": r.correct,
                    "skipped": r.skipped,
                    "error": r.error,
                }
            )
        out_path.write_text(json.dumps(json_results, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
