"""Python wrapper to call the Rust/PyO3 extension for benchmarking.

The Rust extension takes raw JSON bytes and does everything
(parse + filter + transform + aggregate) in Rust.

Run: uv run rust_benchmark/run_rust_bench.py
(after building: cd rust_benchmark && maturin develop --release)
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "events.json"


def main() -> None:
    try:
        import pipeline_rust  # type: ignore[import-not-found]
    except ImportError:
        print("ERROR: Rust extension not built. Run:")
        print("  cd rust_benchmark && maturin develop --release")
        sys.exit(1)

    json_bytes = DATA_PATH.read_bytes()
    result = pipeline_rust.run_pipeline_from_json(json_bytes)

    print(f"Users:       {result['total_users']}")
    print(f"Events:      {result['total_events']}")
    print(f"Amount:      ${result['total_amount']:,.2f}")
    print(f"High-value:  {result['total_high_value']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
