# CPython Version Benchmarks

All benchmarks on Apple M4 Pro, median of 5 runs. CPython 3.14 is the baseline.

## N-body & Spectral-norm

| Version | N-body | vs 3.14 | Spectral-norm | vs 3.14 |
|---|---|---|---|---|
| CPython 3.10 | 1,663ms | 0.75x | 16,826ms | 0.83x |
| CPython 3.11 | 1,200ms | 1.04x | 13,430ms | 1.05x |
| CPython 3.13 | 1,134ms | 1.10x | 13,637ms | 1.03x |
| CPython 3.14 | 1,242ms | 1.0x | 14,046ms | 1.0x |
| CPython 3.14t (free-threaded) | 1,513ms | 0.82x | 14,551ms | 0.97x |

## Key findings

**3.10 to 3.11: the big jump.** A 1.39x speedup on n-body, for free. The [Faster CPython](https://docs.python.org/3/whatsnew/3.11.html) project landed specialized adaptive interpreter, inlined function calls, and faster frame creation. On compute-heavy benchmarks, upgrading from 3.10 to 3.11 gave more speedup than all subsequent versions combined.

**3.14 is slightly slower than 3.13** on these benchmarks (1,242ms vs 1,134ms on n-body). A minor regression.

**Free-threaded Python (3.14t)** is slower than regular 3.14 on single-threaded code — up to 22% on n-body, though only 3% on spectral-norm. The GIL removal requires per-object locking and biased reference counting, adding overhead that varies by workload. The payoff only comes with multiple threads doing real parallel work. For CPU-bound single-threaded code, 3.14t is a regression.

## How to reproduce

```bash
# Self-contained script, works with any CPython 3.10+
uv run --python 3.10 bench_cpython_versions.py
uv run --python 3.14 bench_cpython_versions.py
```
