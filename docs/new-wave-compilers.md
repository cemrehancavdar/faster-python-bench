# The New Wave: Codon, Mojo, Taichi

Three tools promise to compile Python (or Python-like code) to native machine code without learning C or Rust. All benchmarks on Apple M4 Pro, CPython 3.14 baseline, median of 5 runs.

| | N-body | Speedup | Spectral-norm | Speedup |
|---|---|---|---|---|
| CPython 3.14 | 1,242ms | 1.0x | 14,046ms | 1.0x |
| Codon 0.19 | 47ms | **26x** | 99ms | **142x** |
| Mojo nightly | 16ms | **78x** | 118ms | **119x** |
| Taichi 1.7 | 16ms | **78x** | 74ms | **190x** |

---

## Codon: Python syntax, AOT compiler

Codon compiles Python to native code via LLVM, ahead-of-time. You write `.py` files, run `codon build -release`, get a binary. On paper: Python syntax, C speed.

In practice: Codon implements its *own* Python — it doesn't use CPython at all. `import json` doesn't work. `sys.version_info` doesn't exist. Standard library coverage is incomplete. You can't call it from CPython via import; it produces standalone binaries. I had to strip out all stdlib imports and rewrite I/O to get benchmarks to compile. The numbers are real — 26x and 142x — but you're writing for Codon's runtime, not Python's.

You also need `DYLD_LIBRARY_PATH=~/.codon/lib/codon` to run the binary on macOS. And `round()` produces fewer decimal places than CPython (e.g. `round(3.14159..., 9)` gives `3.14159` instead of `3.141592654`). Small things, but they add up.

**DX verdict: impressive performance, but you're not speeding up Python — you're rewriting in a Python-flavored language that can't import your existing code.**

### How to reproduce

```bash
~/.codon/bin/codon build -release codon_benchmark/bench.py -o bench_codon
DYLD_LIBRARY_PATH=~/.codon/lib/codon ./bench_codon
```

---

## Mojo: Python-inspired systems language

Mojo is the most honest of the three about what it is: a *new language* that looks like Python. It has value semantics, ownership, `fn` vs `def`, `comptime` parameters, structs instead of classes, no global variables. My benchmark required a full rewrite — not annotation, not decoration, rewrite.

```mojo
struct NBodySystem:
    var x: InlineArray[Float64, NUM_BODIES]
    var vx: InlineArray[Float64, NUM_BODIES]
    var mass: InlineArray[Float64, NUM_BODIES]

    fn advance(mut self, n: Int):
        for _ in range(n):
            for i in range(NUM_BODIES):
                for j in range(i + 1, NUM_BODIES):
                    var dx = self.x[i] - self.x[j]
                    var dsq = dx * dx + dy * dy + dz * dz
                    var mag = DT / (dsq * sqrt(dsq))
```

78x on n-body, 119x on spectral-norm. Competitive with Cython and Numba. But the language is pre-1.0 and moving fast — I hit three breaking changes between the docs and the nightly compiler (`alias` renamed to `comptime`, `InlineArray` initialization syntax changed, `List` requires explicit `^` transfer). It needs `pixi` as a package manager, not pip.

**DX verdict: real performance, but it's a new language with a new toolchain. The "Python superset" marketing is aspirational — today it's closer to "Rust with Python syntax."**

### How to reproduce

```bash
cd mojo_benchmark
pixi run mojo run bench.mojo
```

---

## Taichi: GPU-first Python extension

Taichi was built for physics simulations and GPU computing. You write `@ti.kernel` decorated functions with Taichi type annotations, and it compiles them via LLVM. Data lives in Taichi's own `ti.field()` containers.

```python
@ti.kernel
def nbody_advance(n: ti.i32):
    ti.loop_config(serialize=True)
    for _step in range(n):
        for i in range(NUM_BODIES):
            for j in range(i + 1, NUM_BODIES):
                dx = nbody_x[i] - nbody_x[j]
                dsq = dx * dx + dy * dy + dz * dz
                mag = dt / (dsq * ti.sqrt(dsq))
```

78x on n-body, 190x on spectral-norm — the fastest spectral-norm result after NumPy.

### Gotchas

- **`from __future__ import annotations` silently breaks all `@ti.kernel` decorators.** Taichi checks annotation types by `id()`, and PEP 563 turns them into strings. The error says `ValueError: Invalid data type ti.f64` with no hint that `__future__` annotations is the cause.
- **`ti.loop_config(serialize=True)` is needed on every loop** to prevent auto-parallelization (which would make single-threaded comparison unfair).
- **No Python 3.14 wheels.** Taichi 1.7.4 supports up to Python 3.13. Numbers above were benchmarked on a separate Python 3.13 environment.

**DX verdict: best spectral-norm performance of any non-NumPy tool. But the decorator-based API has surprising failure modes, and you're locked to Taichi's data containers.**

### How to reproduce

```bash
# Requires Python 3.13 (Taichi doesn't support 3.14)
/tmp/taichi-venv/bin/python taichi_benchmark/bench.py
```
