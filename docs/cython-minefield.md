# The Cython Minefield

Cython's failure mode is the most dangerous one on the entire optimization ladder: **your code works, it's just 12x slower than it should be, and nothing tells you.**

My first Cython n-body got **10.5x**. My final version got **124x**. Same Cython, same compiler. Here's what happened between them.

---

## Landmine 1: `**` operator with float exponents -- 7x penalty

The baseline computes gravitational force magnitude as:

```python
mag = dt * ((dx*dx + dy*dy + dz*dz) ** (-1.5))
```

Even with typed `cython.double` variables and `-ffast-math`, Cython's `**` operator with a float exponent goes through a slow dispatch path instead of compiling to a direct C `pow()` call. Decomposing it into `sqrt()` + arithmetic avoids this entirely:

```python
from cython.cimports.libc.math import sqrt
dsq = dx*dx + dy*dy + dz*dz
mag = dt / (dsq * sqrt(dsq))
```

This single change: **7x speedup**. No warning, no error, no yellow line in the Cython annotation report.

---

## Landmine 2: Pair index arrays vs nested loops -- 2x penalty

The algorithm needs to iterate over all pairs (i, j) where i < j. Two ways:

```python
# Version A: precomputed pair indices
pairs_i = [0,0,0,0,1,1,1,2,2,3]
pairs_j = [1,2,3,4,2,3,4,3,4,4]
for k in range(10):
    i, j = pairs_i[k], pairs_j[k]
    dx = x[i] - x[j]

# Version B: nested loops
for i in range(5):
    for j in range(i+1, 5):
        dx = x[i] - x[j]
```

Version A looks clever -- flatten the pair iteration into a single loop. But the C compiler can't unroll a loop that indexes through an array. Version B's nested loops with compile-time-known bounds get fully unrolled and vectorized. **2x difference**, and version A *looks* more optimized.

---

## Landmine 3: Missing `@cython.cdivision(True)` -- silent penalty

Without this decorator, Cython inserts a zero-division check before every floating-point divide. In the inner loop that runs 5 million times per benchmark run, that's 5 million branches that are never taken.

---

## The diagnostic tool nobody uses

Cython has an annotation report (`cython -a file.py`) that highlights lines in yellow when they involve Python object operations. A score-0 line (no yellow) means pure C. But you have to *know* to run it, and you have to *know* what yellow means. There's no integration with your editor. There's no flag that says "this hot loop is 90% C but 10% Python."

---

## What the final 124x version required

All of these, applied together:

- `@cython.cfunc` instead of `@cython.ccall`
- `cython.double[5]` C arrays instead of Python lists
- `sqrt()` + arithmetic instead of `** (-1.5)` (Cython's `**` dispatch is slow)
- Nested loops instead of index arrays
- `@cython.cdivision(True)`
- Local caching of struct fields

Every line in the hot loop has score-0 in the annotation report. Getting there required: learning C's mental model, expressing it in Python syntax, and using a separate diagnostic tool to verify the compiler did what you think.

**Cython promises to make "writing C extensions for Python as easy as Python itself." The reality is: learn C's mental model, express it in Python syntax, and use a separate diagnostic tool to verify.**

---

## How to reproduce

```bash
# Build Cython n-body extension
uv run python cython_benchmark/setup_nbody.py build_ext --inplace

# Run with annotation report
uv run cython -a cython_benchmark/nbody.py

# Benchmark all suites
uv run bench_all.py --suite nbody
```
