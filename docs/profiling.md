# Finding the Hot Loop: Python Profiling Guide

Before reaching for any optimization tool, you need to know *where* the time goes. Here are the three profilers worth knowing, and the workflow for using them together.

---

## cProfile: the built-in

```bash
python -m cProfile -s cumulative my_script.py
```

Ships with Python. Shows cumulative time per function. Good for identifying which functions are hot. Bad for telling you *which lines* in a function are hot. Start here.

---

## py-spy: the sampling profiler

```bash
py-spy top --pid 12345
py-spy record -o profile.svg -- python my_script.py
```

Attaches to a running process without modifying it. Generates flame graphs. Low overhead (sampling, not tracing). Good for production profiling. The flame graph instantly shows you where the stack spends time.

Install: `pip install py-spy`

---

## line_profiler: the scalpel

```python
@profile
def advance(bodies, dt):
    for i in range(len(bodies)):
        ...
```

```bash
kernprof -l -v my_script.py
```

Shows time per line within a function. Use this *after* cProfile or py-spy tells you which function is hot. It answers: "is the bottleneck the dict lookup, the string operation, or the math?"

Install: `pip install line_profiler`

---

## The workflow

1. **cProfile** to find the hot function
2. **py-spy** if you need a visual (flame graph)
3. **line_profiler** to find the hot line
4. *Then* pick a tool from the optimization ladder

Only after step 3 should you decide between Numba, Cython, Rust, or "just restructure the data."

---

## Decision framework

After profiling:

1. Is the total time actually a problem? (Not "it could be faster" — "it's failing SLAs" or "users are waiting.")
2. Did the profiler confirm where the time goes?
3. Is the hot path compute-bound (loops, math) or data-bound (dicts, strings, I/O)?
4. If compute-bound: Numba or Cython. If data-bound: probably architecture, not language.
