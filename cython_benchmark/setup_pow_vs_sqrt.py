"""Build script for the ** vs sqrt() Cython microbenchmark.

Run: uv run --extra cython python cython_benchmark/setup_pow_vs_sqrt.py build_ext --inplace
"""

from Cython.Build import cythonize  # type: ignore[import-untyped]
from setuptools import Extension, setup

extensions = [
    Extension(
        "cython_benchmark.bench_pow_vs_sqrt_compiled",
        ["cython_benchmark/bench_pow_vs_sqrt.py"],
        extra_compile_args=["-march=native", "-ffast-math"],
    ),
]

setup(
    packages=[],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
