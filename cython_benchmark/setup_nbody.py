"""Build script for Cython n-body compilation.

Run: uv run --extra cython python cython_benchmark/setup_nbody.py build_ext --inplace
"""

from Cython.Build import cythonize  # type: ignore[import-untyped]
from setuptools import Extension, setup

extensions = [
    Extension(
        "cython_benchmark.nbody_compiled",
        ["cython_benchmark/nbody.py"],
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
