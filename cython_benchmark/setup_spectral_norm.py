"""Build script for Cython spectral-norm compilation.

Run: uv run --extra cython python cython_benchmark/setup_spectral_norm.py build_ext --inplace
"""

from Cython.Build import cythonize  # type: ignore[import-untyped]
from setuptools import Extension, setup

extensions = [
    Extension(
        "cython_benchmark.spectral_norm_compiled",
        ["cython_benchmark/spectral_norm.py"],
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
