"""Build script for Cython pure Python mode compilation.

Run: uv run --extra cython python cython_benchmark/setup.py build_ext --inplace
"""

from Cython.Build import cythonize  # type: ignore[import-untyped]
from setuptools import Extension, setup

extensions = [
    Extension(
        "cython_benchmark.pipeline_compiled",
        ["cython_benchmark/pipeline.py"],
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
        },
    ),
)
