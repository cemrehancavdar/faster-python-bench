"""Build script for Cython raw JSON pipeline. Pure Python mode, uses yyjson."""

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "cython_benchmark.pipeline_raw",
        sources=["cython_benchmark/pipeline_raw.py"],
        include_dirs=["/opt/homebrew/include"],
        library_dirs=["/opt/homebrew/lib"],
        libraries=["yyjson"],
        extra_compile_args=["-O3"],
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
