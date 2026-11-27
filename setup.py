#!/usr/bin/env python
import os

import numpy
import versioneer
from setuptools import Extension, setup
from setuptools.dist import Distribution


dist = Distribution()
dist.parse_config_files()


NAME: str = dist.get_name()  # type: ignore

# Build without optional compiled extensions. Keep PYODIDE as a compatibility
# alias for existing downstream builds.
is_pure_python = (
    os.getenv("PYTENSOR_PURE_PYTHON", "0") == "1" or os.getenv("PYODIDE", "0") == "1"
)

if is_pure_python:
    # Omit the optional Cython implementation of scan.
    ext_modules = []
else:
    ext_modules = [
        Extension(
            name="pytensor.scan.scan_perform",
            sources=["pytensor/scan/scan_perform.pyx"],
            include_dirs=[numpy.get_include()],
        ),
    ]

if __name__ == "__main__":
    setup(
        name=NAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        ext_modules=ext_modules,
    )
