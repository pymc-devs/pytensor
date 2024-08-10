#!/usr/bin/env python
import os

import numpy
import versioneer
from setuptools import Extension, setup
from setuptools.dist import Distribution


dist = Distribution()
dist.parse_config_files()


NAME: str = dist.get_name()  # type: ignore

# Check if building for Pyodide
is_pyodide = os.getenv("PYODIDE", "0") == "1"

if is_pyodide:
    # For pyodide we build a universal wheel that must be pure-python
    # so we must omit the cython-version of scan.
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
