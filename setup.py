#!/usr/bin/env python
import numpy
from setuptools import Extension, setup
from setuptools.dist import Distribution
import os
import versioneer

dist = Distribution()
dist.parse_config_files()

NAME = dist.get_name()  # type: ignore

# Check if building for Pyodide
is_pyodide = os.getenv('PYODIDE', '0') == '1'

# Define the ext_modules conditionally
ext_modules = []
if not is_pyodide:
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
