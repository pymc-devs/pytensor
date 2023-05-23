#!/usr/bin/env python
import numpy
from setuptools import Extension, setup
from setuptools.dist import Distribution

import versioneer


dist = Distribution()
dist.parse_config_files()


NAME: str = dist.get_name()  # type: ignore


if __name__ == "__main__":
    setup(
        name=NAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        ext_modules=[
            Extension(
                name="pytensor.scan.scan_perform",
                sources=["pytensor/scan/scan_perform.pyx"],
                include_dirs=[numpy.get_include()],
            ),
        ],
    )
