#!/usr/bin/env python
import os

import numpy
from setuptools import Extension, setup
from setuptools.dist import Distribution

import versioneer


dist = Distribution()
dist.parse_config_files()


NAME: str = dist.get_name()  # type: ignore

# Handle builds of nightly release
if "BUILD_PYTENSOR_NIGHTLY" in os.environ:
    NAME += "-nightly"

    from versioneer import get_versions as original_get_versions

    def get_versions():
        from datetime import datetime, timezone

        suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
        versions = original_get_versions()
        versions["version"] = versions["version"].split("+")[0] + suffix
        return versions

    versioneer.get_versions = get_versions


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
