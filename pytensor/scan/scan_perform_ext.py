"""

To update the `Scan` Cython code you must
- Update `scan_perform.pyx`
- update the version value in this file and in `scan_perform.pyx`

"""

from pytensor.scan.scan_perform import get_version, perform  # noqa: F401


version = 0.326  # must match constant returned in function get_version()
assert version == get_version(), (
    "Invalid extension, check the installation process, "
    "could be problem with .pyx file or Cython ext build process."
)
del get_version
