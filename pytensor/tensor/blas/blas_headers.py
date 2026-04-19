"""Header text for the C and Fortran BLAS interfaces.

There is no standard name or location for this header, so we just insert it
ourselves into the C code.

The static C declarations are stored in .h files under c_code/ for better
IDE support and maintainability. This module reads those files and assembles
the complete header text, adding dynamic parts like the macOS sdot bug workaround.
"""

import functools
import logging
import os
import sys
from pathlib import Path

from pytensor.configdefaults import config
from pytensor.link.c.cmodule import GCC_compiler


_logger = logging.getLogger("pytensor.tensor.blas")

# Directory containing the C header files
_C_CODE_DIR = Path(__file__).parent / "c_code"


def detect_macos_sdot_bug():
    """
    Try to detect a bug in the BLAS sdot_ routine on macOS.

    Apple's Accelerate framework has a long-standing bug where the Fortran
    interface sdot_() returns incorrect values. The C interface cblas_sdot()
    works correctly. This bug has been present since at least macOS 10.6
    and is STILL PRESENT as of macOS 26 (2026).

    This function compiles and runs a test program to detect the bug,
    then tests if a workaround (using cblas_sdot instead) works.

    Three attributes of this function will be set:
        - detect_macos_sdot_bug.tested: True after first call
        - detect_macos_sdot_bug.present: True if bug is detected
        - detect_macos_sdot_bug.fix_works: True if cblas_sdot workaround works
    """
    _logger.debug("Starting detection of bug in Mac OS BLAS sdot_ routine")
    if detect_macos_sdot_bug.tested:
        return detect_macos_sdot_bug.present

    if sys.platform != "darwin" or not config.blas__ldflags:
        _logger.info("Not Mac OS, no sdot_ bug")
        detect_macos_sdot_bug.tested = True
        return False

    # This code will return -1 if the dot product did not return
    # the right value (30.).
    flags = config.blas__ldflags.split()
    for f in flags:
        # Library directories should also be added as rpath,
        # so that they can be loaded even if the environment
        # variable LD_LIBRARY_PATH does not contain them
        lib_path = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":")
        if f.startswith("-L"):
            flags.append("-Wl,-rpath," + f[2:])
            # also append those paths to DYLD_FALLBACK_LIBRARY_PATH to
            # support libraries that have the wrong install_name
            # (such as MKL on canopy installs)
            if f[2:] not in lib_path:
                lib_path.append(f[2:])
        # this goes into the python process environment that is
        # inherited by subprocesses/used by dyld when loading new objects
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(lib_path)

    test_code = _read_c_code_file("macos_sdot_bugfix/macos_sdot_test.cpp")

    _logger.debug("Trying to compile and run test case.")
    compilation_ok, run_ok = GCC_compiler.try_compile_tmp(
        test_code, tmp_prefix="detect_macos_sdot_bug_", flags=flags, try_run=True
    )
    detect_macos_sdot_bug.tested = True

    # If compilation failed, we consider there is a bug,
    # and the fix does not work
    if not compilation_ok:
        _logger.info("Could not compile test case for sdot_.")
        detect_macos_sdot_bug.present = True
        return True

    if run_ok:
        _logger.info("The sdot_ bug is not present on this system.")
        detect_macos_sdot_bug.present = False
        return False

    # Else, the bug is detected.
    _logger.info("The sdot_ bug is present on this system.")
    detect_macos_sdot_bug.present = True

    # Then, try a simple fix
    test_fix_code = _read_c_code_file("macos_sdot_bugfix/macos_sdot_fix_test.cpp")

    _logger.debug("Trying to compile and run tentative workaround.")
    compilation_fix_ok, run_fix_ok = GCC_compiler.try_compile_tmp(
        test_fix_code,
        tmp_prefix="detect_macos_sdot_bug_testfix_",
        flags=flags,
        try_run=True,
    )

    _logger.info(
        "Status of tentative fix -- compilation OK: %s, works: %s",
        compilation_fix_ok,
        run_fix_ok,
    )
    detect_macos_sdot_bug.fix_works = run_fix_ok

    return detect_macos_sdot_bug.present


detect_macos_sdot_bug.tested = False
detect_macos_sdot_bug.present = False
detect_macos_sdot_bug.fix_works = False


@functools.cache
def _read_c_code_file(filename: str) -> str:
    """Read a C code file from the c_code directory."""
    filepath = _C_CODE_DIR / filename
    try:
        return filepath.read_text(encoding="utf-8")
    except OSError as err:
        msg = f"Unable to load C header file: {filepath}"
        raise OSError(msg) from err


def blas_header_text():
    """C header for the fortran blas interface.

    Returns the complete BLAS header text including:
    - Fortran BLAS declarations (from fortran_blas.h)
    - macOS sdot bug workaround (if applicable)
    - NumPy-based fallback BLAS (if no system BLAS available)
    """
    blas_code = ""
    if not config.blas__ldflags:
        # This code can only be reached by compiling a function with a manually specified GEMM Op.
        # Normal PyTensor usage will end up with Dot22 or Dot22Scalar instead,
        # which opt out of C-code completely if the blas flags are missing
        _logger.warning("Using NumPy C-API based implementation for BLAS functions.")

        # Include the Numpy version implementation of [sd]gemm_.
        try:
            common_code = _read_c_code_file("alt_blas_common.h")
            template_code = _read_c_code_file("alt_blas_template.c")
        except OSError as err:
            msg = "Unable to load NumPy implementation of BLAS functions from C source files."
            raise OSError(msg) from err
        sblas_code = template_code % {
            "float_type": "float",
            "float_size": 4,
            "npy_float": "NPY_FLOAT32",
            "precision": "s",
        }
        dblas_code = template_code % {
            "float_type": "double",
            "float_size": 8,
            "npy_float": "NPY_FLOAT64",
            "precision": "d",
        }
        blas_code += common_code
        blas_code += sblas_code
        blas_code += dblas_code

    # Read the Fortran BLAS declarations from the static header file
    header = _read_c_code_file("fortran_blas.h")

    # Add macOS sdot bug workaround if needed
    if detect_macos_sdot_bug():
        if detect_macos_sdot_bug.fix_works:
            header += _read_c_code_file("macos_sdot_bugfix/macos_sdot_workaround.h")
        else:
            # Make sure the buggy version of sdot_ is never used
            header += _read_c_code_file("macos_sdot_bugfix/macos_sdot_error.h")

    return header + blas_code


@functools.cache
def mkl_threads_text():
    """C header for MKL threads interface."""
    return _read_c_code_file("mkl_threads.h")


@functools.cache
def openblas_threads_text():
    """C header for OpenBLAS threads interface."""
    return _read_c_code_file("openblas_threads.h")


def blas_header_version():
    """Return version tuple for cache invalidation.

    This version should be bumped when:
    - The static header files change
    - The sdot bug workaround logic changes
    """
    # Version 13: Restored macOS sdot bug workaround
    version = (13,)
    if detect_macos_sdot_bug():
        if detect_macos_sdot_bug.fix_works:
            version += (1,)
        else:
            version += (2,)
    return version
