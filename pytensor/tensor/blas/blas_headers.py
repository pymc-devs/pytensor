"""Header text for the C and Fortran BLAS interfaces.

There is no standard name or location for this header, so we just insert it
ourselves into the C code.

The static C declarations are stored in .h files under c_code/ for better
IDE support and maintainability. This module reads those files and assembles
the complete header text.
"""

import functools
import logging
from pathlib import Path

from pytensor.configdefaults import config


_logger = logging.getLogger("pytensor.tensor.blas")

# Directory containing the C header files
_C_CODE_DIR = Path(__file__).parent / "c_code"


@functools.cache
def _read_c_code_file(filename: str) -> str:
    """Read a C code file from the c_code directory."""
    filepath = _C_CODE_DIR / filename
    try:
        return filepath.read_text(encoding="utf-8")
    except OSError as err:
        msg = f"Unable to load C header file: {filepath}"
        raise OSError(msg) from err


@functools.cache
def blas_header_text():
    """C header for the fortran blas interface.

    Returns the complete BLAS header text including:
    - Fortran BLAS declarations (from fortran_blas.h)
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

    This version should be bumped when the static header files change.
    """
    # Version 12: Refactored to use external .h files
    return (12,)
