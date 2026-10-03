"""Ops for using BLAS calls

BLAS = Basic Linear Algebra Subroutines
Learn more about BLAS here:
    http://www.netlib.org/blas/blast-forum/
The standard BLAS libraries implement what is called "legacy BLAS" in that
document.

This documentation describes PyTensor's BLAS optimization pipeline.

Where there is a discrepancy between how things do work and how they *should*
work, both aspects should be documented.

The Ops here are pure Python; their SciPy and C implementations live in the
corresponding backend dispatch registries.


Ops
===

GEMM: Dot22, Dot22Scalar, GemmRelated, Gemm
-------------------------------------------

The BLAS GEMM operation implements Z <- a X Y + b Z,
where Z, X and Y are matrices, and a and b are scalars.

Dot22 is a GEMM where a=1, b=0, and Z is allocated every time.

Dot22Scalar is a GEMM where b=0 and Z is allocated every time.

Gemm is a GEMM in all its generality.

In the future we can refactor the GemmRelated, Gemm, Dot22 and
Dot22Scalar Ops into a single Op.  That new Op (Gemm2) is basically a
normal Gemm, but with an additional configuration variable that says
to ignore the input Z.  Setting that configuration variable to True
would make Gemm2 equivalent to the current Dot22 and Dot22Scalar.
This would make the file a lot easier to read, and save a few hundred
lines of library, to say nothing of testing and documentation.


GEMV: Gemv
----------

The BLAS GEMV operation implements Z <- a X Y + b Z,
where X is a matrix, Y, and Z are vectors, and a and b are scalars.


GER: Ger
--------

The BLAS GER operation implements Z <- a X' Y + Z,
where X and Y are vectors, and matrix Z gets a rank-1 update.


Other Notable BLAS-related Ops
------------------------------

SYRK is another useful special case of GEMM. Particularly SYRK preserves
symmetry in the matrix that it updates.  See how the linear-algebra module uses
symmetry hints before implementing this Op, so that this Op is compatible with
that system.


Optimizations associated with these BLAS Ops are in tensor.rewriting.blas

"""

# Re-export everything for backward compatibility.
# All public symbols that were previously in pytensor.tensor.blas
# must remain importable from this path.

from pytensor.tensor.blas._core import (
    _ldflags,
    _logger,
    ldflags,
    must_initialize_y_gemv,
    view_roots,
)
from pytensor.tensor.blas.batched import (
    BatchedDot,
    _batched_dot,
)
from pytensor.tensor.blas.c_code.blas_headers import (
    blas_header_text,
    blas_header_version,
    mkl_threads_text,
    openblas_threads_text,
)
from pytensor.tensor.blas.gemm import (
    Dot22,
    Dot22Scalar,
    Gemm,
    GemmRelated,
    _dot22,
    _dot22scalar,
)
from pytensor.tensor.blas.gemv import Gemv
from pytensor.tensor.blas.ger import Ger


__all__ = [
    "BatchedDot",
    "Dot22",
    "Dot22Scalar",
    "Gemm",
    "GemmRelated",
    "Gemv",
    "Ger",
    "_batched_dot",
    "_dot22",
    "_dot22scalar",
    "_ldflags",
    "_logger",
    "blas_header_text",
    "blas_header_version",
    "ldflags",
    "mkl_threads_text",
    "must_initialize_y_gemv",
    "openblas_threads_text",
    "view_roots",
]
