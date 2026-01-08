from collections.abc import Callable

import numpy as np
from numba.core.extending import overload
from numba.core.types import Float
from numba.np.linalg import ensure_lapack
from scipy import linalg

from pytensor.link.numba.dispatch.linalg.decomposition.lu_factor import _getrf
from pytensor.link.numba.dispatch.linalg.solve.lu_solve import _getrs
from pytensor.link.numba.dispatch.linalg.solve.utils import _solve_check_input_shapes
from pytensor.link.numba.dispatch.linalg.utils import (
    _check_dtypes_match,
    _check_linalg_matrix,
)


def _solve_gen(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    transposed: bool,
):
    """Thin wrapper around scipy.linalg.solve. Used as an overload target for numba to avoid unexpected side-effects
    for users who import pytensor."""
    return linalg.solve(
        A,
        B,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=False,
        assume_a="gen",
        transposed=transposed,
    )


@overload(_solve_gen)
def solve_gen_impl(
    A: np.ndarray,
    B: np.ndarray,
    lower: bool,
    overwrite_a: bool,
    overwrite_b: bool,
    transposed: bool,
) -> Callable[[np.ndarray, np.ndarray, bool, bool, bool, bool], np.ndarray]:
    ensure_lapack()
    _check_linalg_matrix(A, ndim=2, dtype=Float, func_name="solve")
    _check_linalg_matrix(B, ndim=(1, 2), dtype=Float, func_name="solve")
    _check_dtypes_match((A, B), "solve")

    def impl(
        A: np.ndarray,
        B: np.ndarray,
        lower: bool,
        overwrite_a: bool,
        overwrite_b: bool,
        transposed: bool,
    ) -> np.ndarray:
        _N = np.int32(A.shape[-1])
        _solve_check_input_shapes(A, B)

        if overwrite_a and A.flags.c_contiguous:
            # Work with the transposed system to avoid copying A
            A = A.T
            transposed = not transposed

        LU, IPIV, INFO1 = _getrf(A, overwrite_a=overwrite_a)

        X, INFO2 = _getrs(
            LU=LU,
            B=B,
            IPIV=IPIV,
            trans=transposed,
            overwrite_b=overwrite_b,
        )

        if INFO1 != 0 or INFO2 != 0:
            X = np.full_like(X, np.nan)

        return X

    return impl
