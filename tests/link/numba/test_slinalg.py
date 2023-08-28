import numba as nb
import numpy as np
import pytest
from scipy import linalg

import pytensor
import pytensor.tensor as pt
from pytensor import config


ATOL = 0 if config.floatX.endswith("64") else 1e-6
RTOL = 1e-7 if config.floatX.endswith("64") else 1e-6


def test_solve_triangular():
    A = pt.matrix("A")
    b = pt.matrix("b")

    X = pt.linalg.solve_triangular(A, b, lower=True)
    f = pytensor.function([A, b], X, mode="NUMBA")

    A_val = np.random.normal(size=(5, 5)).astype(config.floatX)
    A_sym = A_val @ A_val.T
    A_tri = np.linalg.cholesky(A_sym)

    b = np.random.normal(size=(5, 1)).astype(config.floatX)

    X_np = f(A_tri, b)
    np.testing.assert_allclose(A_tri @ X_np, b, atol=ATOL, rtol=RTOL)


def test_scipy_solve_triangular_not_overloaded():
    A_val = np.random.normal(size=(5, 5)).astype(config.floatX)
    A_sym = A_val @ A_val.T
    A_tri = np.linalg.cholesky(A_sym)
    b = np.random.normal(size=(5, 1)).astype(config.floatX)

    @nb.njit
    def test_solve_tri(a, b):
        return linalg.solve_triangular(a, b)

    with pytest.raises(nb.TypingError, match="Failed in nopython mode"):
        test_solve_tri(A_tri, b)
