import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import config


numba = pytest.importorskip("numba")


ATOL = 0 if config.floatX.endswith("64") else 1e-6
RTOL = 1e-7 if config.floatX.endswith("64") else 1e-6
rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "b_func, b_size",
    [(pt.matrix, (5, 1)), (pt.matrix, (5, 5)), (pt.vector, (5,))],
    ids=["b_col_vec", "b_matrix", "b_vec"],
)
def test_solve_triangular(b_func, b_size):
    A = pt.matrix("A")
    b = b_func("b")

    X = pt.linalg.solve_triangular(A, b, lower=True)
    f = pytensor.function([A, b], X, mode="NUMBA")

    A_val = np.random.normal(size=(5, 5)).astype(config.floatX)
    A_sym = A_val @ A_val.T
    A_tri = np.linalg.cholesky(A_sym)

    b = np.random.normal(size=b_size).astype(config.floatX)

    X_np = f(A_tri, b)
    np.testing.assert_allclose(A_tri @ X_np, b, atol=ATOL, rtol=RTOL)
