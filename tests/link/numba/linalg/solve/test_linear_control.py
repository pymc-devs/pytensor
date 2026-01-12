import numpy as np
import pytest

from pytensor import config
from pytensor import tensor as pt
from tests.link.numba.test_basic import compare_numba_and_py


floatX = config.floatX

pytestmark = pytest.mark.filterwarnings(
    "ignore:numba.core.errors.NumbaPerformanceWarning"
)


def test_solve_sylvester():
    A = pt.matrix("A")
    B = pt.matrix("B")
    C = pt.matrix("C")
    X = pt.linalg.solve_sylvester(A, B, C)

    rng = np.random.default_rng()
    A_val = rng.normal(size=(5, 5)).astype(floatX)
    B_val = rng.normal(size=(5, 5)).astype(floatX)
    C_val = rng.normal(size=(5, 5)).astype(floatX)

    compare_numba_and_py([A, B, C], [X], [A_val, B_val, C_val])


def test_solve_continuous_lyapunov():
    A = pt.matrix("A")
    Q = pt.matrix("Q")
    X = pt.linalg.solve_continuous_lyapunov(A, Q)

    rng = np.random.default_rng()
    A_val = rng.normal(size=(5, 5)).astype(floatX)
    Q_val = rng.normal(size=(5, 5)).astype(floatX)
    Q_val = Q_val @ Q_val.T  # Make Q symmetric positive definite

    compare_numba_and_py([A, Q], [X], [A_val, Q_val])


@pytest.mark.parametrize("method", ["bilinear", "direct"], ids=str)
def test_solve_discrete_lyapunov(method):
    A = pt.matrix("A")
    Q = pt.matrix("Q")
    X = pt.linalg.solve_discrete_lyapunov(A, Q, method=method)

    rng = np.random.default_rng()
    A_val = rng.normal(size=(5, 5)).astype(floatX)
    Q_val = rng.normal(size=(5, 5)).astype(floatX)
    Q_val = Q_val @ Q_val.T  # Make Q symmetric positive definite

    compare_numba_and_py(
        [A, Q],
        [X],
        [A_val, Q_val],
        # object mode fails with 'numpy.dtypes.Int32DType' object has no attribute 'is_precise'
        # when mode is "bilinear"
        eval_obj_mode=method == "direct",
    )
