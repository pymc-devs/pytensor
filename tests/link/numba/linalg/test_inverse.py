import contextlib

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "op, x, exc, op_args",
    [
        (
            MatrixInverse,
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
            (),
        ),
        (
            MatrixInverse,
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            None,
            (),
        ),
        (
            MatrixPinv,
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
            (True,),
        ),
        (
            MatrixPinv,
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            None,
            (False,),
        ),
    ],
)
def test_matrix_inverses(op, x, exc, op_args):
    x, test_x = x
    g = op(*op_args)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            g,
            [test_x],
        )
