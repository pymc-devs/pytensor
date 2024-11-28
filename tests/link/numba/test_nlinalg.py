import contextlib

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor import nlinalg
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "x, exc",
    [
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(rng.poisson(size=(3, 3)).astype("int64")),
            ),
            None,
        ),
    ],
)
def test_Det(x, exc):
    x, test_x = x
    g = nlinalg.Det()(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            g,
            [test_x],
        )


@pytest.mark.parametrize(
    "x, exc",
    [
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(rng.poisson(size=(3, 3)).astype("int64")),
            ),
            None,
        ),
    ],
)
def test_SLogDet(x, exc):
    x, test_x = x
    g = nlinalg.SLogDet()(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            g,
            [test_x],
        )


# We were seeing some weird results in CI where the following two almost
# sign-swapped results were being return from Numba and Python, respectively.
# The issue might be related to https://github.com/numba/numba/issues/4519.
# Regardless, I was not able to reproduce anything like it locally after
# extensive testing.
x = np.array(
    [
        [-0.60407637, -0.71177603, -0.35842241],
        [-0.07735968, 0.50000561, -0.86256007],
        [-0.7931628, 0.49332471, 0.35710434],
    ],
    dtype=np.float64,
)

y = np.array(
    [
        [0.60407637, 0.71177603, -0.35842241],
        [0.07735968, -0.50000561, -0.86256007],
        [0.7931628, -0.49332471, 0.35710434],
    ],
    dtype=np.float64,
)


@pytest.mark.parametrize(
    "x, exc",
    [
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(x),
            ),
            None,
        ),
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(y),
            ),
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            None,
        ),
    ],
)
def test_Eig(x, exc):
    x, test_x = x
    g = nlinalg.Eig()(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            g,
            [test_x],
        )


@pytest.mark.parametrize(
    "x, uplo, exc",
    [
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "L",
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            "U",
            UserWarning,
        ),
    ],
)
def test_Eigh(x, uplo, exc):
    x, test_x = x
    g = nlinalg.Eigh(uplo)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            g,
            [test_x],
        )


@pytest.mark.parametrize(
    "op, x, exc, op_args",
    [
        (
            nlinalg.MatrixInverse,
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
            (),
        ),
        (
            nlinalg.MatrixInverse,
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
            nlinalg.MatrixPinv,
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
            (True,),
        ),
        (
            nlinalg.MatrixPinv,
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


@pytest.mark.parametrize(
    "x, mode, exc",
    [
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "reduced",
            None,
        ),
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "r",
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            "reduced",
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            "complete",
            UserWarning,
        ),
    ],
)
def test_QRFull(x, mode, exc):
    x, test_x = x
    g = nlinalg.QRFull(mode)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            g,
            [test_x],
        )


@pytest.mark.parametrize(
    "x, full_matrices, compute_uv, exc",
    [
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            True,
            True,
            None,
        ),
        (
            (
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            False,
            True,
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            True,
            True,
            None,
        ),
        (
            (
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            True,
            False,
            None,
        ),
    ],
)
def test_SVD(x, full_matrices, compute_uv, exc):
    x, test_x = x
    g = nlinalg.SVD(full_matrices, compute_uv)(x)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            [x],
            g,
            [test_x],
        )
