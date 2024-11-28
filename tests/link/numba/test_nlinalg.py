import contextlib

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor import nlinalg, slinalg
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "A, x, lower, exc,test_values",
    [
        (
            pt.dmatrix(),
            pt.dvector(),
            "gen",
            None,
            [
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
                rng.random(size=(3,)).astype("float64"),
            ],
        ),
        (
            pt.lmatrix(),
            pt.dvector(),
            "gen",
            None,
            [
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
                rng.random(size=(3,)).astype("float64"),
            ],
        ),
    ],
)
def test_Solve(A, x, lower, exc, test_values):
    g = slinalg.Solve(lower=lower, b_ndim=1)(A, x)
    inputs = [A, x]
    outputs = []
    if isinstance(g, list):
        outputs = g
    else:
        outputs = [g]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            outputs,
            test_values,
        )


@pytest.mark.parametrize(
    "x, exc,test_values",
    [
        (
            pt.dmatrix(),
            None,
            [(lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64"))],
        ),
        (
            pt.lmatrix(),
            None,
            [(lambda x: x.T.dot(x))(rng.poisson(size=(3, 3)).astype("int64"))],
        ),
    ],
)
def test_Det(x, exc, test_values):
    g = nlinalg.Det()(x)
    inputs = [x]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            [g],
            test_values,
        )


@pytest.mark.parametrize(
    "x, exc,test_values",
    [
        (
            pt.dmatrix(),
            None,
            [(lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64"))],
        ),
        (
            pt.lmatrix(),
            None,
            [(lambda x: x.T.dot(x))(rng.poisson(size=(3, 3)).astype("int64"))],
        ),
    ],
)
def test_SLogDet(x, exc, test_values):
    g = nlinalg.SLogDet()(x)
    inputs = [x]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            g,
            test_values,
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
    "x, exc, test_values",
    [
        (pt.dmatrix(), None, [(lambda x: x.T.dot(x))(x)]),
        (pt.dmatrix(), None, [(lambda x: x.T.dot(x))(y)]),
        (
            pt.lmatrix(),
            None,
            [(lambda x: x.T.dot(x))(rng.integers(1, 10, size=(3, 3)).astype("int64"))],
        ),
    ],
)
def test_Eig(x, exc, test_values):
    g = nlinalg.Eig()(x)
    inputs = [x]
    outputs = []
    if isinstance(g, list):
        outputs = g
    else:
        outputs = [g]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            outputs,
            test_values,
        )


@pytest.mark.parametrize(
    "x, uplo, exc, test_values",
    [
        (
            pt.dmatrix(),
            "L",
            None,
            [(lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64"))],
        ),
        (
            pt.lmatrix(),
            "U",
            UserWarning,
            [(lambda x: x.T.dot(x))(rng.integers(1, 10, size=(3, 3)).astype("int64"))],
        ),
    ],
)
def test_Eigh(x, uplo, exc, test_values):
    g = nlinalg.Eigh(uplo)(x)
    inputs = [x]
    outputs = []
    if isinstance(g, list):
        outputs = g
    else:
        outputs = [g]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            outputs,
            test_values,
        )


@pytest.mark.parametrize(
    "op, x, exc, op_args,test_values",
    [
        (
            nlinalg.MatrixInverse,
            pt.dmatrix(),
            None,
            (),
            [(lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64"))],
        ),
        (
            nlinalg.MatrixInverse,
            pt.lmatrix(),
            None,
            (),
            [(lambda x: x.T.dot(x))(rng.integers(1, 10, size=(3, 3)).astype("int64"))],
        ),
        (
            nlinalg.MatrixPinv,
            pt.dmatrix(),
            None,
            (True,),
            [(lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64"))],
        ),
        (
            nlinalg.MatrixPinv,
            pt.lmatrix(),
            None,
            (False,),
            [(lambda x: x.T.dot(x))(rng.integers(1, 10, size=(3, 3)).astype("int64"))],
        ),
    ],
)
def test_matrix_inverses(op, x, exc, op_args, test_values):
    g = op(*op_args)(x)
    inputs = [x]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            [g],
            test_values,
        )


@pytest.mark.parametrize(
    "x, mode, exc,  test_values",
    [
        (
            pt.dmatrix(),
            "reduced",
            None,
            [(lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64"))],
        ),
        (
            pt.dmatrix(),
            "r",
            None,
            [(lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64"))],
        ),
        (
            pt.lmatrix(),
            "reduced",
            None,
            [(lambda x: x.T.dot(x))(rng.integers(1, 10, size=(3, 3)).astype("int64"))],
        ),
        (
            pt.lmatrix(),
            "complete",
            UserWarning,
            [(lambda x: x.T.dot(x))(rng.integers(1, 10, size=(3, 3)).astype("int64"))],
        ),
    ],
)
def test_QRFull(x, mode, exc, test_values):
    g = nlinalg.QRFull(mode)(x)
    inputs = [x]
    outputs = []
    if isinstance(g, list):
        outputs = g
    else:
        outputs = [g]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            outputs,
            test_values,
        )


@pytest.mark.parametrize(
    "x, full_matrices, compute_uv, exc,test_values",
    [
        (
            pt.dmatrix(),
            True,
            True,
            None,
            [(lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64"))],
        ),
        (
            pt.dmatrix(),
            False,
            True,
            None,
            [(lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64"))],
        ),
        (
            pt.lmatrix(),
            True,
            True,
            None,
            [(lambda x: x.T.dot(x))(rng.integers(1, 10, size=(3, 3)).astype("int64"))],
        ),
        (
            pt.lmatrix(),
            True,
            False,
            None,
            [(lambda x: x.T.dot(x))(rng.integers(1, 10, size=(3, 3)).astype("int64"))],
        ),
    ],
)
def test_SVD(x, full_matrices, compute_uv, exc, test_values):
    g = nlinalg.SVD(full_matrices, compute_uv)(x)
    inputs = [x]
    outputs = []
    if isinstance(g, list):
        outputs = g
    else:
        outputs = [g]

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            inputs,
            outputs,
            test_values,
        )
