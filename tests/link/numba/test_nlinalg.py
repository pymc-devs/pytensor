import contextlib

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import nlinalg, slinalg
from tests.link.numba.test_basic import compare_numba_and_py, set_test_value


rng = np.random.default_rng(42849)


@pytest.mark.parametrize(
    "inputs, lower, exc",
    [
        (
            [
                set_test_value(
                    pt.dmatrix(),
                    (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
                ),
                set_test_value(pt.dvector(), rng.random(size=(3,)).astype("float64")),
            ],
            "gen",
            None,
        ),
        (
            [
                set_test_value(
                    pt.lmatrix(),
                    (lambda x: x.T.dot(x))(
                        rng.integers(1, 10, size=(3, 3)).astype("int64")
                    ),
                ),
                set_test_value(pt.dvector(), rng.random(size=(3,)).astype("float64")),
            ],
            "gen",
            None,
        ),
    ],
)
def test_Solve(inputs, lower, exc):
    test_values = {k: v for d in inputs for k, v in d.items()}
    inputs = list(test_values.keys())
    g = slinalg.Solve(lower=lower, b_ndim=1)(*inputs)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "test_values, exc",
    [
        (
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
        ),
        (
            set_test_value(
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(rng.poisson(size=(3, 3)).astype("int64")),
            ),
            None,
        ),
    ],
)
def test_Det(test_values, exc):
    x = next(iter(test_values.keys()))
    g = nlinalg.Det()(x)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "test_values, exc",
    [
        (
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
        ),
        (
            set_test_value(
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(rng.poisson(size=(3, 3)).astype("int64")),
            ),
            None,
        ),
    ],
)
def test_SLogDet(test_values, exc):
    x = next(iter(test_values.keys()))
    g = nlinalg.SLogDet()(x)
    g_fg = FunctionGraph(outputs=g)

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
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
    "test_values, exc",
    [
        (
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(x),
            ),
            None,
        ),
        (
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(y),
            ),
            None,
        ),
        (
            set_test_value(
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            None,
        ),
    ],
)
def test_Eig(test_values, exc):
    x = next(iter(test_values.keys()))
    g = nlinalg.Eig()(x)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "test_values, uplo, exc",
    [
        (
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "L",
            None,
        ),
        (
            set_test_value(
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
def test_Eigh(test_values, uplo, exc):
    x = next(iter(test_values.keys()))
    g = nlinalg.Eigh(uplo)(x)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "op, test_values, exc, op_args",
    [
        (
            nlinalg.MatrixInverse,
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
            (),
        ),
        (
            nlinalg.MatrixInverse,
            set_test_value(
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
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            None,
            (True,),
        ),
        (
            nlinalg.MatrixPinv,
            set_test_value(
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
def test_matrix_inverses(op, test_values, exc, op_args):
    x = next(iter(test_values.keys()))
    g = op(*op_args)(x)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "test_values, mode, exc",
    [
        (
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "reduced",
            None,
        ),
        (
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            "r",
            None,
        ),
        (
            set_test_value(
                pt.lmatrix(),
                (lambda x: x.T.dot(x))(
                    rng.integers(1, 10, size=(3, 3)).astype("int64")
                ),
            ),
            "reduced",
            None,
        ),
        (
            set_test_value(
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
def test_QRFull(test_values, mode, exc):
    x = next(iter(test_values.keys()))
    g = nlinalg.QRFull(mode)(x)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "test_values, full_matrices, compute_uv, exc",
    [
        (
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            True,
            True,
            None,
        ),
        (
            set_test_value(
                pt.dmatrix(),
                (lambda x: x.T.dot(x))(rng.random(size=(3, 3)).astype("float64")),
            ),
            False,
            True,
            None,
        ),
        (
            set_test_value(
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
            set_test_value(
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
def test_SVD(test_values, full_matrices, compute_uv, exc):
    x = next(iter(test_values.keys()))
    g = nlinalg.SVD(full_matrices, compute_uv)(x)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                test_values[i]
                for i in test_values
                if not isinstance(i, SharedVariable | Constant)
            ],
        )
