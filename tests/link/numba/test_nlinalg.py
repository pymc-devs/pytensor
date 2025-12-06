import contextlib

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.tensor import nlinalg
from tests.link.numba.test_basic import compare_numba_and_py


rng = np.random.default_rng(42849)


@pytest.mark.parametrize("dtype", ("float64", "int64"))
@pytest.mark.parametrize(
    "op", (nlinalg.Det(), nlinalg.SLogDet()), ids=["det", "slogdet"]
)
def test_Det_SLogDet(op, dtype):
    x = pt.matrix(dtype=dtype)

    rng = np.random.default_rng([50, sum(map(ord, dtype))])
    x_ = rng.random(size=(3, 3)).astype(dtype)
    test_x = x_.T.dot(x_)

    g = op(x)

    compare_numba_and_py([x], g, [test_x])


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


@pytest.mark.parametrize("input_dtype", ["int64", "float64", "complex128"])
@pytest.mark.parametrize("symmetric", [True, False], ids=["symmetric", "general"])
def test_Eig(input_dtype, symmetric):
    x = pt.matrix("x", dtype=input_dtype)
    if x.type.numpy_dtype.kind in "fc":
        x_val = rng.normal(size=(3, 3)).astype(input_dtype)
    else:
        x_val = rng.integers(1, 10, size=(3, 3)).astype("int64")

    if symmetric:
        x_val = x_val + x_val.T

    def assert_fn(x, y):
        # eig can return equivalent values with some sign flips depending on impl, allow for that
        np.testing.assert_allclose(np.abs(x), np.abs(y), strict=True)

    g = nlinalg.eig(x)
    _, [eigen_values, eigen_vectors] = compare_numba_and_py(
        graph_inputs=[x],
        graph_outputs=g,
        test_inputs=[x_val],
        assert_fn=assert_fn,
    )
    # Check eig is correct
    np.testing.assert_allclose(
        x_val @ eigen_vectors,
        eigen_vectors @ np.diag(eigen_values),
        atol=1e-7,
        rtol=1e-5,
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
