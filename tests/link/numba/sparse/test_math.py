import numpy as np
import pytest
import scipy

import pytensor.sparse as ps
import pytensor.tensor as pt
from tests.link.numba.sparse.test_basic import compare_numba_and_py_sparse


pytestmark = pytest.mark.filterwarnings("error")

DOT_SHAPES = [((20, 11), (11, 4)), ((10, 3), (3, 1))]


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("y_ndim", [0, 1, 2])
def test_sparse_dense_multiply(y_ndim, format):
    x = ps.matrix(format, name="x", shape=(3, 3))
    y = pt.tensor("y", shape=(3,) * y_ndim)
    z = x * y

    rng = np.random.default_rng((155, y_ndim, format == "csr"))
    x_test = scipy.sparse.random(3, 3, density=0.5, format=format, random_state=rng)
    y_test = rng.normal(size=(3,) * y_ndim)

    compare_numba_and_py_sparse(
        [x, y],
        z,
        [x_test, y_test],
    )


@pytest.mark.parametrize("sp_format", ["csr", "csc"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_dot_sparse_dense(sp_format, x_shape, y_shape):
    x = ps.matrix(format=sp_format, name="x", shape=x_shape)
    y = pt.matrix("y", shape=y_shape)
    z = ps.dot(x, y)

    rng = np.random.default_rng(sum(map(ord, sp_format)) + sum(x_shape) + sum(y_shape))
    x_test = scipy.sparse.random(
        *x_shape, density=0.5, format=sp_format, random_state=rng
    )
    y_test = rng.normal(size=y_shape)

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize("sp_format", ["csr", "csc"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_dot_dense_sparse(sp_format, x_shape, y_shape):
    x = pt.matrix(name="x", shape=x_shape)
    y = ps.matrix(format=sp_format, name="y", shape=y_shape)
    z = ps.dot(x, y)

    rng = np.random.default_rng(sum(map(ord, sp_format)) + sum(x_shape) + sum(y_shape))
    x_test = rng.normal(size=x_shape)
    y_test = scipy.sparse.random(
        *y_shape, density=0.5, format=sp_format, random_state=rng
    )

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])


@pytest.mark.parametrize("x_format", ["csr", "csc"])
@pytest.mark.parametrize("y_format", ["csr", "csc"])
@pytest.mark.parametrize("x_shape, y_shape", DOT_SHAPES)
def test_sparse_dot_sparse_sparse(x_format, y_format, x_shape, y_shape):
    x = ps.matrix(x_format, name="x", shape=x_shape)
    y = ps.matrix(y_format, name="y", shape=y_shape)
    z = ps.dot(x, y)

    rng = np.random.default_rng(sum(map(ord, x_format)) + sum(map(ord, y_format)))
    x_test = scipy.sparse.random(
        *x_shape, density=0.5, format=x_format, random_state=rng
    )
    y_test = scipy.sparse.random(
        *y_shape, density=0.5, format=y_format, random_state=rng
    )

    compare_numba_and_py_sparse([x, y], z, [x_test, y_test])
