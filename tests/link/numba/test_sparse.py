from functools import partial

import numpy as np
import pytest
import scipy
import scipy as sp

import pytensor.sparse as ps
import pytensor.tensor as pt


numba = pytest.importorskip("numba")


# Make sure the Numba customizations are loaded
import pytensor.link.numba.dispatch.sparse  # noqa: F401
from pytensor import config
from pytensor.sparse import Dot, SparseTensorType
from tests.link.numba.test_basic import compare_numba_and_py


def sparse_assert_fn(a, b):
    a_is_sparse = sp.sparse.issparse(a)
    assert a_is_sparse == sp.sparse.issparse(b)
    if a_is_sparse:
        assert a.format == b.format
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        np.testing.assert_allclose(a.data, b.data, strict=True)
        np.testing.assert_allclose(a.indices, b.indices, strict=True)
        np.testing.assert_allclose(a.indptr, b.indptr, strict=True)
    else:
        np.testing.assert_allclose(a, b, strict=True)


compare_numba_and_py_sparse = partial(compare_numba_and_py, assert_fn=sparse_assert_fn)


pytestmark = pytest.mark.filterwarnings("error")


def test_sparse_unboxing():
    @numba.njit
    def test_unboxing(x, y):
        return x.shape, y.shape

    x_val = sp.sparse.csr_matrix(np.eye(100))
    y_val = sp.sparse.csc_matrix(np.eye(101))

    res = test_unboxing(x_val, y_val)

    assert res == (x_val.shape, y_val.shape)


def test_sparse_boxing():
    @numba.njit
    def test_boxing(x, y):
        return x, y

    x_val = sp.sparse.csr_matrix(np.eye(100))
    y_val = sp.sparse.csc_matrix(np.eye(101))

    res_x_val, res_y_val = test_boxing(x_val, y_val)

    assert np.array_equal(res_x_val.data, x_val.data)
    assert np.array_equal(res_x_val.indices, x_val.indices)
    assert np.array_equal(res_x_val.indptr, x_val.indptr)
    assert res_x_val.shape == x_val.shape

    assert np.array_equal(res_y_val.data, y_val.data)
    assert np.array_equal(res_y_val.indices, y_val.indices)
    assert np.array_equal(res_y_val.indptr, y_val.indptr)
    assert res_y_val.shape == y_val.shape


def test_sparse_shape():
    @numba.njit
    def test_fn(x):
        return np.shape(x)

    x_val = sp.sparse.csr_matrix(np.eye(100))

    res = test_fn(x_val)

    assert res == (100, 100)


def test_sparse_ndim():
    @numba.njit
    def test_fn(x):
        return x.ndim

    x_val = sp.sparse.csr_matrix(np.eye(100))

    res = test_fn(x_val)

    assert res == 2


def test_sparse_copy():
    @numba.njit
    def test_fn(x):
        return x.copy()

    x = sp.sparse.csr_matrix(np.eye(100))

    y = test_fn(x)
    assert y is not x and np.all(x.data == y.data) and np.all(x.indices == y.indices)


def test_sparse_objmode():
    x = SparseTensorType("csc", dtype=config.floatX)()
    y = SparseTensorType("csc", dtype=config.floatX)()

    out = Dot()(x, y)

    x_val = sp.sparse.random(2, 2, density=0.25, dtype=config.floatX)
    y_val = sp.sparse.random(2, 2, density=0.25, dtype=config.floatX)

    with pytest.warns(
        UserWarning,
        match="Numba will use object mode to run SparseDot's perform method",
    ):
        compare_numba_and_py([x, y], out, [x_val, y_val])


def test_overload_csr_matrix_constructor():
    @numba.njit
    def csr_matrix_constructor(data, indices, indptr):
        return sp.sparse.csr_matrix((data, indices, indptr), shape=(3, 3))

    inp = sp.sparse.random(3, 3, density=0.5, format="csr")

    # Test with pure scipy csr_matrix constructor
    out = sp.sparse.csr_matrix((inp.data, inp.indices, inp.indptr), copy=False)
    # CSR_matrix does a useless slice on data and indices to trim away useless zeros
    # which means these attributes are views of the original arrays.
    assert out.data is not inp.data
    assert not out.data.flags.owndata

    assert out.indices is not inp.indices
    assert not out.indices.flags.owndata

    assert out.indptr is inp.indptr
    assert out.indptr.flags.owndata

    # Test ours
    out_pt = csr_matrix_constructor(inp.data, inp.indices, inp.indptr)
    # Should work the same as Scipy's constructor, because it's ultimately used
    assert isinstance(out_pt, scipy.sparse.csr_matrix)
    assert out_pt.data is not inp.data
    assert not out_pt.data.flags.owndata
    assert (out_pt.data == inp.data).all()

    assert out_pt.indices is not inp.indices
    assert not out_pt.indices.flags.owndata
    assert (out_pt.indices == inp.indices).all()

    assert out_pt.indptr is inp.indptr
    assert out_pt.indptr.flags.owndata
    assert (out_pt.indptr == inp.indptr).all()


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("y_ndim", [0, 1, 2])
def test_simple_graph(y_ndim, format):
    ps_matrix = ps.csr_matrix if format == "csr" else ps.csc_matrix
    x = ps_matrix("x", shape=(3, 3))
    y = pt.tensor("y", shape=(3,) * y_ndim)
    z = ps.sin(x * y)

    rng = np.random.default_rng((155, y_ndim, format == "csr"))
    x_test = sp.sparse.random(3, 3, density=0.5, format=format, random_state=rng)
    y_test = rng.normal(size=(3,) * y_ndim)

    compare_numba_and_py_sparse(
        [x, y],
        z,
        [x_test, y_test],
    )
