from functools import partial
from sys import getrefcount

import numpy as np
import pytest
import scipy
import scipy as sp

import pytensor.sparse as ps
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.sparse.variable import SparseConstant
from pytensor.tensor.type import DenseTensorType


numba = pytest.importorskip("numba")


# Make sure the Numba customizations are loaded
import pytensor.link.numba.dispatch.sparse  # noqa: F401
from pytensor import config
from pytensor.sparse import SparseTensorType
from tests.link.numba.test_basic import compare_numba_and_py


pytestmark = pytest.mark.filterwarnings("error")


def sparse_assert_fn(a, b):
    a_is_sparse = sp.sparse.issparse(a)
    assert a_is_sparse == sp.sparse.issparse(b)
    if a_is_sparse:
        # Attributes can be compared only if both matrices have sorted indices
        if not a.has_sorted_indices:
            a = a.sorted_indices()
        if not b.has_sorted_indices:
            b = b.sorted_indices()

        assert a.format == b.format
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        np.testing.assert_allclose(a.data, b.data, strict=True)
        np.testing.assert_allclose(a.indices, b.indices, strict=True)
        np.testing.assert_allclose(a.indptr, b.indptr, strict=True)
    else:
        np.testing.assert_allclose(a, b, strict=True)


compare_numba_and_py_sparse = partial(compare_numba_and_py, assert_fn=sparse_assert_fn)


def test_sparse_boxing():
    @numba.njit
    def boxing_fn(x, y):
        return x, y, y.data.sum()

    x_val = sp.sparse.csr_matrix(np.eye(100))
    y_val = sp.sparse.csc_matrix(np.eye(101))

    res_x_val, res_y_val, res_y_sum = boxing_fn(x_val, y_val)

    assert np.array_equal(res_x_val.data, x_val.data)
    assert np.array_equal(res_x_val.indices, x_val.indices)
    assert np.array_equal(res_x_val.indptr, x_val.indptr)
    assert res_x_val.shape == x_val.shape

    assert np.array_equal(res_y_val.data, y_val.data)
    assert np.array_equal(res_y_val.indices, y_val.indices)
    assert np.array_equal(res_y_val.indptr, y_val.indptr)
    assert res_y_val.shape == y_val.shape

    np.testing.assert_allclose(res_y_sum, y_val.sum())


def test_sparse_creation_refcount():
    @numba.njit
    def create_csr_matrix(data, indices, ind_ptr):
        return scipy.sparse.csr_matrix((data, indices, ind_ptr), shape=(5, 5))

    x = scipy.sparse.random(5, 5, density=0.5, format="csr")

    x_data = x.data
    x_indptr = x.indptr
    assert getrefcount(x_data) == 3
    assert getrefcount(x_indptr) == 3

    for i in range(5):
        a = create_csr_matrix(x.data, x.indices, x.indptr)

    # a.data is a view of the underlying data under x.data, but doesn't reference it directly
    assert getrefcount(x_data) == 3
    # x_indptr is reused directly
    assert getrefcount(x_indptr) == 4

    del a
    assert getrefcount(x_data) == 3
    assert getrefcount(x_indptr) == 3


def test_sparse_passthrough_refcount():
    @numba.njit
    def identity(a):
        return a

    x = scipy.sparse.random(5, 5, density=0.5, format="csr")

    x_data = x.data
    assert getrefcount(x_data) == 3

    for i in range(5):
        identity(x)

    assert getrefcount(x_data) == 3


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
    assert y is not x
    for attr in ("data", "indices", "indptr"):
        y_data = getattr(y, attr)
        x_data = getattr(x, attr)
        assert y_data is not x_data
        assert not np.shares_memory(y_data, x_data)
        assert (y_data == x_data).all()


@pytest.mark.parametrize(
    "func", [sp.sparse.csr_matrix, sp.sparse.csc_matrix], ids=["csr", "csc"]
)
def test_sparse_constructor(func):
    @numba.njit
    def csr_matrix_constructor(data, indices, indptr):
        return func((data, indices, indptr), shape=(3, 3))

    inp = sp.sparse.random(3, 3, density=0.5, format="csr")

    # Test with pure scipy constructor
    out = func((inp.data, inp.indices, inp.indptr), copy=False)
    # Scipy does a useless slice on data and indices to trim away useless zeros
    # which means these attributes are views of the original arrays.
    assert out.data is not inp.data
    assert not out.data.flags.owndata

    assert out.indices is not inp.indices
    assert not out.indices.flags.owndata

    assert out.indptr is inp.indptr

    # Test numba impl
    out_pt = csr_matrix_constructor(inp.data, inp.indices, inp.indptr)
    # Should work the same as Scipy's constructor, because it's ultimately used
    assert type(out_pt) is type(out)

    assert out_pt.data is not inp.data
    assert not out_pt.data.flags.owndata
    assert np.shares_memory(out_pt.data, inp.data)
    assert (out_pt.data == inp.data).all()

    assert out_pt.indices is not inp.indices
    assert not out_pt.indices.flags.owndata
    assert np.shares_memory(out_pt.indices, inp.indices)
    assert (out_pt.indices == inp.indices).all()

    assert out_pt.indptr is inp.indptr


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sparse_constant(format, cache):
    x = sp.sparse.random(3, 3, density=0.5, format=format, random_state=166)
    x = ps.as_sparse(x)
    assert isinstance(x, SparseConstant)
    assert x.type.format == format
    y = pt.vector("y", shape=(3,))
    out = x * y

    y_test = np.array([np.pi, np.e, np.euler_gamma])
    with config.change_flags(numba__cache=cache):
        compare_numba_and_py_sparse(
            [y],
            [out],
            [y_test],
            eval_obj_mode=False,
        )


@pytest.mark.parametrize("format", ["csc", "csr"])
@pytest.mark.parametrize("dense_out", [True, False])
def test_sparse_objmode(format, dense_out):
    class SparseTestOp(Op):
        def make_node(self, x):
            out = x.type.clone(shape=(1, x.type.shape[-1]))()
            if dense_out:
                out = out.todense().type()
            return Apply(self, [x], [out])

        def perform(self, node, inputs, output_storage):
            [x] = inputs
            [out] = output_storage
            out[0] = x[0]
            if dense_out:
                out[0] = out[0].todense()

    x = ps.matrix(format, dtype=config.floatX, shape=(5, 5), name="x")

    out = SparseTestOp()(x)
    assert out.type.shape == (1, 5)
    assert isinstance(out.type, DenseTensorType if dense_out else SparseTensorType)

    x_val = sp.sparse.random(5, 5, density=0.25, dtype=config.floatX, format=format)

    with pytest.warns(
        UserWarning,
        match="Numba will use object mode to run SparseTestOp's perform method",
    ):
        compare_numba_and_py_sparse([x], out, [x_val])


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_simple_graph(format):
    x = ps.matrix(format, name="x", shape=(3, 3))
    y = pt.tensor("y", shape=(3,))
    z = ps.math.sin(x * y)

    rng = np.random.default_rng((155, format == "csr"))
    x_test = sp.sparse.random(3, 3, density=0.5, format=format, random_state=rng)
    y_test = rng.normal(size=(3,))

    compare_numba_and_py_sparse(
        [x, y],
        z,
        [x_test, y_test],
    )


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_deepcopy(format):
    x = ps.matrix(shape=(3, 3), format=format)
    x_test = sp.sparse.random(3, 3, density=0.5, format=format)
    compare_numba_and_py_sparse([x], [x], [x_test])


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_dense_from_sparse(format):
    x = ps.matrix(shape=(5, 3), format=format)
    x_test = sp.sparse.random(5, 3, density=0.5, format=format)
    y = ps.dense_from_sparse(x)
    compare_numba_and_py_sparse([x], y, [x_test])


def test_sparse_conversion():
    @numba.njit
    def to_csr(matrix):
        return matrix.tocsr()

    @numba.njit
    def to_csc(matrix):
        return matrix.tocsc()

    x_csr = scipy.sparse.random(5, 5, density=0.5, format="csr")
    x_csc = x_csr.tocsc()
    x_dense = x_csr.todense()

    for x_inp in (x_csr, x_csc):
        for output_format in ("csr", "csc"):
            if output_format == "csr":
                res = to_csr(x_inp)
            else:
                res = to_csc(x_inp)
            assert res.format == output_format
            np.testing.assert_array_equal(res.todense(), x_dense)


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_from_dense(format):
    x = pt.matrix(name="x", shape=(11, 7), dtype=config.floatX)
    x_sparse = sp.sparse.random(11, 7, density=0.4, format=format, dtype=config.floatX)
    x_test = x_sparse.toarray()

    if format == "csr":
        y = ps.csr_from_dense(x)
    else:
        y = ps.csc_from_dense(x)

    compare_numba_and_py_sparse([x], y, [x_test])
