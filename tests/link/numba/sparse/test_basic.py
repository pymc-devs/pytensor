from functools import partial
from sys import getrefcount

import numpy as np
import pytest
import scipy as sp

import pytensor.sparse as ps
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.sparse.variable import SparseConstant
from pytensor.tensor.type import DenseTensorType


numba = pytest.importorskip("numba")


# Make sure the Numba customizations are loaded
import pytensor.link.numba.dispatch.sparse  # noqa: F401
from pytensor import config, function
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
        return sp.sparse.csr_matrix((data, indices, ind_ptr), shape=(5, 5))

    x = sp.sparse.random(5, 5, density=0.5, format="csr")

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

    x = sp.sparse.random(5, 5, density=0.5, format="csr")

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

    x_csr = sp.sparse.random(5, 5, density=0.5, format="csr")
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


@pytest.mark.parametrize("output_format", ("csr", "csc"))
@pytest.mark.parametrize(
    "input_formats",
    (
        ("csr", "csr", "csr"),
        ("csc", "csc", "csc"),
        ("csr", "csc", "csr"),
        ("csc", "csr", "csc"),
        ("csc", "csc", "csr"),
    ),
)
def test_sparse_hstack(output_format, input_formats):
    x1 = ps.matrix(
        name="x1", shape=(7, 2), format=input_formats[0], dtype=config.floatX
    )
    x2 = ps.matrix(
        name="x2", shape=(7, 1), format=input_formats[1], dtype=config.floatX
    )
    x3 = ps.matrix(
        name="x3", shape=(7, 5), format=input_formats[2], dtype=config.floatX
    )
    z = ps.hstack([x1, x2, x3], format=output_format, dtype=config.floatX)
    x1_test = sp.sparse.random(
        7,
        2,
        density=0.5,
        format=input_formats[0],
        dtype=config.floatX,
    )
    x2_test = sp.sparse.random(
        7,
        1,
        density=0.3,
        format=input_formats[1],
        dtype=config.floatX,
    )
    x3_test = sp.sparse.random(
        7,
        5,
        density=0.4,
        format=input_formats[2],
        dtype=config.floatX,
    )

    compare_numba_and_py_sparse([x1, x2, x3], z, [x1_test, x2_test, x3_test])


@pytest.mark.parametrize("output_format", ("csr", "csc"))
@pytest.mark.parametrize(
    "input_formats",
    (
        ("csr", "csr", "csr"),
        ("csc", "csc", "csc"),
        ("csr", "csc", "csr"),
        ("csc", "csr", "csc"),
        ("csc", "csc", "csr"),
    ),
)
def test_sparse_vstack(output_format, input_formats):
    x1 = ps.matrix(
        name="x1", shape=(2, 11), format=input_formats[0], dtype=config.floatX
    )
    x2 = ps.matrix(
        name="x2", shape=(1, 11), format=input_formats[1], dtype=config.floatX
    )
    x3 = ps.matrix(
        name="x3", shape=(5, 11), format=input_formats[2], dtype=config.floatX
    )
    z = ps.vstack([x1, x2, x3], format=output_format, dtype=config.floatX)
    x1_test = sp.sparse.random(
        2,
        11,
        density=0.4,
        format=input_formats[0],
        dtype=config.floatX,
    )
    x2_test = sp.sparse.random(
        1,
        11,
        density=0.5,
        format=input_formats[1],
        dtype=config.floatX,
    )
    x3_test = sp.sparse.random(
        5,
        11,
        density=0.2,
        format=input_formats[2],
        dtype=config.floatX,
    )

    compare_numba_and_py_sparse([x1, x2, x3], z, [x1_test, x2_test, x3_test])


def test_sparse_hstack_mismatched_rows_raises():
    x = ps.matrix(name="x", shape=(None, 5), format="csr", dtype=config.floatX)
    y = ps.matrix(name="y", shape=(None, 7), format="csr", dtype=config.floatX)
    z = ps.hstack([x, y], format="csr", dtype=config.floatX)
    fn = function([x, y], z, mode="NUMBA")

    x_test = sp.sparse.random(3, 5, density=0.4, format="csr", dtype=config.floatX)
    y_test = sp.sparse.random(4, 7, density=0.4, format="csr", dtype=config.floatX)

    with pytest.raises(ValueError, match="Mismatching dimensions along axis 0"):
        fn(x_test, y_test)


def test_sparse_vstack_mismatched_cols_raises():
    x = ps.matrix(name="x", shape=(10, None), format="csr", dtype=config.floatX)
    y = ps.matrix(name="y", shape=(13, None), format="csr", dtype=config.floatX)
    z = ps.vstack([x, y], format="csr", dtype=config.floatX)
    fn = function([x, y], z, mode="NUMBA")

    x_test = sp.sparse.random(10, 3, density=0.4, format="csr", dtype=config.floatX)
    y_test = sp.sparse.random(13, 4, density=0.4, format="csr", dtype=config.floatX)

    with pytest.raises(ValueError, match="Mismatching dimensions along axis 1"):
        fn(x_test, y_test)


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_col_scale(format):
    x = ps.matrix(format, name="x", shape=(8, 10), dtype=config.floatX)
    v = pt.vector(name="v", shape=(10,), dtype=config.floatX)
    z = ps.col_scale(x, v)
    x_test = sp.sparse.random(8, 10, density=0.4, format=format, dtype=config.floatX)
    s_test = np.random.random(10).astype(config.floatX)

    compare_numba_and_py_sparse([x, v], z, [x_test, s_test])


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_row_scale(format):
    x = ps.matrix(format, name="x", shape=(7, 10), dtype=config.floatX)
    v = pt.vector(name="v", shape=(7,), dtype=config.floatX)
    z = ps.row_scale(x, v)
    x_test = sp.sparse.random(7, 10, density=0.4, format=format, dtype=config.floatX)
    v_test = np.random.random(7).astype(config.floatX)

    compare_numba_and_py_sparse([x, v], z, [x_test, v_test])


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_get_item_list(format):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    idx = pt.ivector("idx")
    z = ps.get_item_list(x, idx)

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)
    idx_test = np.asarray([0, 2, 5, 2], dtype=np.int32)

    compare_numba_and_py_sparse([x, idx], z, [x_test, idx_test])


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_get_item_list_wrong_index(format):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    idx = pt.ivector("idx")
    z = ps.get_item_list(x, idx)
    fn = function([x, idx], z, mode="NUMBA")

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)
    idx_test = np.asarray([0, 6], dtype=np.int32)

    with pytest.raises(IndexError):
        fn(x_test, idx_test)


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_get_item_list_grad(format):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    idx = pt.ivector("idx")
    gz = ps.matrix(format, name="gz", shape=(4, 5), dtype=config.floatX)
    z = ps.get_item_list_grad(x, idx, gz)

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)
    gz_test = sp.sparse.random(4, 5, density=0.4, format=format, dtype=config.floatX)
    idx_test = np.asarray([0, 2, 5, 2], dtype=np.int32)

    with pytest.warns(sp.sparse.SparseEfficiencyWarning):
        # GetItemListGrad.perform does sparse row assignment into an initially empty sparse
        # matrix, which changes sparsity structure incrementally and triggers the warning.
        compare_numba_and_py_sparse([x, idx, gz], z, [x_test, idx_test, gz_test])


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_get_item_list_grad_wrong_index(format):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    idx = pt.ivector("idx")
    gz = ps.matrix(format, name="gz", shape=(2, 5), dtype=config.floatX)
    z = ps.get_item_list_grad(x, idx, gz)
    fn = function([x, idx, gz], z, mode="NUMBA")

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)
    gz_test = sp.sparse.random(2, 5, density=0.4, format=format, dtype=config.floatX)
    idx_test = np.asarray([0, 6], dtype=np.int32)

    with pytest.raises(IndexError):
        fn(x_test, idx_test, gz_test)


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_get_item_2lists(format):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    ind1 = pt.ivector("ind1")
    ind2 = pt.ivector("ind2")
    z = ps.get_item_2lists(x, ind1, ind2)

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)
    ind1_test = np.asarray([0, 0, 3, 5], dtype=np.int32)
    ind2_test = np.asarray([0, 4, 2, 1], dtype=np.int32)

    compare_numba_and_py_sparse([x, ind1, ind2], z, [x_test, ind1_test, ind2_test])


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_get_item_2d(format):
    x = ps.matrix(format, name="x", shape=(100, 97), dtype=config.floatX)
    a = pt.iscalar("a")
    b = pt.iscalar("b")
    c = pt.iscalar("c")
    d = pt.iscalar("d")
    e = pt.iscalar("e")
    f = pt.iscalar("f")

    z1 = x[a:b:e, c:d:f]
    z2 = x[a:b:e]
    z3 = x[:a, :b]
    z4 = x[:, a:]
    z5 = x[1:10:2, 10:20:3]
    z6 = x[10:1:-2, 15:2:-3]

    x_test = sp.sparse.random(100, 97, density=0.4, format=format, dtype=config.floatX)

    compare_numba_and_py_sparse(
        [x, a, b, c, d, e, f],
        [z1, z2, z3, z4],
        [x_test, 1, 5, 10, 15, 2, 3],
    )
    compare_numba_and_py_sparse([x], z5, [x_test])
    compare_numba_and_py_sparse([x], z6, [x_test])


@pytest.mark.parametrize("format", ("csr", "csc"))
@pytest.mark.parametrize(
    ("ind1_test", "ind2_test"),
    [
        (np.asarray([0, 6], dtype=np.int32), np.asarray([0, 3], dtype=np.int32)),
        (np.asarray([0, 3], dtype=np.int32), np.asarray([0, 5], dtype=np.int32)),
    ],
)
def test_sparse_get_item_2lists_wrong_index(format, ind1_test, ind2_test):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    ind1 = pt.ivector("ind1")
    ind2 = pt.ivector("ind2")
    z = ps.get_item_2lists(x, ind1, ind2)
    fn = function([x, ind1, ind2], z, mode="NUMBA")

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)

    with pytest.raises(IndexError):
        fn(x_test, ind1_test, ind2_test)


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_get_item_2lists_grad(format):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    ind1 = pt.ivector("ind1")
    ind2 = pt.ivector("ind2")
    gz = pt.vector(name="gz", shape=(4,), dtype=config.floatX)
    z = ps.get_item_2lists_grad(x, ind1, ind2, gz)

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)
    ind1_test = np.asarray([0, 2, 5, 2], dtype=np.int32)
    ind2_test = np.asarray([1, 0, 4, 0], dtype=np.int32)
    gz_test = np.asarray([0.5, -1.25, 2.0, 4.5], dtype=config.floatX)

    with pytest.warns(sp.sparse.SparseEfficiencyWarning):
        # GetItem2ListsGrad.perform does sparse item assignment into an initially empty
        # sparse matrix, which changes sparsity structure incrementally.
        compare_numba_and_py_sparse(
            [x, ind1, ind2, gz], z, [x_test, ind1_test, ind2_test, gz_test]
        )


@pytest.mark.parametrize("format", ("csr", "csc"))
@pytest.mark.parametrize(
    ("ind1_test", "ind2_test"),
    [
        (np.asarray([0, 6], dtype=np.int32), np.asarray([0, 3], dtype=np.int32)),
        (np.asarray([0, 3], dtype=np.int32), np.asarray([0, 5], dtype=np.int32)),
    ],
)
def test_sparse_get_item_2lists_grad_wrong_index(format, ind1_test, ind2_test):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    ind1 = pt.ivector("ind1")
    ind2 = pt.ivector("ind2")
    gz = pt.vector(name="gz", shape=(2,), dtype=config.floatX)
    z = ps.get_item_2lists_grad(x, ind1, ind2, gz)
    fn = function([x, ind1, ind2, gz], z, mode="NUMBA")

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)
    gz_test = np.asarray([1.0, -2.0], dtype=config.floatX)

    with pytest.raises(IndexError):
        fn(x_test, ind1_test, ind2_test, gz_test)


@pytest.mark.parametrize("format", ("csr", "csc"))
@pytest.mark.parametrize(("row_idx", "col_idx"), [(3, 2), (-1, -2)])
def test_sparse_get_item_scalar(format, row_idx, col_idx):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    row = pt.iscalar("row")
    col = pt.iscalar("col")
    z_var = x[row, col]
    z_lit = x[3, 2]
    z_lit_neg = x[-1, -2]

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)

    compare_numba_and_py_sparse([x, row, col], z_var, [x_test, row_idx, col_idx])
    compare_numba_and_py_sparse([x], z_lit, [x_test])
    compare_numba_and_py_sparse([x], z_lit_neg, [x_test])


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_get_item_scalar_wrong_index(format):
    x = ps.matrix(format, name="x", shape=(6, 5), dtype=config.floatX)
    row = pt.iscalar("row")
    col = pt.iscalar("col")
    z = x[row, col]
    fn = function([x, row, col], z, mode="NUMBA")

    x_test = sp.sparse.random(6, 5, density=0.4, format=format, dtype=config.floatX)

    with pytest.raises(IndexError, match="row index out of bounds"):
        fn(x_test, 6, 0)

    with pytest.raises(IndexError, match="column index out of bounds"):
        fn(x_test, 0, 5)


@pytest.mark.parametrize("format", ("csr", "csc"))
def test_sparse_neg(format):
    x = ps.matrix(format, name="x", shape=(7, 6), dtype=config.floatX)
    z = -x

    x_test = sp.sparse.random(7, 6, density=0.4, format=format, dtype=config.floatX)

    compare_numba_and_py_sparse([x], z, [x_test])
