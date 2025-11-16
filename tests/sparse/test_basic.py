import numpy as np
import pytest
import scipy.sparse as scipy_sparse
from packaging import version
from scipy import __version__ as scipy_version

import pytensor
import pytensor.sparse.math
import pytensor.tensor as pt
from pytensor import sparse
from pytensor.compile.function import function
from pytensor.compile.io import In
from pytensor.configdefaults import config
from pytensor.gradient import GradientError
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.sparse.basic import (
    CSC,
    CSM,
    CSR,
    Cast,
    ConstructSparseFromList,
    CSMGrad,
    CSMProperties,
    DenseFromSparse,
    Diag,
    EnsureSortedIndices,
    GetItemScalar,
    HStack,
    Neg,
    Remove0,
    SparseFromDense,
    SparseTensorType,
    SquareDiagonal,
    Transpose,
    VStack,
    _is_sparse,
    _is_sparse_variable,
    _mtypes,
    all_dtypes,
    as_sparse_or_tensor_variable,
    as_sparse_variable,
    cast,
    clean,
    construct_sparse_from_list,
    csc_from_dense,
    csm_properties,
    csr_from_dense,
    dense_from_sparse,
    diag,
    ensure_sorted_indices,
    sp_ones_like,
    sparse_formats,
    square_diagonal,
    transpose,
)
from pytensor.sparse.rewriting import (
    CSMGradC,
)
from pytensor.sparse.variable import SparseConstant
from pytensor.tensor.basic import MakeVector
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.shape import Shape_i
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor1,
)
from pytensor.tensor.type import (
    TensorType,
    float_dtypes,
    iscalar,
    ivector,
    lvector,
    matrix,
    tensor,
    vector,
)
from tests import unittest_tools as utt


def as_sparse_format(data, format):
    if format == "csc":
        return scipy_sparse.csc_matrix(data)
    elif format == "csr":
        return scipy_sparse.csr_matrix(data)
    else:
        raise NotImplementedError()


def eval_outputs(outputs):
    return function([], outputs)()[0]


# scipy 0.17 will return sparse values in all cases while previous
# version sometimes wouldn't.  This will make everything dense so that
# we can use assert_allclose.
def as_ndarray(val):
    if hasattr(val, "toarray"):
        return val.toarray()
    return val


def random_lil(shape, dtype, nnz):
    rval = scipy_sparse.lil_matrix(shape, dtype=dtype)
    huge = 2**30
    for k in range(nnz):
        # set non-zeros in random locations (row x, col y)
        idx = np.random.default_rng().integers(1, huge + 1, size=2) % shape
        value = np.random.random()
        # if dtype *int*, value will always be zeros!
        if dtype in sparse.integer_dtypes:
            value = int(value * 100)
        # The call to tuple is needed as scipy 0.13.1 do not support
        # ndarray with length 2 as idx tuple.
        rval.__setitem__(tuple(idx), value)
    return rval


def sparse_random_inputs(
    format,
    shape,
    n=1,
    out_dtype=None,
    p=0.5,
    gap=None,
    explicit_zero=False,
    unsorted_indices=False,
):
    """
    Return a tuple containing everything needed to perform a test.

    If `out_dtype` is `None`, pytensor.config.floatX is used.

    :param format: Sparse format.
    :param shape: Shape of data.
    :param n: Number of variable.
    :param out_dtype: dtype of output.
    :param p: Sparsity proportion.
    :param gap: Tuple for the range of the random sample. When
                length is 1, it is assumed to be the exclusive
                max, when `gap` = (`a`, `b`) it provide a sample
                from [a, b[. If `None` is used, it provide [0, 1]
                for float dtypes and [0, 50[ for integer dtypes.
    :param explicit_zero: When True, we add explicit zero in the
                          returned sparse matrix
    :param unsorted_indices: when True, we make sure there is
                             unsorted indices in the returned
                             sparse matrix.
    :return: (variable, data) where both `variable` and `data` are list.

    :note: explicit_zero and unsorted_indices was added in PyTensor 0.6rc4
    """

    if out_dtype is None:
        out_dtype = pytensor.config.floatX

    assert 0 <= p <= 1
    assert len(shape) == 2
    assert out_dtype in sparse.all_dtypes
    assert gap is None or isinstance(gap, tuple | list)
    if gap is not None and out_dtype.startswith("u"):
        assert gap[0] >= 0

    def _rand():
        where = np.random.binomial(1, p, size=shape).astype("int8")

        if out_dtype in sparse.discrete_dtypes:
            if not gap:
                value = np.random.default_rng().integers(50, size=shape)
            elif len(gap) == 2:
                value = np.random.default_rng().integers(gap[0], gap[1], size=shape)
            else:
                value = np.random.default_rng().integers(gap[0], size=shape)
        else:
            if not gap:
                value = np.random.random(shape)
            elif len(gap) == 2:
                a, b = gap
                value = a + np.random.random(shape) * (b - a)
            else:
                value = np.random.random(shape) * gap[0]
        return (where * value).astype(out_dtype)

    variable = [
        getattr(pytensor.sparse, format + "_matrix")(dtype=out_dtype) for k in range(n)
    ]
    data = [
        getattr(scipy_sparse, format + "_matrix")(_rand(), dtype=out_dtype)
        for k in range(n)
    ]
    if unsorted_indices:
        for idx in range(n):
            d = data[idx]
            # these flip the matrix, but it's random anyway
            if format == "csr":
                d = scipy_sparse.csr_matrix(
                    (d.data, d.shape[1] - 1 - d.indices, d.indptr), shape=d.shape
                )
            if format == "csc":
                d = scipy_sparse.csc_matrix(
                    (d.data, d.shape[0] - 1 - d.indices, d.indptr), shape=d.shape
                )
            assert not d.has_sorted_indices
            data[idx] = d
    if explicit_zero:
        for idx in range(n):
            assert data[idx].nnz > 1, "can't make a sparse matrix with explicit 0"
            d_idx = np.random.default_rng().integers(data[idx].nnz)
            data[idx].data[d_idx] = 0

    # numpy 1.5.0 with scipy 0.9.0 have scipy_sparse.XXX_matrix return
    # typenum 10(ulonglong) instead of 8(uint64) event if they are the same!
    # PyTensor don't like ulonglong type_num
    dtype = np.dtype(out_dtype)  # Convert into dtype object.
    if data[0].dtype.num != dtype.num and dtype.str == data[0].dtype.str:
        data[0].data = np.asarray(data[0].data, out_dtype)
    assert data[0].dtype.num == dtype.num
    return (variable, data)


def verify_grad_sparse(op, pt, structured=False, *args, **kwargs):
    """
    Wrapper for pytensor.test.unittest_tools.py:verify_grad which
    converts sparse variables back and forth.

    Parameters
    ----------
    op
        Op to check.
    pt
        List of inputs to realize the tests.
    structured
        True to tests with a structured grad, False otherwise.
    args
        Other `verify_grad` parameters if any.
    kwargs
        Other `verify_grad` keywords if any.

    Returns
    -------
    None
    """

    def conv_none(x):
        return x

    def conv_csr(ind, indptr, shp):
        def f(spdata):
            return CSR(spdata, ind, indptr, shp)

        return f

    def conv_csc(ind, indptr, shp):
        def f(spdata):
            return CSC(spdata, ind, indptr, shp)

        return f

    iconv = []
    dpt = []

    for p in pt:
        if _is_sparse(p):
            if structured:
                dpt.append(p.data)
            else:
                dpt.append(p.toarray())
            if p.format == "csr":
                if structured:
                    iconv.append(conv_csr(p.indices[: p.size], p.indptr, p.shape))
                else:
                    iconv.append(csr_from_dense)
            elif p.format == "csc":
                if structured:
                    iconv.append(conv_csc(p.indices[: p.size], p.indptr, p.shape))
                else:
                    iconv.append(csc_from_dense)
            else:
                raise NotImplementedError(f"No conv for {p.format}")
        else:
            dpt.append(p)
            iconv.append(conv_none)
    output = op(*[as_sparse_or_tensor_variable(p) for p in pt])
    if isinstance(output, list | tuple):
        raise NotImplementedError("verify_grad can't deal with multiple outputs")
    if _is_sparse_variable(output):
        oconv = DenseFromSparse(structured=structured)
    else:
        oconv = conv_none

    def conv_op(*inputs):
        ipt = [conv(i) for i, conv in zip(inputs, iconv, strict=True)]
        out = op(*ipt)
        return oconv(out)

    return utt.verify_grad(conv_op, dpt, *args, **kwargs)


class TestVerifyGradSparse:
    class FailOp(Op):
        def __init__(self, structured):
            self.structured = structured

        def __eq__(self, other):
            return (type(self) is type(other)) and self.structured == other.structured

        def __hash__(self):
            return hash(type(self)) ^ hash(self.structured)

        def make_node(self, x):
            x = as_sparse_variable(x)
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, outputs):
            (x,) = inputs
            (out,) = outputs
            assert _is_sparse(x)
            out[0] = -x

        def grad(self, inputs, gout):
            (x,) = inputs
            (gz,) = gout
            assert _is_sparse_variable(x) and _is_sparse_variable(gz)
            if self.structured:
                return (sp_ones_like(x) * dense_from_sparse(gz),)
            else:
                return (gz,)

        def infer_shape(self, fgraph, node, shapes):
            return [shapes[0]]

    def test_grad_fail(self):
        with pytest.raises(GradientError):
            verify_grad_sparse(
                self.FailOp(structured=False),
                [scipy_sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            )

        with pytest.raises(GradientError):
            verify_grad_sparse(
                self.FailOp(structured=True),
                [scipy_sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            )


class TestTranspose:
    def test_transpose_csc(self):
        spe = scipy_sparse.csc_matrix(scipy_sparse.eye(5, 3))
        a = as_sparse_variable(spe)
        assert a.data is not spe
        assert a.data.shape == (5, 3)
        assert a.type.dtype == "float64", a.type.dtype
        assert a.type.format == "csc", a.type.format
        ta = transpose(a)
        assert ta.type.dtype == "float64", ta.type.dtype
        assert ta.type.format == "csr", ta.type.format

        vta = eval_outputs([ta])
        assert vta.shape == (3, 5)

    def test_transpose_csr(self):
        a = as_sparse_variable(scipy_sparse.csr_matrix(scipy_sparse.eye(5, 3)))
        assert a.data.shape == (5, 3)
        assert a.type.dtype == "float64"
        assert a.type.format == "csr"
        ta = transpose(a)
        assert ta.type.dtype == "float64", ta.type.dtype
        assert ta.type.format == "csc", ta.type.format

        vta = eval_outputs([ta])
        assert vta.shape == (3, 5)


class TestSparseInferShape(utt.InferShapeTester):
    @pytest.mark.skip(reason="infer_shape not implemented for GetItem2d yet")
    def test_getitem_2d(self):
        pass

    def test_getitem_scalar(self):
        x = SparseTensorType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [x[2, 2]],
            [scipy_sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            GetItemScalar,
        )

    def test_csm(self):
        for sparsetype in ("csr", "csc"):
            x = vector()
            y = ivector()
            z = ivector()
            s = ivector()
            call = getattr(scipy_sparse, sparsetype + "_matrix")
            spm = call(random_lil((300, 400), config.floatX, 5))
            out = CSM(sparsetype)(x, y, z, s)
            self._compile_and_check(
                [x, y, z, s], [out], [spm.data, spm.indices, spm.indptr, spm.shape], CSM
            )

    def test_csm_grad(self):
        for sparsetype in ("csr", "csc"):
            x = vector()
            y = ivector()
            z = ivector()
            s = ivector()
            call = getattr(scipy_sparse, sparsetype + "_matrix")
            spm = call(random_lil((300, 400), config.floatX, 5))
            out = pytensor.grad(dense_from_sparse(CSM(sparsetype)(x, y, z, s)).sum(), x)
            self._compile_and_check(
                [x, y, z, s],
                [out],
                [spm.data, spm.indices, spm.indptr, spm.shape],
                (CSMGrad, CSMGradC),
            )

    def test_transpose(self):
        x = SparseTensorType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [x.T],
            [scipy_sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            Transpose,
        )

    def test_neg(self):
        x = SparseTensorType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [-x],
            [scipy_sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            Neg,
        )

    def test_remove0(self):
        x = SparseTensorType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [Remove0()(x)],
            [scipy_sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            Remove0,
        )

    def test_dense_from_sparse(self):
        x = SparseTensorType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [dense_from_sparse(x)],
            [scipy_sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            dense_from_sparse.__class__,
        )

    def test_sparse_from_dense(self):
        x = matrix()
        self._compile_and_check(
            [x],
            [csc_from_dense(x)],
            [np.random.standard_normal((10, 40)).astype(config.floatX)],
            csc_from_dense.__class__,
        )

    def test_sparse_from_list(self):
        x = matrix("x")
        vals = matrix("vals")
        ilist = lvector("ilist")

        out = construct_sparse_from_list(x, vals, ilist)
        self._compile_and_check(
            [x, vals, ilist],
            [out],
            [
                np.zeros((40, 10), dtype=config.floatX),
                np.random.standard_normal((12, 10)).astype(config.floatX),
                np.random.default_rng().integers(low=0, high=40, size=(12,)),
            ],
            ConstructSparseFromList,
        )


class TestConstructSparseFromList:
    def test_adv_sub1_sparse_grad(self):
        v = ivector()

        m = matrix()

        with pytest.raises(TypeError):
            pytensor.sparse.sparse_grad(v)

        with pytest.raises(TypeError):
            sub = m[v, v]
            pytensor.sparse.sparse_grad(sub)

        # Assert we don't create a sparse grad by default
        sub = m[v]
        g = pytensor.grad(sub.sum(), m)
        assert isinstance(g.owner.op, AdvancedIncSubtensor)

        # Test that we create a sparse grad when asked
        # USER INTERFACE
        m = matrix()
        v = ivector()
        sub = pytensor.sparse.sparse_grad(m[v])
        g = pytensor.grad(sub.sum(), m)
        assert isinstance(g.owner.op, ConstructSparseFromList)

        # Test that we create a sparse grad when asked
        # Op INTERFACE
        m = matrix()
        v = ivector()
        sub = AdvancedSubtensor1(sparse_grad=True)(m, v)
        g = pytensor.grad(sub.sum(), m)
        assert isinstance(g.owner.op, ConstructSparseFromList)

        # Test the sparse grad
        valm = np.random.random((5, 4)).astype(config.floatX)
        valv = np.random.default_rng().integers(0, 5, 10)
        m = matrix()
        shared_v = pytensor.shared(valv)

        def fn(m):
            return pytensor.sparse.sparse_grad(m[shared_v])

        verify_grad_sparse(fn, [valm])

    def test_err(self):
        for ndim in [1, 3]:
            t = TensorType(dtype=config.floatX, shape=(None,) * ndim)()
            v = ivector()
            sub = t[v]

            # Assert we don't create a sparse grad by default
            g = pytensor.grad(sub.sum(), t)
            assert isinstance(g.owner.op, AdvancedIncSubtensor)

            # Test that we raise an error, as we can't create a sparse
            # grad from tensors that don't have 2 dimensions.
            sub = pytensor.sparse.sparse_grad(sub)
            with pytest.raises(TypeError):
                pytensor.grad(sub.sum(), t)


class TestConversion:
    def test_basic(self):
        test_val = np.random.random((5,)).astype(config.floatX)
        a = pt.as_tensor_variable(test_val)
        s = csc_from_dense(a)
        val = eval_outputs([s])
        assert str(val.dtype) == config.floatX
        assert val.format == "csc"

        a = pt.as_tensor_variable(test_val)
        s = csr_from_dense(a)
        val = eval_outputs([s])
        assert str(val.dtype) == config.floatX
        assert val.format == "csr"

        test_val = np.eye(3).astype(config.floatX)
        a = scipy_sparse.csr_matrix(test_val)
        s = as_sparse_or_tensor_variable(a)
        res = pt.as_tensor_variable(s)
        assert isinstance(res, SparseConstant)

        a = scipy_sparse.csr_matrix(test_val)
        s = as_sparse_or_tensor_variable(a)
        from pytensor.tensor.exceptions import NotScalarConstantError

        with pytest.raises(NotScalarConstantError):
            pt.get_underlying_scalar_constant_value(s, only_process_constants=True)

    def test_dense_from_sparse(self):
        # call dense_from_sparse
        for t in _mtypes:
            s = t(scipy_sparse.identity(5))
            s = as_sparse_variable(s)
            d = dense_from_sparse(s)
            val = eval_outputs([d])
            assert str(val.dtype) == s.dtype
            assert np.all(val[0] == [1, 0, 0, 0, 0])

    def test_todense(self):
        # call sparse_var.todense()
        for t in _mtypes:
            s = t(scipy_sparse.identity(5))
            s = as_sparse_variable(s)
            d = s.toarray()
            val = eval_outputs([d])
            assert str(val.dtype) == s.dtype
            assert np.all(val[0] == [1, 0, 0, 0, 0])

    @staticmethod
    def check_format_ndim(format, ndim):
        x = tensor(dtype=config.floatX, shape=(None,) * ndim, name="x")

        s = SparseFromDense(format)(x)
        s_m = -s
        d = dense_from_sparse(s_m)
        c = d.sum()
        g = pytensor.grad(c, x)
        f = pytensor.function([x], [s, g])
        f(np.array(0, dtype=config.floatX, ndmin=ndim))
        f(np.array(7, dtype=config.floatX, ndmin=ndim))

    def test_format_ndim(self):
        for format in "csc", "csr":
            for ndim in 0, 1, 2:
                self.check_format_ndim(format, ndim)

            with pytest.raises(TypeError):
                self.check_format_ndim(format, 3)
            with pytest.raises(TypeError):
                self.check_format_ndim(format, 4)


class TestCsmProperties:
    def test_csm_properties_grad(self):
        sp_types = {"csc": scipy_sparse.csc_matrix, "csr": scipy_sparse.csr_matrix}

        for format in ("csc", "csr"):
            for dtype in ("float32", "float64"):
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                verify_grad_sparse(
                    lambda *x: CSMProperties()(*x)[0], [spmat], structured=True
                )

                verify_grad_sparse(
                    lambda *x: CSMProperties()(*x)[1], [spmat], structured=True
                )

                verify_grad_sparse(
                    lambda *x: CSMProperties()(*x)[2], [spmat], structured=True
                )

                verify_grad_sparse(
                    lambda *x: CSMProperties()(*x)[2], [spmat], structured=True
                )

    def test_csm_properties(self):
        sp_types = {"csc": scipy_sparse.csc_matrix, "csr": scipy_sparse.csr_matrix}

        for format in ("csc", "csr"):
            for dtype in ("float32", "float64"):
                x = SparseTensorType(format, dtype=dtype)()
                f = pytensor.function([x], csm_properties(x))

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                data, indices, indptr, shape = f(spmat)

                assert np.all(data == spmat.data)
                assert np.all(indices == spmat.indices)
                assert np.all(indptr == spmat.indptr)
                assert np.all(shape == spmat.shape)


class TestCsm:
    def test_csm_grad(self):
        sp_types = {"csc": scipy_sparse.csc_matrix, "csr": scipy_sparse.csr_matrix}

        for format in ("csc", "csr"):
            for dtype in ("float32", "float64"):
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                verify_grad_sparse(
                    lambda x: CSM(format)(
                        x, spmat.indices, spmat.indptr, np.asarray(spmat.shape, "int32")
                    ),
                    [spmat.data],
                    structured=True,
                )

    @pytest.mark.skipif(
        version.parse(scipy_version) >= version.parse("1.16.0"),
        reason="Scipy 1.16 introduced some changes that make this test fail",
    )
    def test_csm_sparser(self):
        # Test support for gradients sparser than the input.

        sp_types = {"csc": scipy_sparse.csc_matrix, "csr": scipy_sparse.csr_matrix}

        for format in ("csc", "csr"):
            for dtype in ("float32", "float64"):
                x = tensor(dtype=dtype, shape=(None,))
                y = ivector()
                z = ivector()
                s = ivector()

                a = as_sparse_variable(sp_types[format](random_lil((4, 3), dtype, 1)))

                f = pytensor.function(
                    [x, y, z, s],
                    pytensor.grad(
                        dense_from_sparse(a * CSM(format)(x, y, z, s)).sum(), x
                    ),
                )

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                res = f(
                    spmat.data,
                    spmat.indices,
                    spmat.indptr,
                    np.asarray(spmat.shape, "int32"),
                )

                assert len(spmat.data) == len(res)

    @pytest.mark.skipif(
        version.parse(scipy_version) >= version.parse("1.16.0"),
        reason="Scipy 1.16 introduced some changes that make this test fail",
    )
    def test_csm_unsorted(self):
        # Test support for gradients of unsorted inputs.

        for format in [
            "csr",
            "csc",
        ]:
            for dtype in ("float32", "float64"):
                # Sparse advanced indexing produces unsorted sparse matrices
                a = sparse_random_inputs(
                    format, (8, 6), out_dtype=dtype, unsorted_indices=True
                )[1][0]
                # Make sure it's unsorted
                assert not a.has_sorted_indices

                def my_op(x):
                    y = pt.constant(a.indices)
                    z = pt.constant(a.indptr)
                    s = pt.constant(a.shape)
                    return pt_sum(dense_from_sparse(CSM(format)(x, y, z, s) * a))

                verify_grad_sparse(my_op, [a.data])

    def test_csm(self):
        sp_types = {"csc": scipy_sparse.csc_matrix, "csr": scipy_sparse.csr_matrix}

        for format in ("csc", "csr"):
            for dtype in ("float32", "float64"):
                x = tensor(dtype=dtype, shape=(None,))
                y = ivector()
                z = ivector()
                s = ivector()
                f = pytensor.function([x, y, z, s], CSM(format)(x, y, z, s))

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                res = f(
                    spmat.data,
                    spmat.indices,
                    spmat.indptr,
                    np.asarray(spmat.shape, "int32"),
                )

                assert np.all(res.data == spmat.data)
                assert np.all(res.indices == spmat.indices)
                assert np.all(res.indptr == spmat.indptr)
                assert np.all(res.shape == spmat.shape)


class TestZerosLike:
    def test(self):
        x = sparse.csr_matrix()
        f = pytensor.function([x], sparse.sp_zeros_like(x))
        vx = scipy_sparse.csr_matrix(
            np.asarray(
                np.random.binomial(1, 0.5, (100, 100)), dtype=pytensor.config.floatX
            )
        )

        fx = f(vx)

        assert fx.nnz == 0
        assert fx.shape == vx.shape


def test_shape_i():
    sparse_dtype = "float32"

    a = SparseTensorType("csr", dtype=sparse_dtype)()
    f = pytensor.function([a], a.shape[1])
    assert f(scipy_sparse.csr_matrix(random_lil((100, 10), sparse_dtype, 3))) == 10


def test_shape():
    # Test that getting the shape of a sparse variable
    # does not actually create a dense tensor in the process.
    sparse_dtype = "float32"

    a = SparseTensorType("csr", dtype=sparse_dtype)()
    f = pytensor.function([a], a.shape)
    assert np.all(
        f(scipy_sparse.csr_matrix(random_lil((100, 10), sparse_dtype, 3))) == (100, 10)
    )
    if pytensor.config.mode != "FAST_COMPILE":
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 3
        assert isinstance(topo[0].op, Shape_i)
        assert isinstance(topo[1].op, Shape_i)
        assert isinstance(topo[2].op, MakeVector)


def test_may_share_memory():
    a = scipy_sparse.csc_matrix(scipy_sparse.eye(5, 3))
    b = scipy_sparse.csc_matrix(scipy_sparse.eye(4, 3))

    def as_ar(a):
        return np.asarray(a, dtype="int32")

    for a_, b_, rep in [
        (a, a, True),
        (b, b, True),
        (a, b, False),
        (a, a.data, True),
        (a, a.indptr, True),
        (a, a.indices, True),
        (a, as_ar(a.shape), False),
        (a.data, a, True),
        (a.indptr, a, True),
        (a.indices, a, True),
        (as_ar(a.shape), a, False),
        (b, b.data, True),
        (b, b.indptr, True),
        (b, b.indices, True),
        (b, as_ar(b.shape), False),
        (b.data, b, True),
        (b.indptr, b, True),
        (b.indices, b, True),
        (as_ar(b.shape), b, False),
        (b.data, a, False),
        (b.indptr, a, False),
        (b.indices, a, False),
        (as_ar(b.shape), a, False),
        (a.transpose(), a, True),
        (b.transpose(), b, True),
        (a.transpose(), b, False),
        (b.transpose(), a, False),
    ]:
        assert SparseTensorType.may_share_memory(a_, b_) == rep


def test_sparse_shared_memory():
    # Note : There are no inplace ops on sparse matrix yet. If one is
    # someday implemented, we could test it here.
    a = random_lil((3, 4), "float32", 3).tocsr()
    m1 = random_lil((4, 4), "float32", 3).tocsr()
    m2 = random_lil((4, 4), "float32", 3).tocsr()
    x = SparseTensorType("csr", dtype="float32")()
    y = SparseTensorType("csr", dtype="float32")()

    sdot = sparse.math.structured_dot
    z = sdot(x * 3, m1) + sdot(y * 2, m2)

    f = pytensor.function(
        [In(x, mutable=True), In(y, mutable=True)], z, mode="FAST_RUN"
    )

    def f_(x, y, m1=m1, m2=m2):
        return ((x * 3) * m1) + ((y * 2) * m2)

    assert SparseTensorType.may_share_memory(a, a)  # This is trivial
    result = f(a, a)
    result_ = f_(a, a)
    assert (result_.todense() == result.todense()).all()


def test_size():
    # Ensure the `size` attribute of sparse matrices behaves as in numpy.

    for sparse_type in ("csc_matrix", "csr_matrix"):
        x = getattr(pytensor.sparse, sparse_type)()
        y = getattr(scipy_sparse, sparse_type)((5, 7)).astype(config.floatX)
        get_size = pytensor.function([x], x.size)

        def check():
            assert y.size == get_size(y)

        # We verify that the size is correctly updated as we store more data
        # into the sparse matrix (including zeros).
        check()
        y[0, 0] = 1
        check()
        y[0, 1] = 0
        check()


class TestColScaleCSC(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op = sparse.col_scale

    def test_op(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(10).astype(config.floatX))

            f = pytensor.function(variable, self.op(*variable))

            tested = f(*data)
            x, s = data[0].toarray(), data[1][np.newaxis, :]
            expected = x * s

            assert tested.format == format
            utt.assert_allclose(expected, tested.toarray())

    def test_infer_shape(self):
        for format, cls in [("csc", sparse.ColScaleCSC), ("csr", sparse.RowScaleCSC)]:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(10).astype(config.floatX))

            self._compile_and_check(variable, [self.op(*variable)], data, cls)

    def test_grad(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(10).astype(config.floatX))

            verify_grad_sparse(self.op, data, structured=True)


class TestRowScaleCSC(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op = sparse.row_scale

    def test_op(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(8).astype(config.floatX))

            f = pytensor.function(variable, self.op(*variable))

            tested = f(*data)
            x, s = data[0].toarray(), data[1][:, np.newaxis]
            expected = x * s

            assert tested.format == format
            utt.assert_allclose(expected, tested.toarray())

    def test_infer_shape(self):
        for format, cls in [("csc", sparse.RowScaleCSC), ("csr", sparse.ColScaleCSC)]:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(8).astype(config.floatX))

            self._compile_and_check(variable, [self.op(*variable)], data, cls)

    def test_grad(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(8).astype(config.floatX))

            verify_grad_sparse(self.op, data, structured=True)


class TestDiag(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = Diag
        self.op = diag

    def test_op(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(10, 10))

            z = self.op(*variable)
            assert z.type.broadcastable == (False,)

            f = pytensor.function(variable, z)
            tested = f(*data)
            expected = data[0].toarray().diagonal()

            utt.assert_allclose(expected, tested)

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(10, 10))
            self._compile_and_check(
                variable, [self.op(*variable)], data, self.op_class, warn=False
            )

    def test_grad(self):
        for format in sparse.sparse_formats:
            _variable, data = sparse_random_inputs(format, shape=(10, 10))
            verify_grad_sparse(self.op, data, structured=False)


class TestSquareDiagonal(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = SquareDiagonal
        self.op = square_diagonal

    def test_op(self):
        for format in sparse.sparse_formats:
            for size in range(5, 9):
                variable = [vector()]
                data = [np.random.random(size).astype(config.floatX)]

                f = pytensor.function(variable, self.op(*variable))
                tested = f(*data).toarray()

                expected = np.diag(*data)
                utt.assert_allclose(expected, tested)
                assert tested.dtype == expected.dtype
                assert tested.shape == expected.shape

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            for size in range(5, 9):
                variable = [vector()]
                data = [np.random.random(size).astype(config.floatX)]

                self._compile_and_check(
                    variable, [self.op(*variable)], data, self.op_class
                )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for size in range(5, 9):
                data = [np.random.random(size).astype(config.floatX)]

                verify_grad_sparse(self.op, data, structured=False)


class TestEnsureSortedIndices(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = EnsureSortedIndices
        self.op = ensure_sorted_indices

    def test_op(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1], strict=True):
                variable, data = sparse_random_inputs(format, shape=shape)

                f = pytensor.function(variable, self.op(*variable))
                tested = f(*data).toarray()
                expected = data[0].sorted_indices().toarray()

                utt.assert_allclose(expected, tested)

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1], strict=True):
                variable, data = sparse_random_inputs(format, shape=shape)
                self._compile_and_check(
                    variable, [self.op(*variable)], data, self.op_class
                )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1], strict=True):
                _variable, data = sparse_random_inputs(format, shape=shape)
                verify_grad_sparse(self.op, data, structured=False)


class TestClean(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op = clean

    def test_op(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1], strict=True):
                variable, data = sparse_random_inputs(format, shape=shape)

                data[0][0, 0] = data[0][1, 1] = 0

                f = pytensor.function(variable, self.op(*variable))
                tested = f(*data)
                expected = data[0]
                expected.eliminate_zeros()

                assert all(tested.data == expected.data)
                assert not all(tested.data == 0)

                tested = tested.toarray()
                expected = expected.toarray()
                utt.assert_allclose(expected, tested)

    def test_grad(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1], strict=True):
                _variable, data = sparse_random_inputs(format, shape=shape)
                verify_grad_sparse(self.op, data, structured=False)


class TestRemove0(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = Remove0

    def test_remove0(self):
        configs = [
            # structure type, numpy matching class
            ("csc", scipy_sparse.csc_matrix),
            ("csr", scipy_sparse.csr_matrix),
        ]

        for format, matrix_class in configs:
            for zero, unsor in [
                (True, True),
                (True, False),
                (False, True),
                (False, False),
            ]:
                (x,), (mat,) = sparse_random_inputs(
                    format,
                    (6, 8),
                    out_dtype=config.floatX,
                    explicit_zero=zero,
                    unsorted_indices=unsor,
                )
                assert 0 in mat.data or not zero
                assert not (mat.has_sorted_indices and unsor)

                # the In thingy has to be there because pytensor has as rule not
                # to optimize inputs
                f = pytensor.function([In(x, borrow=True, mutable=True)], Remove0()(x))

                # assert optimization local_inplace_remove0 is applied in
                # modes with optimization
                if pytensor.config.mode not in ["FAST_COMPILE"]:
                    # list of apply nodes in the optimized graph.
                    nodes = f.maker.fgraph.toposort()
                    # Check there isn't any Remove0 instance not inplace.
                    assert not any(
                        isinstance(node.op, Remove0) and not node.op.inplace
                        for node in nodes
                    ), "Inplace optimization should have been applied"
                    # Check there is at least one Remove0 inplace.
                    assert any(
                        isinstance(node.op, Remove0) and node.op.inplace
                        for node in nodes
                    )
                # checking
                # makes sense to change its name
                target = mat
                result = f(mat)
                mat.eliminate_zeros()
                msg = "Matrices sizes differ. Have zeros been removed ?"
                assert result.size == target.size, msg
                if unsor:
                    assert not result.has_sorted_indices
                    assert not target.has_sorted_indices
                else:
                    assert result.has_sorted_indices
                    assert target.has_sorted_indices

    def test_infer_shape(self):
        mat = (np.arange(12) + 1).reshape((4, 3))
        mat[0, 1] = mat[1, 0] = mat[2, 2] = 0

        x_csc = sparse.csc_matrix(dtype=pytensor.config.floatX)
        mat_csc = scipy_sparse.csc_matrix(mat, dtype=pytensor.config.floatX)
        self._compile_and_check([x_csc], [Remove0()(x_csc)], [mat_csc], self.op_class)

        x_csr = sparse.csr_matrix(dtype=pytensor.config.floatX)
        mat_csr = scipy_sparse.csr_matrix(mat, dtype=pytensor.config.floatX)
        self._compile_and_check([x_csr], [Remove0()(x_csr)], [mat_csr], self.op_class)

    def test_grad(self):
        mat = (np.arange(9) + 1).reshape((3, 3))
        mat[0, 1] = mat[1, 0] = mat[2, 2] = 0

        mat_csc = scipy_sparse.csc_matrix(mat, dtype=pytensor.config.floatX)
        verify_grad_sparse(Remove0(), [mat_csc])

        mat_csr = scipy_sparse.csr_matrix(mat, dtype=pytensor.config.floatX)
        verify_grad_sparse(Remove0(), [mat_csr])


class TestGetItem:
    def setup_method(self):
        self.rng = np.random.default_rng(utt.fetch_seed())

    def test_GetItemList(self):
        a, A = sparse_random_inputs("csr", (4, 5))
        b, B = sparse_random_inputs("csc", (4, 5))
        y = a[0][[0, 1, 2, 3, 1]]
        z = b[0][[0, 1, 2, 3, 1]]

        fa = pytensor.function([a[0]], y)
        fb = pytensor.function([b[0]], z)

        t_geta = fa(A[0]).todense()
        t_getb = fb(B[0]).todense()

        s_geta = scipy_sparse.csr_matrix(A[0])[[0, 1, 2, 3, 1]].todense()
        s_getb = scipy_sparse.csc_matrix(B[0])[[0, 1, 2, 3, 1]].todense()

        utt.assert_allclose(t_geta, s_geta)
        utt.assert_allclose(t_getb, s_getb)

    def test_GetItemList_wrong_index(self):
        a, A = sparse_random_inputs("csr", (4, 5))
        y = a[0][[0, 4]]
        f = pytensor.function([a[0]], y)

        with pytest.raises(IndexError):
            f(A[0])

    def test_get_item_list_grad(self):
        op = sparse.basic.GetItemList()

        def op_with_fixed_index(x):
            return op(x, index=np.asarray([0, 1]))

        _x, x_val = sparse_random_inputs("csr", (4, 5))

        try:
            verify_grad_sparse(op_with_fixed_index, x_val)
        except NotImplementedError as e:
            assert "Scipy version is to old" in str(e)

    def test_GetItem2Lists(self):
        a, A = sparse_random_inputs("csr", (4, 5))
        b, B = sparse_random_inputs("csc", (4, 5))
        y = a[0][[0, 0, 1, 3], [0, 1, 2, 4]]
        z = b[0][[0, 0, 1, 3], [0, 1, 2, 4]]

        fa = pytensor.function([a[0]], y)
        fb = pytensor.function([b[0]], z)

        t_geta = fa(A[0])
        t_getb = fb(B[0])

        s_geta = np.asarray(scipy_sparse.csr_matrix(A[0])[[0, 0, 1, 3], [0, 1, 2, 4]])
        s_getb = np.asarray(scipy_sparse.csc_matrix(B[0])[[0, 0, 1, 3], [0, 1, 2, 4]])

        utt.assert_allclose(t_geta, s_geta)
        utt.assert_allclose(t_getb, s_getb)

    def test_GetItem2Lists_wrong_index(self):
        a, A = sparse_random_inputs("csr", (4, 5))
        y1 = a[0][[0, 5], [0, 3]]
        y2 = a[0][[0, 3], [0, 5]]

        f1 = pytensor.function([a[0]], y1)
        f2 = pytensor.function([a[0]], y2)

        with pytest.raises(IndexError):
            f1(A[0])
        with pytest.raises(IndexError):
            f2(A[0])

    def test_get_item_2lists_grad(self):
        op = sparse.basic.GetItem2Lists()

        def op_with_fixed_index(x):
            return op(x, ind1=np.asarray([0, 1]), ind2=np.asarray([2, 3]))

        _x, x_val = sparse_random_inputs("csr", (4, 5))

        verify_grad_sparse(op_with_fixed_index, x_val)

    def test_GetItem2D(self):
        sparse_formats = ("csc", "csr")
        for format in sparse_formats:
            x = sparse.matrix(format, name="x")
            a = iscalar("a")
            b = iscalar("b")
            c = iscalar("c")
            d = iscalar("d")
            e = iscalar("e")
            f = iscalar("f")

            # index
            m = 1
            n = 5
            p = 10
            q = 15
            j = 2
            k = 3

            vx = as_sparse_format(self.rng.binomial(1, 0.5, (100, 97)), format).astype(
                pytensor.config.floatX
            )

            # mode_no_debug = pytensor.compile.mode.get_default_mode()
            # if isinstance(mode_no_debug, pytensor.compile.debugmode.DebugMode):
            #    mode_no_debug = 'FAST_RUN'
            f1 = pytensor.function([x, a, b, c, d, e, f], x[a:b:e, c:d:f])
            r1 = f1(vx, m, n, p, q, j, k)
            t1 = vx[m:n:j, p:q:k]

            assert r1.shape == t1.shape
            assert np.all(t1.toarray() == r1.toarray())

            """
            Important: based on a discussion with both Fred and James
            The following indexing methods is not supported because the rval
            would be a sparse matrix rather than a sparse vector, which is a
            deviation from numpy indexing rule. This decision is made largely
            for keeping the consistency between numpy and pytensor.

            f2 = pytensor.function([x, a, b, c], x[a:b, c])
            r2 = f2(vx, m, n, p)
            t2 = vx[m:n, p]
            assert r2.shape == t2.shape
            assert np.all(t2.toarray() == r2.toarray())

            f3 = pytensor.function([x, a, b, c], x[a, b:c])
            r3 = f3(vx, m, n, p)
            t3 = vx[m, n:p]
            assert r3.shape == t3.shape
            assert np.all(t3.toarray() == r3.toarray())

            f5 = pytensor.function([x], x[1:2,3])
            r5 = f5(vx)
            t5 = vx[1:2, 3]
            assert r5.shape == t5.shape
            assert np.all(r5.toarray() == t5.toarray())

            f7 = pytensor.function([x], x[50])
            r7 = f7(vx)
            t7 = vx[50]
            assert r7.shape == t7.shape
            assert np.all(r7.toarray() == t7.toarray())
            """
            f4 = pytensor.function([x, a, b, e], x[a:b:e])
            r4 = f4(vx, m, n, j)
            t4 = vx[m:n:j]

            assert r4.shape == t4.shape
            assert np.all(t4.toarray() == r4.toarray())

            # -----------------------------------------------------------
            # test cases using int indexing instead of pytensor variable
            f6 = pytensor.function([x], x[1:10:j, 10:20:k])
            r6 = f6(vx)
            t6 = vx[1:10:j, 10:20:k]
            assert r6.shape == t6.shape
            assert np.all(r6.toarray() == t6.toarray())

            # ----------------------------------------------------------
            # test cases with indexing both with pytensor variable and int
            f8 = pytensor.function([x, a, b, e], x[a:b:e, 10:20:1])
            r8 = f8(vx, m, n, j)
            t8 = vx[m:n:j, 10:20:1]

            assert r8.shape == t8.shape
            assert np.all(r8.toarray() == t8.toarray())

            f9 = pytensor.function([x, a, b], x[1:a:j, 1:b:k])
            r9 = f9(vx, p, q)
            t9 = vx[1:p:j, 1:q:k]
            assert r9.shape == t9.shape
            assert np.all(r9.toarray() == t9.toarray())

            # -----------------------------------------------------------
            # Test mixing None and variables
            f10 = pytensor.function([x, a, b], x[:a, :b])
            r10 = f10(vx, p, q)
            t10 = vx[:p, :q]
            assert r10.shape == t10.shape
            assert np.all(r10.toarray() == t10.toarray())

            f11 = pytensor.function([x, a], x[:, a:])
            r11 = f11(vx, p)
            t11 = vx[:, p:]
            assert r11.shape == t11.shape
            assert np.all(r11.toarray() == t11.toarray())

            # Test that is work with shared variable
            sx = pytensor.shared(vx)
            f12 = pytensor.function([a], sx[:, a:])
            r12 = f12(p)
            t12 = vx[:, p:]
            assert r12.shape == t12.shape
            assert np.all(r12.toarray() == t12.toarray())

            # ------------------------------------------------------------
            # Invalid things
            # The syntax is a bit awkward because assertRaises forbids
            # the [] shortcut for getitem.
            # x[a:b] is not accepted because we don't have sparse vectors
            with pytest.raises(NotImplementedError):
                x.__getitem__((slice(a, b), c))

    def test_GetItemScalar(self):
        sparse_formats = ("csc", "csr")
        for format in sparse_formats:
            x = sparse.csc_matrix("x")
            a = iscalar()
            b = iscalar()

            m = 50
            n = 42

            vx = as_sparse_format(self.rng.binomial(1, 0.5, (97, 100)), format).astype(
                pytensor.config.floatX
            )

            f1 = pytensor.function([x, a, b], x[a, b])
            r1 = f1(vx, 10, 10)
            t1 = vx[10, 10]
            assert r1.shape == t1.shape
            assert np.all(t1 == r1)

            f2 = pytensor.function([x, a], x[50, a])
            r2 = f2(vx, m)
            t2 = vx[50, m]
            assert r2.shape == t2.shape
            assert np.all(t2 == r2)

            f3 = pytensor.function([x, a], x[a, 50])
            r3 = f3(vx, m)
            t3 = vx[m, 50]
            assert r3.shape == t3.shape
            assert np.all(t3 == r3)

            f4 = pytensor.function([x], x[50, 42])
            r4 = f4(vx)
            t4 = vx[m, n]
            assert r3.shape == t3.shape
            assert np.all(t4 == r4)

            # Test that is work with shared variable
            sx = pytensor.shared(vx)
            f1 = pytensor.function([a, b], sx[a, b])
            r1 = f1(10, 10)
            t1 = vx[10, 10]
            assert r1.shape == t1.shape
            assert np.all(t1 == r1)


class TestCasting(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()

    # slow but only test
    def test_cast(self):
        for format in sparse.sparse_formats:
            for i_dtype in sparse.all_dtypes:
                for o_dtype in sparse.all_dtypes:
                    (variable,), (data,) = sparse_random_inputs(
                        format, shape=(4, 7), out_dtype=i_dtype
                    )

                    func = pytensor.function([variable], cast(variable, o_dtype))
                    cls = pytensor.function([variable], Cast(o_dtype)(variable))
                    prop = pytensor.function([variable], variable.astype(o_dtype))

                    t_func, t_cls, t_prop = func(data), cls(data), prop(data)

                    expected = data.toarray().astype(o_dtype)

                    assert t_func.format == format
                    assert t_cls.format == format
                    assert t_prop.format == format

                    t_func = t_func.toarray()
                    t_cls = t_cls.toarray()
                    t_prop = t_prop.toarray()

                    utt.assert_allclose(expected, t_func)
                    utt.assert_allclose(expected, t_cls)
                    utt.assert_allclose(expected, t_prop)

    @pytest.mark.slow
    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            for i_dtype in sparse.all_dtypes:
                for o_dtype in sparse.all_dtypes:
                    variable, data = sparse_random_inputs(
                        format, shape=(4, 7), out_dtype=i_dtype
                    )
                    self._compile_and_check(
                        variable, [Cast(o_dtype)(*variable)], data, Cast
                    )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for i_dtype in sparse.float_dtypes:
                for o_dtype in float_dtypes:
                    if o_dtype == "float16":
                        # Don't test float16 output.
                        continue
                    _, data = sparse_random_inputs(
                        format, shape=(4, 7), out_dtype=i_dtype
                    )

                    eps = None
                    if o_dtype == "float32":
                        eps = 1e-2

                    verify_grad_sparse(Cast(o_dtype), data, eps=eps)


def _format_info(nb):
    x = {}
    mat = {}

    for format in sparse.sparse_formats:
        variable = getattr(pytensor.sparse, format + "_matrix")
        spa = getattr(scipy_sparse, format + "_matrix")

        x[format] = [variable() for t in range(nb)]
        mat[format] = [
            spa(random_lil((3, 4), pytensor.config.floatX, 8)) for t in range(nb)
        ]
    return x, mat


class _TestHVStack(utt.InferShapeTester):
    """
    Test for both HStack and VStack.
    """

    nb = 3  # Number of sparse matrix to stack
    x, mat = _format_info(nb)

    def test_op(self):
        for format in sparse_formats:
            for out_f in sparse_formats:
                for dtype in all_dtypes:
                    blocks = self.mat[format]

                    f = pytensor.function(
                        self.x[format],
                        self.op_class(format=out_f, dtype=dtype)(*self.x[format]),
                        allow_input_downcast=True,
                    )

                    tested = f(*blocks)
                    expected = self.expected_f(blocks, format=out_f, dtype=dtype)

                    utt.assert_allclose(expected.toarray(), tested.toarray())
                    assert tested.format == expected.format
                    assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            self._compile_and_check(
                self.x[format],
                [self.op_class(dtype="float64")(*self.x[format])],
                self.mat[format],
                self.op_class,
            )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for out_f in sparse.sparse_formats:
                for dtype in sparse.float_dtypes:
                    verify_grad_sparse(
                        self.op_class(format=out_f, dtype=dtype),
                        self.mat[format],
                        structured=False,
                        eps=1e-2,
                    )


def _hv_switch(op, expected_function):
    """
    Return the right test class for HStack or VStack.

    :Parameters:
    - `op`: HStack or VStack class.
    - `expected_function`: function from scipy for comparison.
    """

    class TestXStack(_TestHVStack):
        op_class = op

        def expected_f(self, a, format=None, dtype=None):
            return expected_function(a, format, dtype)

    TestXStack.__name__ = op.__name__ + "Tester"
    if hasattr(TestXStack, "__qualname__"):
        TestXStack.__qualname__ = TestXStack.__name__
    return TestXStack


TestHStack = _hv_switch(HStack, scipy_sparse.hstack)
TestVStack = _hv_switch(VStack, scipy_sparse.vstack)


def test_hstack_vstack():
    # Tests sparse.hstack and sparse.vstack (as opposed to the HStack and VStack
    # classes that they wrap).

    def make_block(dtype):
        return sparse.csr_matrix(name=f"{dtype} block", dtype=dtype)

    def get_expected_dtype(blocks, to_dtype):
        if to_dtype is None:
            block_dtypes = tuple(b.dtype for b in blocks)
            return pytensor.scalar.upcast(*block_dtypes)
        else:
            return to_dtype

    # a deliberately weird mix of dtypes to stack
    dtypes = ("complex128", pytensor.config.floatX)

    blocks = [make_block(dtype) for dtype in dtypes]

    for stack_dimension, stack_function in enumerate((sparse.vstack, sparse.hstack)):
        for to_dtype in (None, *dtypes):
            stacked_blocks = stack_function(blocks, dtype=to_dtype)
            expected_dtype = get_expected_dtype(blocks, to_dtype)
            assert stacked_blocks.dtype == expected_dtype
