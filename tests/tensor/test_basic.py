import itertools
from functools import partial
from tempfile import mkstemp

import numpy as np
import pytest

import pytensor
import pytensor.scalar as ps
import pytensor.tensor.basic as ptb
import pytensor.tensor.math as ptm
from pytensor import compile, config, function, shared
from pytensor.compile import SharedVariable
from pytensor.compile.io import In, Out
from pytensor.compile.mode import Mode, get_default_mode
from pytensor.compile.ops import DeepCopyOp
from pytensor.gradient import grad, hessian
from pytensor.graph.basic import Apply, equal_computations
from pytensor.graph.op import Op
from pytensor.graph.replace import clone_replace
from pytensor.raise_op import Assert
from pytensor.scalar import autocast_float, autocast_float_as
from pytensor.tensor import NoneConst, vectorize
from pytensor.tensor.basic import (
    Alloc,
    AllocEmpty,
    ARange,
    Choose,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    PermuteRowElements,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
    Tri,
    alloc,
    alloc_diag,
    arange,
    as_tensor_variable,
    atleast_Nd,
    cast,
    choose,
    constant,
    default,
    diag,
    expand_dims,
    eye,
    fill,
    flatnonzero,
    flatten,
    full_like,
    get_scalar_constant_value,
    get_underlying_scalar_constant_value,
    get_vector_length,
    horizontal_stack,
    identity_like,
    infer_static_shape,
    inverse_permutation,
    join,
    make_vector,
    mgrid,
    moveaxis,
    nonzero,
    nonzero_values,
    ogrid,
    ones_like,
    permute_row_elements,
    roll,
    scalar_from_tensor,
    second,
    stack,
    stacklists,
    swapaxes,
    switch,
    tensor_copy,
    tensor_from_scalar,
    tile,
    trace,
    tri,
    tril,
    tril_indices,
    tril_indices_from,
    triu,
    triu_indices,
    triu_indices_from,
    vertical_stack,
    where,
    zeros_like,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import dense_dot
from pytensor.tensor.math import sum as pt_sum
from pytensor.tensor.shape import Reshape, Shape_i, shape_padright, specify_shape
from pytensor.tensor.type import (
    TensorType,
    bscalar,
    bvector,
    col,
    dmatrix,
    dscalar,
    dscalars,
    dtensor3,
    dvector,
    fmatrix,
    fscalar,
    fscalars,
    fvector,
    imatrix,
    int_dtypes,
    iscalar,
    iscalars,
    itensor3,
    ivector,
    lscalar,
    lvector,
    matrices,
    matrix,
    row,
    scalar,
    scalars,
    tensor,
    tensor3,
    tensor4,
    vector,
    vectors,
    wvector,
)
from pytensor.tensor.variable import TensorConstant
from pytensor.utils import PYTHON_INT_BITWIDTH
from tests import unittest_tools as utt
from tests.tensor.utils import (
    ALL_DTYPES,
    COMPLEX_DTYPES,
    REAL_DTYPES,
    _good_broadcast_unary_normal,
    _grad_broadcast_unary_normal,
    eval_outputs,
    inplace_func,
    integers,
    integers_ranged,
    makeBroadcastTester,
    makeTester,
    multi_dtype_cast_checks,
    multi_dtype_checks,
    random,
    random_of_dtype,
)


pytestmark = pytest.mark.filterwarnings("error")

if config.mode == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
else:
    mode_opt = get_default_mode()

TestSwitchBroadcast = makeBroadcastTester(
    op=switch,
    expected=np.where,
    good=dict(
        all_true=(np.asarray(1, dtype=config.floatX), random(4, 5), random(4, 5)),
        false_true=(np.asarray(0, dtype=config.floatX), random(4, 5), random(4, 5)),
        mixed=(integers_ranged(0, 1, (4, 5)), random(4, 5), random(4, 5)),
    ),
    bad_build=dict(all_true=(np.asarray(1, dtype=config.floatX), random(4, 5))),
    bad_runtime=dict(
        all_true=(np.asarray(1, dtype=config.floatX), random(3, 5), random(4, 5)),
        false_true=(np.asarray(0, dtype=config.floatX), random(4, 6), random(4, 5)),
    ),
    # We suppose that cond+eps do not switch branch in switch.grad()
    # So we can't call verify_grad with cond 0.
    grad=dict(
        all_true=(np.asarray(1, dtype=config.floatX), random(4, 5), random(4, 5)),
        # false_true=(np.asarray(0, dtype=config.floatX),
        #             random(4, 5), random(4, 5)),
        # mixed=(integers_ranged(0, 1, (4, 5)).astype(config.floatX),
        #        random(4, 5), random(4, 5))
    ),
)


def _numpy_second(x, y):
    return np.broadcast_arrays(x, y)[1]


TestSecondBroadcast = makeTester(
    name="SecondBroadcastTester",
    op=second,
    expected=_numpy_second,
    good=dict(
        itertools.chain(
            multi_dtype_checks((4, 5), (5,)),
            multi_dtype_checks((2, 3, 2), (3, 2)),
            multi_dtype_checks((2, 3, 2), (2,)),
        )
    ),
    # I can't think of any way to make this fail at build time
    # Just some simple smoke tests
    bad_runtime=dict(
        fail1=(random(5, 4), random(5)),
        fail2=(random(3, 2, 3), random(6, 9)),
        fail3=(integers(6, 2, 9), random(3, 2)),
    ),
)

# We exclude local_fill_to_alloc because it optimizes the "second" node
# away from the graph.
TestSecondSameRank = makeTester(
    name="SecondSameRankTester",
    op=second,
    expected=_numpy_second,
    good=dict(
        itertools.chain(
            multi_dtype_checks((4, 5), (4, 5)),
            multi_dtype_checks((1, 2), (3, 2)),
            multi_dtype_checks((3, 2), (1, 2)),
        )
    ),
    # These sizes are not broadcastable to one another
    # and SHOULD raise an error, but currently don't.
    bad_runtime=dict(
        itertools.chain(
            multi_dtype_checks((4, 5), (5, 4)),
            multi_dtype_checks((1, 5), (5, 4)),
        )
    ),
    mode=get_default_mode().excluding("local_fill_to_alloc", "local_useless_fill"),
)

# Alloc
TestAllocBroadcast = makeBroadcastTester(
    name="AllocTester",
    op=alloc,
    expected=(lambda x, *shp: np.zeros(shp, dtype=x.dtype) + x),
    good=dict(
        correct01=(random(), np.int32(7)),
        correct01_bcast=(random(1), np.int32(7)),
        correct02=(random(), np.int32(4), np.int32(7)),
        correct12=(random(7), np.int32(4), np.int32(7)),
        correct13=(random(7), np.int32(2), np.int32(4), np.int32(7)),
        correct23=(random(4, 7), np.int32(2), np.int32(4), np.int32(7)),
        correctb1=(random(1, 7), np.int32(4), np.int32(7)),
        correctb2=(random(1, 7), np.int32(2), np.int32(4), np.int32(7)),
        correctb3=(random(7, 1), np.int32(7), np.int32(4)),
        correctb4=(random(7, 1), np.int32(2), np.int32(7), np.int32(4)),
    ),
    bad_runtime=dict(
        bad_shape12=(random(7), np.int32(7), np.int32(5)),
    ),
    bad_build=dict(
        vec=(random(1), [np.int32(2)]),
        too_big32=(random(6, 2, 4), np.int32(6), np.int32(2)),
        too_big32b=(random(6, 2, 4), np.int32(6), np.int32(4)),
        too_big32c=(random(6, 2, 4), np.int32(2), np.int32(4)),
        too_big32d=(random(6, 2, 4), np.int32(2), np.int32(6)),
        too_big32e=(random(6, 2, 4), np.int32(4), np.int32(6)),
        too_big32f=(random(6, 2, 4), np.int32(4), np.int32(2)),
    ),
)

# Since not all inputs of Alloc are differentiable, we need different testers
s1, s2, s3 = integers_ranged(1, 13, (3,))
# alloc a scalar into a vector
TestAlloc01GradBroadcast = makeBroadcastTester(
    name="Alloc01GradTester",
    op=(lambda x: alloc(x, s1)),
    expected=(lambda x: np.zeros((s1,), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(),),
        x2=(random(),),
        x3=(random(),),
    ),
)

# alloc a vector into a tensor3
TestAlloc13GradBroadcast = makeBroadcastTester(
    name="Alloc13GradTester",
    op=(lambda x: alloc(x, s1, s2, s3)),
    expected=(lambda x: np.zeros((s1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s3),),
        x2=(random(s3),),
        x3=(random(s3),),
    ),
)

# broadcast a row to a matrix
TestAllocb1GradBroadcast = makeBroadcastTester(
    name="Allocb1GradTester",
    op=lambda x: alloc(x, s1, s2),
    expected=(lambda x: np.zeros((s1, s2), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(1, s2),),
        x2=(random(1, s2),),
        x3=(random(1, s2),),
    ),
)

# broadcast a row to a tensor3
TestAllocb2GradBroadcast = makeBroadcastTester(
    name="Allocb2GradTester",
    op=lambda x: alloc(x, s1, s2, s3),
    expected=(lambda x: np.zeros((s1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(1, s3),),
        x2=(random(1, s3),),
        x3=(random(1, s3),),
    ),
)

# broadcast a col to a matrix
TestAllocb3GradBroadcast = makeBroadcastTester(
    name="Allocb3GradTester",
    op=lambda x: alloc(x, s1, s2),
    expected=(lambda x: np.zeros((s1, s2), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s1, 1),),
        x2=(random(s1, 1),),
        x3=(random(s1, 1),),
    ),
)

# broadcast a col to a tensor3
TestAllocb4GradBroadcast = makeBroadcastTester(
    name="Allocb4GradTester",
    op=lambda x: alloc(x, s1, s2, s3),
    expected=(lambda x: np.zeros((s1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s2, 1),),
        x2=(random(s2, 1),),
        x3=(random(s2, 1),),
    ),
)


# Partial broadcast of a dimshuffled input
TestAllocDimshuffleGradBroadcast = makeBroadcastTester(
    name="Allocb4GradTester",
    op=lambda x: alloc(x.dimshuffle("x", "x", 0), 1, s2, s3),
    expected=(lambda x: np.zeros((1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s3),),
        x2=(random(s3),),
        x3=(random(s3),),
    ),
)
TestAllocDimshuffleGrad2Broadcast = makeBroadcastTester(
    name="Allocb4GradTester",
    op=lambda x: alloc(x.dimshuffle("x", 0), 1, s2, s3),
    expected=(lambda x: np.zeros((1, s2, s3), dtype=x.dtype) + x),
    grad=dict(
        x1=(random(s3),),
        x2=(random(s3),),
        x3=(random(s3),),
    ),
)

TestZerosLikeBroadcast = makeBroadcastTester(
    op=zeros_like,
    expected=np.zeros_like,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
    name="ZerosLike",
)

TestOnesLikeBroadcast = makeBroadcastTester(
    op=ones_like,
    expected=np.ones_like,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
    name="OnesLike",
)


class TestMakeVector(utt.InferShapeTester):
    b = bscalar()
    i = iscalar()
    d = dscalar()

    def setup_method(self):
        self.rng = np.random.default_rng(utt.fetch_seed())
        super().setup_method()

    @pytest.mark.parametrize(
        "dtype, inputs",
        [
            ("int8", (b, b)),
            ("int32", (i, b)),
            ("int32", (b, i)),
            ("float64", (b, i)),
            ("float64", (b, d)),
            ("float64", (d, i)),
            ("float64", ()),
            ("int64", ()),
        ],
    )
    def test_make_vector(self, dtype, inputs):
        b, i, d = self.b, self.i, self.d

        val = {b: 2, i: -3, d: 0.7}

        mv = MakeVector(dtype=dtype)(*inputs)
        assert mv.dtype == dtype
        f = function([b, i, d], mv, on_unused_input="ignore")
        f(val[b], val[i], val[d])

        s = mv.sum()
        gb = pytensor.gradient.grad(s, b, disconnected_inputs="ignore")
        gi = pytensor.gradient.grad(s, i, disconnected_inputs="ignore")
        gd = pytensor.gradient.grad(s, d, disconnected_inputs="ignore")

        g = function([b, i, d], [gb, gi, gd])
        g_val = g(val[b], val[i], val[d])

        if dtype in int_dtypes:
            # The gradient should be 0
            utt.assert_allclose(g_val, 0)
        else:
            for var, grval in zip((b, i, d), g_val, strict=True):
                float_inputs = []
                if var.dtype in int_dtypes:
                    pass
                    # Currently we don't do any checks on these variables
                    # verify_grad doesn't support integer inputs yet
                    # however, the gradient on them is *not* defined to
                    # be 0
                elif var not in inputs:
                    assert grval == 0
                else:
                    float_inputs.append(var)

            # Build a function that takes float_inputs, use fix values for the
            # other inputs, and returns the MakeVector. Use it for verify_grad.
            if float_inputs:

                def fun(*fl_inputs):
                    f_inputs = []
                    for var in f_inputs:
                        if var in fl_inputs:
                            # use symbolic variable
                            f_inputs.append(var)
                        else:
                            # use constant value
                            f_inputs.append(val[var])
                    return MakeVector(dtype=dtype)(*f_inputs)

                utt.verify_grad(fun, [val[ri] for ri in float_inputs])

    def test_make_vector_fail(self):
        with pytest.raises(ValueError):
            a, b = vector(), vector()
            MakeVector()(a, b)

        a, b = iscalar(), lscalar()
        res = MakeVector("int64")(a, b)
        assert res.dtype == "int64"

        with pytest.raises(TypeError):
            res = MakeVector("int32")(a, b)

        res = MakeVector()(a)
        assert res.type.shape == (1,)

        res = MakeVector()()
        assert res.type.shape == (0,)

    def test_infer_shape(self):
        adscal = dscalar()
        bdscal = dscalar()
        aiscal = iscalar()
        biscal = iscalar()
        ciscal = iscalar()
        discal = iscalar()
        adscal_val = np.random.random()
        bdscal_val = np.random.random()
        aiscal_val = self.rng.integers(10)
        biscal_val = self.rng.integers(10)
        ciscal_val = self.rng.integers(10)
        discal_val = self.rng.integers(10)
        self._compile_and_check(
            [adscal, aiscal],
            [MakeVector("float64")(adscal, aiscal)],
            [adscal_val, aiscal_val],
            MakeVector,
        )

        self._compile_and_check(
            [adscal, bdscal, aiscal],
            [MakeVector("float64")(adscal, bdscal, aiscal)],
            [adscal_val, bdscal_val, aiscal_val],
            MakeVector,
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal, discal],
            [MakeVector("int32")(aiscal, biscal, ciscal, discal)],
            [aiscal_val, biscal_val, ciscal_val, discal_val],
            MakeVector,
        )


class ApplyDefaultTestOp(Op):
    def __init__(self, id, n_outs=1):
        self.default_output = id
        self.n_outs = n_outs

    def make_node(self, x):
        x = ptb.as_tensor_variable(x)
        return Apply(self, [x], [x.type() for _ in range(self.n_outs)])

    def perform(self, *args, **kwargs):
        raise NotImplementedError()


def test_constant():
    int8_vector_type = TensorType(dtype="int8", shape=(None,))

    # Make sure we return a `TensorConstant` unchanged
    x = TensorConstant(int8_vector_type, [1, 2])
    y = constant(x)
    assert y is x

    # Make sure we can add and remove broadcastable dimensions
    int8_scalar_type = TensorType(dtype="int8", shape=())
    x_data = np.array(2, dtype="int8")

    x = TensorConstant(int8_scalar_type, x_data)
    y = constant(x, ndim=1)
    assert y.ndim == 1
    assert np.array_equal(y.data, np.expand_dims(x_data, 0))

    y = constant(x, ndim=2)
    assert y.ndim == 2
    assert np.array_equal(y.data, np.expand_dims(x_data, (0, 1)))

    z = constant(y, ndim=0)
    assert y.ndim == 2 and z.ndim == 0
    assert np.array_equal(z.data, x_data)


def test_constant_masked_array_not_implemented():
    x = np.ma.masked_greater(np.array([1, 2, 3, 4]), 3)
    with pytest.raises(NotImplementedError, match="MaskedArrays are not supported"):
        constant(x)


class TestAsTensorVariable:
    """
    Unit test for ensuring that as_tensor_variable handles Apply objects
    correctly and removes leading broadcastable dimensions when possible.
    """

    def setup_method(self):
        self.x = scalar("x")

    def test_tensor_from_scalar(self):
        y = as_tensor_variable(ps.int8())
        assert isinstance(y.owner.op, TensorFromScalar)

    def test_default_output(self):
        good_apply_var = ApplyDefaultTestOp(0, n_outs=1).make_node(self.x)
        as_tensor_variable(good_apply_var) is good_apply_var

        good_apply_var = ApplyDefaultTestOp(-1, n_outs=1).make_node(self.x)
        as_tensor_variable(good_apply_var) is good_apply_var

        bad_apply_var = ApplyDefaultTestOp(1, n_outs=1).make_node(self.x)
        with pytest.raises(IndexError):
            _ = as_tensor_variable(bad_apply_var)

        bad_apply_var = ApplyDefaultTestOp(2.0, n_outs=1).make_node(self.x)
        with pytest.raises(TypeError):
            _ = as_tensor_variable(bad_apply_var)

        good_apply_var = ApplyDefaultTestOp(1, n_outs=2).make_node(self.x)
        as_tensor_variable(good_apply_var) is good_apply_var.outputs[1]

        bad_apply_var = ApplyDefaultTestOp(None, n_outs=2).make_node(self.x)
        with pytest.raises(TypeError, match="Multi-output Op without default_output"):
            _ = as_tensor_variable(bad_apply_var)

    def test_list(self):
        # Make sure our exception handling during `Sequence` processing doesn't
        # mask exceptions caused by unrelated logic (e.g.  computing test
        # values)
        with config.change_flags(compute_test_value="raise"), pytest.raises(ValueError):
            a = lscalar("a")
            y = (a, a, 1)
            _ = as_tensor_variable(y)

        bad_apply_var = ApplyDefaultTestOp([0, 1]).make_node(self.x)
        with pytest.raises(TypeError):
            as_tensor_variable(bad_apply_var)

    def test_ndim_strip_leading_broadcastable(self):
        x = TensorType(config.floatX, shape=(1, None))("x")
        x = as_tensor_variable(x, ndim=1)
        assert x.ndim == 1

    def test_ndim_all_broadcastable(self):
        x = TensorType(config.floatX, shape=(1, 1))("x")
        res = as_tensor_variable(x, ndim=0)
        assert res.ndim == 0

    def test_ndim_incompatible(self):
        x = TensorType(config.floatX, shape=(1, None))("x")
        with pytest.raises(ValueError, match="^Tensor of type.*"):
            as_tensor_variable(x, ndim=0)

    def test_bool(self):
        # We should not allow `as_tensor_variable` to accept `True` or `False`,
        # but it should up-cast an `ndarray` of `bool` to uint8
        with pytest.raises(TypeError):
            as_tensor_variable(True)

        ten = as_tensor_variable(np.array([True, False, False, True, True]))
        assert ten.type.dtype == "bool"

    def test_dtype(self):
        res = as_tensor_variable([])
        assert res.type.dtype == config.floatX

        res = as_tensor_variable([], dtype="int64")
        assert res.type.dtype == "int64"

        res = as_tensor_variable(np.array([1], dtype="int32"), dtype="int64")
        assert res.type.dtype == "int64"

        res = as_tensor_variable(np.array([1.0], dtype=config.floatX), dtype="int64")
        # TODO: This cross-type conversion probably shouldn't be the default.
        assert res.type.dtype == "int64"

        x = as_tensor_variable(np.array([1.0, 2.0], dtype="float64"))
        # This shouldn't convert the dtype, because it's already a `Variable`
        # with a set dtype
        res = as_tensor_variable(x, dtype="int64")
        assert res.type.dtype == "float64"

    def test_memmap(self):
        inp = np.random.random((4, 3))
        _, fname = mkstemp()
        new_inp = np.memmap(fname, dtype=inp.dtype, mode="w+", shape=inp.shape)
        new_inp[...] = inp
        res = as_tensor_variable(new_inp)
        assert isinstance(res, TensorConstant)
        assert res.data is new_inp

    @pytest.mark.parametrize(
        "dtype",
        [
            "float16",
            "float32",
            "float64",
        ],
    )
    def test_empty_dtype(self, dtype):
        with config.change_flags(floatX=dtype):
            assert as_tensor_variable(()).dtype == dtype
            assert as_tensor_variable([]).dtype == dtype

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            ([1, 2], [1, 2]),
            ([ptb.as_tensor(1), ptb.as_tensor(2)], [1, 2]),
            ([ps.constant(1), ps.constant(2)], [1, 2]),
        ],
    )
    def test_constant_consistency(self, x, y):
        a = as_tensor_variable(x)
        assert isinstance(a, TensorConstant)
        assert np.array_equal(a.data, y)

    def test_constant_identity(self):
        # Values that are already `TensorType`s shouldn't be recreated by
        # `as_tensor_variable`
        x_scalar = TensorConstant(TensorType(dtype="int8", shape=()), 2)
        a_scalar = as_tensor_variable(x_scalar)
        assert x_scalar is a_scalar

        x_vector = TensorConstant(
            TensorType(dtype="int8", shape=(None,)),
            np.array([1, 2], dtype="int8"),
        )
        a_vector = as_tensor_variable(x_vector)
        assert x_vector is a_vector

    def test_make_vector(self):
        a = iscalar()
        x = ptb.tile(a, (1, 1, 1))
        y = (constant(1, dtype="int64"), x.shape[2])
        res = ptb.as_tensor(y, ndim=1)
        assert isinstance(res.owner.op, MakeVector)
        assert tuple(res.owner.inputs) == y

        y = (1, x.shape[2])
        res = ptb.as_tensor(y)
        assert isinstance(res.owner.op, MakeVector)

    def test_multi_out(self):
        class TestOp(Op):
            def make_node(self, a, b):
                return Apply(self, [a, b], [a, b])

        with pytest.raises(TypeError):
            ptb.as_tensor(TestOp(matrix(), matrix()))

    def test_masked_array_not_implemented(
        self,
    ):
        x = np.ma.masked_greater(np.array([1, 2, 3, 4]), 3)
        with pytest.raises(NotImplementedError, match="MaskedArrays are not supported"):
            ptb.as_tensor(x)


def check_alloc_runtime_broadcast(mode):
    """Check we emmit a clear error when runtime broadcasting would occur according to Numpy rules."""
    floatX = config.floatX
    x_v = vector("x", shape=(None,))

    out = alloc(x_v, 5, 3)
    f = pytensor.function([x_v], out, mode=mode)
    TestAlloc.check_allocs_in_fgraph(f.maker.fgraph, 1)

    np.testing.assert_array_equal(
        f(x=np.zeros((3,), dtype=floatX)),
        np.zeros((5, 3), dtype=floatX),
    )
    with pytest.raises(ValueError, match="Runtime broadcasting not allowed"):
        f(x=np.zeros((1,), dtype=floatX))

    out = alloc(specify_shape(x_v, (1,)), 5, 3)
    f = pytensor.function([x_v], out, mode=mode)
    TestAlloc.check_allocs_in_fgraph(f.maker.fgraph, 1)

    np.testing.assert_array_equal(
        f(x=np.zeros((1,), dtype=floatX)),
        np.zeros((5, 3), dtype=floatX),
    )


class TestAlloc:
    dtype = config.floatX
    mode = mode_opt
    shared = staticmethod(pytensor.shared)
    allocs = [Alloc()] * 3

    @staticmethod
    def check_allocs_in_fgraph(fgraph, n):
        assert (
            len([node for node in fgraph.apply_nodes if isinstance(node.op, Alloc)])
            == n
        )

    def setup_method(self):
        self.rng = np.random.default_rng(seed=utt.fetch_seed())

    def test_alloc_constant_folding(self):
        test_params = np.asarray(self.rng.standard_normal(50 * 60), self.dtype)

        some_vector = vector("some_vector", dtype=self.dtype)
        some_matrix = some_vector.reshape((60, 50))
        variables = self.shared(np.ones((50,), dtype=self.dtype))
        idx = constant(np.arange(50))

        for alloc_, (subtensor, n_alloc) in zip(
            self.allocs,
            [
                # IncSubtensor1
                (some_matrix[:60], 2),
                # AdvancedIncSubtensor1
                (some_matrix[arange(60)], 2),
                # AdvancedIncSubtensor
                (some_matrix[idx, idx], 1),
            ],
            strict=True,
        ):
            derp = pt_sum(dense_dot(subtensor, variables))

            fobj = pytensor.function([some_vector], derp, mode=self.mode)
            grad_derp = pytensor.grad(derp, some_vector)
            fgrad = pytensor.function([some_vector], grad_derp, mode=self.mode)

            topo_obj = fobj.maker.fgraph.toposort()
            assert sum(isinstance(node.op, type(alloc_)) for node in topo_obj) == 0

            topo_grad = fgrad.maker.fgraph.toposort()
            assert (
                sum(isinstance(node.op, type(alloc_)) for node in topo_grad) == n_alloc
            ), (alloc_, subtensor, n_alloc, topo_grad)
            fobj(test_params)
            fgrad(test_params)

    def test_alloc_output(self):
        val = constant(self.rng.standard_normal((1, 1)), dtype=self.dtype)
        for alloc_ in self.allocs:
            # The output is the result of the alloc operation,
            # we do not want it to be constant-folded
            out = alloc_(val, 50, 60)

            f = pytensor.function([], out, mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert sum(isinstance(node.op, type(alloc_)) for node in topo) == 1
            assert not isinstance(topo[0].op, DeepCopyOp)

    def test_ones(self):
        for shp in [[], 1, [1], [1, 2], [1, 2, 3], np.r_[1, 2, 3]]:
            ones = pytensor.function([], [ptb.ones(shp)], mode=self.mode)
            assert np.allclose(ones(), np.ones(shp))
            # When shape is a TensorConstant
            ones_const = pytensor.function(
                [], [ptb.ones(ptb.constant(shp))], mode=self.mode
            )
            assert np.allclose(ones_const(), np.ones(shp))

        # scalar doesn't have to be provided as input
        x = scalar()
        shp = []
        ones_scalar = pytensor.function([], [ptb.ones(x.shape)], mode=self.mode)
        assert np.allclose(ones_scalar(), np.ones(shp))

        for typ, shp in [(vector, [3]), (matrix, [3, 4])]:
            x = typ()
            ones_tensor = pytensor.function([x], [ptb.ones(x.shape)], mode=self.mode)
            inp = np.zeros(shp, dtype=config.floatX)
            assert np.allclose(ones_tensor(inp), np.ones(shp))

    def test_zeros(self):
        for shp in [[], 1, [1], [1, 2], [1, 2, 3], np.r_[1, 2, 3]]:
            zeros = pytensor.function([], [ptb.zeros(shp)], mode=self.mode)
            assert np.allclose(zeros(), np.zeros(shp))
            # When shape is a TensorConstant
            zeros_const = pytensor.function(
                [], [ptb.zeros(ptb.constant(shp))], mode=self.mode
            )
            assert np.allclose(zeros_const(), np.zeros(shp))

        # scalar doesn't have to be provided as input
        x = scalar()
        shp = []
        zeros_scalar = pytensor.function([], [ptb.zeros(x.shape)], mode=self.mode)
        assert np.allclose(zeros_scalar(), np.zeros(shp))

        for typ, shp in [(vector, [3]), (matrix, [3, 4])]:
            x = typ()
            zeros_tensor = pytensor.function([x], [ptb.zeros(x.shape)], mode=self.mode)
            inp = np.zeros(shp, dtype=config.floatX)
            assert np.allclose(zeros_tensor(inp), np.zeros(shp))

    @pytest.mark.parametrize(
        "shape", [(2, 3), 5, np.int32(5), np.array(5), constant(5)]
    )
    def test_full(self, shape):
        full_pt = ptb.full(shape, 3, dtype="int64")
        res = pytensor.function([], full_pt, mode=self.mode)()
        if isinstance(shape, ptb.TensorVariable):
            shape = shape.eval()
        assert np.array_equal(res, np.full(shape, 3, dtype="int64"))

    @pytest.mark.parametrize("func", (ptb.zeros, ptb.empty))
    def test_rebuild(self, func):
        x = vector(shape=(50,))
        x_test = np.zeros((50,), dtype=config.floatX)
        y = func(x.shape)
        assert y.type.shape == (50,)
        assert y.shape.eval({x: x_test}) == (50,)
        assert y.eval({x: x_test}).shape == (50,)

        x_new = vector(shape=(100,))
        x_new_test = np.zeros((100,), dtype=config.floatX)
        y_new = clone_replace(y, {x: x_new}, rebuild_strict=False)
        assert y_new.type.shape == (100,)
        assert y_new.shape.eval({x_new: x_new_test}) == (100,)
        assert y_new.eval({x_new: x_new_test}).shape == (100,)

    def test_static_shape(self):
        x = tensor(shape=(None, 1, 5))
        d0 = scalar("d0", dtype=int)
        d1 = scalar("d1", dtype=int)
        assert ptb.alloc(x, 3, 1, 5).type.shape == (3, 1, 5)
        assert ptb.alloc(x, 3, 4, 5).type.shape == (3, 4, 5)
        assert ptb.alloc(x, d0, d1, 5).type.shape == (None, None, 5)
        assert ptb.alloc(x, d0, 1, d1).type.shape == (None, 1, 5)

        msg = "Alloc static input type and target shape are incompatible"
        with pytest.raises(ValueError, match=msg):
            ptb.alloc(x, 3, 1, 1)

        with pytest.raises(ValueError, match=msg):
            ptb.alloc(x, 3, 1, 6)

    def test_alloc_of_view_linker(self):
        """Check we can allocate a new array properly in the C linker when input is a view."""
        floatX = config.floatX

        x_v = vector("x", shape=(None,))
        dim_len = scalar("dim_len", dtype=int)
        out = alloc(specify_shape(x_v, (1,)), 5, dim_len)

        f = pytensor.function([x_v, dim_len], out, mode=Mode("c"))
        assert equal_computations(
            f.maker.fgraph.outputs, [alloc(specify_shape(x_v, (1,)), 5, dim_len)]
        )

        np.testing.assert_array_equal(
            f(x=np.zeros((1,), dtype=floatX), dim_len=3),
            np.zeros((5, 3), dtype=floatX),
        )

    @pytest.mark.parametrize("mode", (Mode("py"), Mode("c")))
    def test_runtime_broadcast(self, mode):
        check_alloc_runtime_broadcast(mode)


def test_infer_static_shape():
    with pytest.raises(TypeError, match="^Shapes must be scalar integers.*"):
        infer_static_shape([constant(1.0)])

    with (
        config.change_flags(exception_verbosity="high"),
        pytest.raises(TypeError, match=r"A\. x"),
    ):
        infer_static_shape([dscalar("x")])

    with pytest.raises(ValueError, match=".*could not be cast to have 0 dimensions"):
        infer_static_shape((as_tensor_variable([[1, 2]]),))

    constant_size = constant([1])
    specify_size = specify_shape(constant_size, [1])
    sh, static_shape = infer_static_shape(specify_size)
    assert static_shape == (1,)

    x = scalar("x")
    sh, static_shape = infer_static_shape([x.size])
    assert static_shape == (1,)


class TestEye:
    # This is slow for the ('int8', 3) version.
    def test_basic(self):
        def check(dtype, N, M_=None, k=0):
            # PyTensor does not accept None as a tensor.
            # So we must use a real value.
            M = M_
            # Currently DebugMode does not support None as inputs even if this is
            # allowed.
            if M is None and config.mode in ["DebugMode", "DEBUG_MODE"]:
                M = N
            N_symb = iscalar()
            M_symb = iscalar()
            k_symb = iscalar()
            f = function(
                [N_symb, M_symb, k_symb], eye(N_symb, M_symb, k_symb, dtype=dtype)
            )
            result = f(N, M, k)
            assert np.allclose(result, np.eye(N, M_, k, dtype=dtype))
            assert result.dtype == np.dtype(dtype)

        for dtype in ALL_DTYPES:
            check(dtype, 3)
            # M != N, k = 0
            check(dtype, 3, 5)
            check(dtype, 5, 3)
            # N == M, k != 0
            check(dtype, 3, 3, 1)
            check(dtype, 3, 3, -1)
            # N < M, k != 0
            check(dtype, 3, 5, 1)
            check(dtype, 3, 5, -1)
            # N > M, k != 0
            check(dtype, 5, 3, 1)
            check(dtype, 5, 3, -1)

    def test_static_output_type(self):
        l = lscalar("l")
        assert eye(5, 3, l).type.shape == (5, 3)
        assert eye(1, l, 3).type.shape == (1, None)


class TestTriangle:
    def test_tri(self):
        def check(dtype, N, M_=None, k=0):
            # PyTensor does not accept None as a tensor.
            # So we must use a real value.
            M = M_
            # Currently DebugMode does not support None as inputs even if this is
            # allowed.
            if M is None and config.mode in ["DebugMode", "DEBUG_MODE"]:
                M = N
            N_symb = iscalar()
            M_symb = iscalar()
            k_symb = iscalar()
            f = function(
                [N_symb, M_symb, k_symb], tri(N_symb, M_symb, k_symb, dtype=dtype)
            )
            result = f(N, M, k)
            assert np.allclose(result, np.tri(N, M_, k, dtype=dtype))
            assert result.dtype == np.dtype(dtype)

        for dtype in ["int32", "int64", "float32", "float64", "uint16", "complex64"]:
            check(dtype, 3)
            # M != N, k = 0
            check(dtype, 3, 5)
            check(dtype, 5, 3)
            # N == M, k != 0
            check(dtype, 3, 3, 1)
            check(dtype, 3, 3, -1)
            # N < M, k != 0
            check(dtype, 3, 5, 1)
            check(dtype, 3, 5, -1)
            # N > M, k != 0
            check(dtype, 5, 3, 1)
            check(dtype, 5, 3, -1)

    def test_tril_triu(self):
        """
        TODO FIXME: Parameterize this.
        """

        def check_l(m, k=0):
            m_symb = matrix(dtype=m.dtype)
            k_symb = iscalar()
            f = function([m_symb, k_symb], tril(m_symb, k_symb))
            f_indx = function(
                [m_symb, k_symb], tril_indices(m_symb.shape[0], k_symb, m_symb.shape[1])
            )
            f_indx_from = function([m_symb, k_symb], tril_indices_from(m_symb, k_symb))
            result = f(m, k)
            result_indx = f_indx(m, k)
            result_from = f_indx_from(m, k)
            assert np.allclose(result, np.tril(m, k))
            assert np.allclose(result_indx, np.tril_indices(m.shape[0], k, m.shape[1]))
            assert np.allclose(result_from, np.tril_indices_from(m, k))
            assert np.allclose(result_indx, result_from)
            assert result.dtype == np.dtype(dtype)

        def check_u(m, k=0):
            m_symb = matrix(dtype=m.dtype)
            k_symb = iscalar()
            f = function([m_symb, k_symb], triu(m_symb, k_symb))
            f_indx = function(
                [m_symb, k_symb], triu_indices(m_symb.shape[0], k_symb, m_symb.shape[1])
            )
            f_indx_from = function([m_symb, k_symb], triu_indices_from(m_symb, k_symb))
            result = f(m, k)
            result_indx = f_indx(m, k)
            result_from = f_indx_from(m, k)
            assert np.allclose(result, np.triu(m, k))
            assert np.allclose(result_indx, np.triu_indices(m.shape[0], k, m.shape[1]))
            assert np.allclose(result_from, np.triu_indices_from(m, k))
            assert np.allclose(result_indx, result_from)
            assert result.dtype == np.dtype(dtype)

        def check_l_batch(m, k=0):
            m_symb = tensor3(dtype=m.dtype)
            k_symb = iscalar()
            f = function([m_symb, k_symb], tril(m_symb, k_symb))
            for k in [-1, 0, 1]:
                result = f(m, k)
                assert np.allclose(result, np.tril(m, k))
                assert result.dtype == np.dtype(dtype)

        def check_u_batch(m):
            m_symb = tensor3(dtype=m.dtype)
            k_symb = iscalar()
            f = function([m_symb, k_symb], triu(m_symb, k_symb))
            for k in [-1, 0, 1]:
                result = f(m, k)
                assert np.allclose(result, np.triu(m, k))
                assert result.dtype == np.dtype(dtype)

        for dtype in ["int32", "int64", "float32", "float64", "uint16", "complex64"]:
            m = random_of_dtype((10, 10), dtype)
            check_l(m, 0)
            check_l(m, 1)
            check_l(m, -1)

            check_u(m, 0)
            check_u(m, 1)
            check_u(m, -1)

            m = random_of_dtype((10, 5), dtype)
            check_l(m, 0)
            check_l(m, 1)
            check_l(m, -1)

            check_u(m, 0)
            check_u(m, 1)
            check_u(m, -1)

            m = random_of_dtype((5, 5, 5), dtype)
            check_l_batch(m)
            check_u_batch(m)

            m = random_of_dtype((5, 10, 5), dtype)
            check_l_batch(m)
            check_u_batch(m)

        m = random_of_dtype((10,), dtype)
        for fn in (triu_indices_from, tril_indices_from):
            with pytest.raises(ValueError, match="must be two dimensional"):
                fn(m)


class TestNonzero:
    @config.change_flags(compute_test_value="raise")
    def test_nonzero(self):
        def check(m):
            m_symb = tensor(dtype=m.dtype, shape=(None,) * m.ndim)
            m_symb.tag.test_value = m

            res_tuple_pt = nonzero(m_symb, return_matrix=False)
            res_matrix_pt = nonzero(m_symb, return_matrix=True)

            res_tuple = tuple(r.tag.test_value for r in res_tuple_pt)
            res_matrix = res_matrix_pt.tag.test_value

            assert np.allclose(res_matrix, np.vstack(np.nonzero(m)))

            for i, j in zip(res_tuple, np.nonzero(m), strict=True):
                assert np.allclose(i, j)

        rand0d = np.empty(())
        with pytest.raises(ValueError):
            check(rand0d)

        rand1d = np.empty((8,))
        rand1d[:4] = 0
        check(rand1d)

        rand2d = np.empty((8, 9))
        rand2d[:4] = 0
        check(rand2d)

    @config.change_flags(compute_test_value="raise")
    def test_flatnonzero(self):
        def check(m):
            m_symb = tensor(dtype=m.dtype, shape=(None,) * m.ndim)
            m_symb.tag.test_value = m

            res_pt = flatnonzero(m_symb)

            result = res_pt.tag.test_value
            assert np.allclose(result, np.flatnonzero(m))

        rand0d = np.empty(())
        with pytest.raises(ValueError):
            check(rand0d)

        rand1d = np.empty((8,))
        rand1d[:4] = 0
        check(rand1d)

        rand2d = np.empty((8, 9))
        rand2d[:4] = 0
        check(rand2d)

        # Test passing a list
        m = [1, 2, 0]
        out = flatnonzero(m)
        f = function([], out)
        assert np.array_equal(f(), np.flatnonzero(m))

    @config.change_flags(compute_test_value="raise")
    def test_nonzero_values(self):
        def check(m):
            m_symb = tensor(dtype=m.dtype, shape=(None,) * m.ndim)
            m_symb.tag.test_value = m

            res_pt = nonzero_values(m_symb)

            result = res_pt.tag.test_value
            assert np.allclose(result, m[np.nonzero(m)], equal_nan=True)

        rand0d = np.empty(())
        with pytest.raises(ValueError):
            check(rand0d)

        rand1d = np.empty((8,))
        rand1d[:4] = 0
        check(rand1d)

        rand2d = np.empty((8, 9))
        rand2d[:4] = 0
        check(rand2d)


def test_identity():
    def check(dtype):
        obj = random_of_dtype((2,), dtype)
        sym = vector(dtype=dtype)
        f = function([sym], tensor_copy(sym))
        assert np.all(obj == f(obj))
        assert obj.dtype == f(obj).dtype
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        if config.mode != "FAST_COMPILE":
            assert isinstance(topo[0].op, DeepCopyOp)

    for dtype in ALL_DTYPES:
        check(dtype)


class TestCast:
    def test_can_use_numpy_types(self):
        x = vector(dtype=np.int32)
        y = cast(x, np.int64)
        f = function([x], y)
        assert f(np.array([1, 2], dtype=np.int32)).dtype == np.int64

    @pytest.mark.parametrize(
        "test_name, obj_dtype",
        itertools.chain(
            multi_dtype_cast_checks((2,), dtypes=REAL_DTYPES),
            # Casts from foo to foo
            [
                (
                    f"{random_of_dtype((2,), dtype)}_{dtype}",
                    (random_of_dtype((2,), dtype), dtype),
                )
                for dtype in ALL_DTYPES
            ],
        ),
    )
    def test_good_between_real_types(self, test_name, obj_dtype):
        (obj, dtype) = obj_dtype
        inp = vector(dtype=obj.dtype)
        out = cast(inp, dtype=dtype)
        f = function([inp], out)
        assert f(obj).dtype == np.dtype(dtype)

        # Test astype too
        out2 = inp.astype(dtype=dtype)
        assert out2.type == out.type

    @pytest.mark.parametrize("real_dtype", REAL_DTYPES)
    @pytest.mark.parametrize("complex_dtype", COMPLEX_DTYPES)
    def test_cast_from_real_to_complex(self, real_dtype, complex_dtype):
        inp = vector(dtype=real_dtype)
        out = cast(inp, dtype=complex_dtype)
        f = function([inp], out)
        obj = random_of_dtype((2,), real_dtype)
        assert f(obj).dtype == np.dtype(complex_dtype)

    @pytest.mark.parametrize("real_dtype", REAL_DTYPES)
    @pytest.mark.parametrize("complex_dtype", COMPLEX_DTYPES)
    def test_cast_from_complex_to_real_raises_error(self, real_dtype, complex_dtype):
        inp = vector(dtype=complex_dtype)
        with pytest.raises(TypeError):
            cast(inp, dtype=real_dtype)


# TODO: consider moving this function / functionality to gradient.py
#      rationale: it's tricky, and necessary every time you want to verify
#      gradient numerically


def test_basic_allclose():
    # This was raised by a user in https://github.com/Theano/Theano/issues/2975
    assert ptm._allclose(-0.311023883434, -0.311022856884)


def test_get_vector_length():
    # Test `Constant`s
    empty_tuple = as_tensor_variable(())
    assert 0 == get_vector_length(empty_tuple)

    x = as_tensor_variable((1, 2, 3))
    assert 3 == get_vector_length(x)

    # Test `Join`s
    z = join(0, as_tensor_variable(1, ndim=1), as_tensor_variable(x.shape[0], ndim=1))
    assert isinstance(z.owner.op, Join)
    assert get_vector_length(z) == 2

    z = join(
        0, as_tensor_variable([1, 2], ndim=1), as_tensor_variable(x.shape[0], ndim=1)
    )
    assert isinstance(z.owner.op, Join)
    assert get_vector_length(z) == 3

    z = join(
        lscalar(),
        as_tensor_variable([1, 2], ndim=1),
        as_tensor_variable([3, 4], ndim=1),
    )
    with pytest.raises(ValueError, match="^Length of .*"):
        get_vector_length(z)

    # Test `MakeVector`s
    x = lscalar("x")
    y = dscalar("y")

    triple = as_tensor_variable((x, y, 9.0))
    assert 3 == get_vector_length(triple)

    triple = cast(as_tensor_variable((x, y, 9.0)), "int64")
    assert 3 == get_vector_length(triple)

    a, b, c = triple
    mode = pytensor.compile.get_default_mode().excluding("constant_folding")
    f = function([x, y], [b, c, a], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert any(True for node in topo if isinstance(node.op, MakeVector))

    assert np.allclose(f(4, 5), [5, 9, 4])

    # Test `Alloc`s
    assert 3 == get_vector_length(alloc(0, 3))

    assert 5 == get_vector_length(tensor(dtype=np.float64, shape=(5,)))


class TestJoinAndSplit:
    # Split is tested by each verify_grad method.
    def setup_method(self):
        Join.debug = False

        self.mode = pytensor.compile.get_default_mode().excluding("constant_folding")
        self.join_op = Join()
        self.split_op_class = Split
        self.make_vector_op = MakeVector()
        self.floatX = config.floatX
        self.shared = shared

    def eval_outputs_and_check_join(self, outputs):
        f = pytensor.function([], outputs, self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]
        variables = f()
        if isinstance(variables, tuple | list) and len(variables) == 1:
            return variables[0]
        return variables

    def eval_outputs_and_check_vector(self, outputs, make_vector_op=None):
        if make_vector_op is None:
            make_vector_op = self.make_vector_op
        f = pytensor.function([], outputs, self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(make_vector_op))]
        variables = f()
        if isinstance(variables, tuple | list) and len(variables) == 1:
            return variables[0]
        return variables

    def test_input_validation(self):
        with pytest.raises(TypeError, match=".*integer.*"):
            Split(2)(matrix(), dscalar(), [1, 1])

        with pytest.raises(TypeError, match=".*integer.*"):
            Split(2)(matrix(), ivector(), [1, 1])

        with pytest.raises(TypeError, match=".*integer.*"):
            join(dscalar(), matrix(), matrix())

    def test_join_scalar(self):
        a = as_tensor_variable(1)
        b = as_tensor_variable(2)
        with pytest.raises(TypeError):
            join(0, a, b)

    def test_stack_mixed_type_constants(self):
        # tested only on cpu as gpu support only float32
        a = as_tensor_variable(1)
        b = as_tensor_variable(2.0)
        c = pytensor.shared(np.asarray(3.0, dtype=self.floatX))
        s = stack([a, b, c])
        want = np.array([1, 2, 3])
        out = self.eval_outputs_and_check_vector([s], MakeVector())
        assert (out == want).all()

    def test_stack_scalar(self):
        a = self.shared(np.asarray(1.0, dtype=self.floatX))
        b = as_tensor_variable(2.0)
        c = as_tensor_variable(3.0)
        s = stack([a, b, c])

        want = np.array([1, 2, 3])
        out = self.eval_outputs_and_check_vector([s])
        assert (out == want).all()

    def test_stack_scalar_make_vector(self):
        # Test that calling stack() on scalars instantiates MakeVector,
        # not Join. Test that the floatX dtype stay floatX, not downcasted
        # to int64
        a = scalar("a", dtype=self.floatX)
        b = scalar("b", dtype=self.floatX)
        s = stack([a, b, a, b])
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        # print val
        assert np.all(val == [1, 2, 1, 2])
        topo = f.maker.fgraph.toposort()
        assert len([n for n in topo if isinstance(n.op, MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, type(self.join_op))]) == 0
        assert f.maker.fgraph.outputs[0].dtype == self.floatX

    def test_stack_scalar_make_vector_dtype(self):
        # Test that calling stack() on scalars instantiates MakeVector,
        # event when the scalar don't have the same dtype.
        a = iscalar("a")
        b = lscalar("b")
        s = stack([a, b, a, b])
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        assert np.all(val == [1, 2, 1, 2])
        topo = f.maker.fgraph.toposort()
        assert len([n for n in topo if isinstance(n.op, MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, type(self.join_op))]) == 0
        assert f.maker.fgraph.outputs[0].dtype == "int64"

    def test_stack_scalar_make_vector_constant(self):
        # Test that calling stack() on scalars instantiates MakeVector,
        # even when the scalars are non-symbolic ints.
        a = iscalar("a")
        b = lscalar("b")
        # test when the constant is the first element.
        # The first element is used in a special way
        s = stack([10, a, b, np.int8(3), np.array(4, dtype=np.int8)])
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        assert np.all(val == [10, 1, 2, 3, 4])
        topo = f.maker.fgraph.toposort()
        assert len([n for n in topo if isinstance(n.op, MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, type(self.join_op))]) == 0
        assert f.maker.fgraph.outputs[0].dtype == "int64"

    def test_stack_new_interface(self):
        # Test the new numpy-like interface: stack(tensors, axis=0).

        a = imatrix("a")
        b = imatrix("b")

        # Testing axis parameter
        s3 = stack([a, b], 1)
        f = function([a, b], s3, mode=self.mode)
        v3 = f([[1, 2]], [[3, 4]])
        v4 = np.array([[[1, 2], [3, 4]]])
        assert v3.shape == v4.shape
        assert np.all(v3 == v4)
        # Testing negative axis
        v1 = [[1, 2, 3], [4, 5, 6]]
        v2 = [[7, 8, 9], [10, 11, 12]]
        s = stack([a, b], axis=-1)
        f = function([a, b], s, mode=self.mode)
        v = np.zeros((2, 3, 2))
        v[:, :, 0] = v1
        v[:, :, 1] = v2
        out = f(v1, v2)
        assert v.shape == out.shape
        assert np.all(v == out)
        s = stack([a, b], axis=-2)
        f = function([a, b], s, mode=self.mode)
        v = np.zeros((2, 2, 3))
        v[:, 0, :] = v1
        v[:, 1, :] = v2
        out = f(v1, v2)
        assert v.shape == out.shape
        assert np.all(v == out)
        # Testing out-of-bounds axis
        with pytest.raises(IndexError):
            stack([a, b], 4)
        with pytest.raises(IndexError):
            stack([a, b], -4)

        # Testing depreciation warning is now an informative error
        with pytest.raises(TypeError, match="First argument should be a Sequence"):
            s = stack(a, b)

    def test_stack_empty(self):
        # Do not support stacking an empty sequence
        with pytest.raises(ValueError, match="No tensor arguments provided"):
            stack([])

    def test_stack_hessian(self):
        # Test the gradient of stack when used in hessian, see gh-1589
        a = dvector("a")
        b = dvector("b")
        A = stack([a, b])
        B = A.T.dot(A)
        Ha, Hb = hessian(B.sum(), [a, b])

        # Try some values
        a_v = np.random.random(4)
        b_v = np.random.random(4)
        f = pytensor.function([a, b], [Ha, Hb])
        Ha_v, Hb_v = f(a_v, b_v)
        # The Hessian is always a matrix full of 2
        assert Ha_v.shape == (4, 4)
        assert Hb_v.shape == (4, 4)
        assert np.allclose(Ha_v, 2.0)
        assert np.allclose(Hb_v, 2.0)

    def test_stack_hessian2(self):
        # Test the hessian macro when the gradient itself does not depend
        # on the input (but the cost does)
        a = dvector("a")
        b = dvector("b")
        A = stack([a, b])
        Ha, Hb = hessian(A.sum(), [a, b])

        # Try some values
        a_v = np.random.random(4)
        b_v = np.random.random(4)
        f = pytensor.function([a, b], [Ha, Hb])
        Ha_v, Hb_v = f(a_v, b_v)
        # The Hessian is always a matrix full of 0
        assert Ha_v.shape == (4, 4)
        assert Hb_v.shape == (4, 4)
        assert np.allclose(Ha_v, 0.0)
        assert np.allclose(Hb_v, 0.0)

    def test_join_concatenate_one_element(self):
        # Fast test of concatenate as this is an alias for join.
        # also test that we remove the Join op if there is only 1 input
        m = fmatrix()
        c = ptb.concatenate([m])
        f = pytensor.function(
            inputs=[m], outputs=[c], mode=self.mode.including("local_join_1")
        )
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, DeepCopyOp)

    def test_join_vector(self):
        a = self.shared(np.array([1, 2, 3], dtype=self.floatX))
        b = as_tensor_variable(np.array([7, 8, 9], dtype=self.floatX))

        s = join(0, a, b)
        want = np.array([1, 2, 3, 7, 8, 9])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_roll(self):
        for get_shift in [lambda a: a, lambda x: pytensor.shared(x)]:
            # Test simple 1D example
            a = self.shared(np.array([1, 2, 3, 4, 5, 6], dtype=self.floatX))
            b = roll(a, get_shift(2))
            want = np.array([5, 6, 1, 2, 3, 4])
            out = pytensor.function([], b)()

            assert (out == want).all()

            # Test simple 1D example with explicit 0 axis
            b = roll(a, get_shift(-1), 0)
            want = np.array([2, 3, 4, 5, 6, 1])
            out = pytensor.function([], b)()

            assert (out == want).all()

            # Test 2D example - ensure that behavior matches np.roll behavior
            a = self.shared(np.arange(21).reshape((3, 7)).astype(self.floatX))
            b = roll(a, get_shift(-2), 1)

            want = np.roll(a.get_value(borrow=True), -2, 1)
            out = pytensor.function([], b)()

            assert (out == want).all()

            # Test example when axis < 0 - ensure that behavior matches np.roll behavior
            a = self.shared(np.arange(24).reshape((3, 2, 4)).astype(self.floatX))
            b = roll(a, get_shift(-2), -2)

            want = np.roll(a.get_value(borrow=True), -2, -2)
            out = pytensor.function([], b)()

            assert (out == want).all()

            # Test rolling on axis 0
            want = np.roll(a.get_value(borrow=True), -2, 0)
            b = roll(a, get_shift(-2), 0)
            out = pytensor.function([], b)()

            assert (out == want).all()

            # Test rolling on default axis with ndim > 1
            want = np.roll(a.get_value(borrow=True), 2)
            b = roll(a, get_shift(2))
            out = pytensor.function([], b)()

            assert (out == want).all()

            # Test rolling on axis 0 with a positive shift that is
            # larger than axis size
            want = np.roll(a.get_value(borrow=True), 4, 0)
            b = roll(a, get_shift(4), 0)
            out = pytensor.function([], b)()

            assert (out == want).all()

            # Test rolling on axis 0 with a negative shift that is
            # larger than axis size
            want = np.roll(a.get_value(borrow=True), -4, 0)
            b = roll(a, get_shift(-4), 0)
            out = pytensor.function([], b)()

            assert (out == want).all()

            a = [1, 2, 3, 4, 5, 6]
            b = roll(a, get_shift(2))
            want = np.array([5, 6, 1, 2, 3, 4])
            out = pytensor.function([], b)()

            assert (out == want).all()

    def test_stack_vector(self):
        a = self.shared(np.array([1, 2, 3], dtype=self.floatX))
        b = as_tensor_variable(np.array([7, 8, 9], dtype=self.floatX))

        s = stack([a, b])
        want = np.array([[1, 2, 3], [7, 8, 9]])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_join_matrix0(self):
        a = self.shared(np.array([[1, 2, 3], [4, 5, 6]], dtype=self.floatX))
        b = as_tensor_variable(np.array([[7, 8, 9]], dtype=self.floatX))
        s = join(0, a, b)

        want = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    @pytest.mark.parametrize("py_impl", (False, True))
    def test_join_matrix1(self, py_impl):
        if py_impl:
            impl_ctxt = pytensor.config.change_flags(cxx="")
        else:
            impl_ctxt = pytensor.config.change_flags()

        av = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
        bv = np.array([[0.7], [0.8]], dtype="float32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = np.array([[0.1, 0.2, 0.3, 0.7], [0.4, 0.5, 0.6, 0.8]], dtype="float32")

        with impl_ctxt:
            out = self.eval_outputs_and_check_join([s])
            assert (out == want).all()

            utt.verify_grad(lambda a, b: join(1, a, b), [av, bv], mode=self.mode)

    def test_join_matrix_dtypes(self):
        if "float32" in self.shared.__name__:
            pytest.skip(
                "The shared variable constructor"
                " need to support other dtype then float32"
            )
        # Test mixed dtype. There was a bug that caused crash in the past.
        av = np.array([[1, 2, 3], [4, 5, 6]], dtype="int8")
        bv = np.array([[7], [8]], dtype="float32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = np.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype="float32")
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        grad(s.sum(), b)
        grad(s.sum(), a)
        utt.verify_grad(lambda b: join(1, a, b), [bv], eps=1.0e-2, mode=self.mode)

    def test_join_matrix_ints(self):
        if "float32" in self.shared.__name__:
            pytest.skip(
                "The shared variable constructor"
                " need to support other dtype then float32"
            )
        # Test mixed dtype. There was a bug that caused crash in the past.
        av = np.array([[1, 2, 3], [4, 5, 6]], dtype="int8")
        bv = np.array([[7], [8]], dtype="int32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = np.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype="float32")
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        assert (np.asarray(grad(s.sum(), b).eval()) == 0).all()
        assert (np.asarray(grad(s.sum(), a).eval()) == 0).all()

    def test_join_matrix1_using_vertical_stack(self):
        a = self.shared(np.array([[1, 2, 3], [4, 5, 6]], dtype=self.floatX))
        b = as_tensor_variable(np.array([[7, 8, 9]], dtype=self.floatX))
        c = as_tensor_variable(np.array([[9, 8, 7]], dtype=self.floatX))
        s = vertical_stack(a, b, c)

        want = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [9, 8, 7]])
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

    def test_join_matrix1_using_horizontal_stack(self):
        av = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="float32")
        bv = np.array([[0.7], [0.8]], dtype="float32")
        cv = np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]], dtype="float32")
        a = self.shared(av)
        b = as_tensor_variable(bv)
        c = as_tensor_variable(cv)
        s = horizontal_stack(a, b, c)
        want = np.array(
            [[0.1, 0.2, 0.3, 0.7, 0.3, 0.2, 0.1], [0.4, 0.5, 0.6, 0.8, 0.6, 0.5, 0.4]],
            dtype="float32",
        )
        out = self.eval_outputs_and_check_join([s])
        assert (out == want).all()

        utt.verify_grad(lambda a, b: join(1, a, b), [av, bv], mode=self.mode)

    def test_join_matrixV(self):
        # variable join axis
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)
        ax = lscalar()
        s = join(ax, a, b)

        f = inplace_func([ax], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        got = f(0)
        assert np.allclose(got, want)

        want = np.array(
            [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]]
        )
        got = f(1)
        assert np.allclose(got, want)

        utt.verify_grad(lambda a, b: join(0, a, b), [v, 2 * v], mode=self.mode)
        utt.verify_grad(lambda a, b: join(1, a, b), [v, 2 * v], mode=self.mode)

    def test_join_matrixV_negative_axis(self):
        # variable join negative axis
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)
        ax = lscalar()
        s = join(ax, a, b)

        f = inplace_func([ax], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]]
        )

        got = f(-1)
        assert np.allclose(got, want)

        want = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        got = f(-2)
        assert np.allclose(got, want)

        with pytest.raises(ValueError):
            f(-3)

    @pytest.mark.parametrize("py_impl", (False, True))
    def test_join_matrixC_negative_axis(self, py_impl):
        if py_impl:
            impl_ctxt = pytensor.config.change_flags(cxx="")
        else:
            impl_ctxt = pytensor.config.change_flags()

        # constant join negative axis
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)

        s = join(-1, a, b)

        with impl_ctxt:
            f = pytensor.function([], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.4, 0.5, 0.6]]
        )

        got = f()
        assert np.allclose(got, want)

        s = join(-2, a, b)

        with impl_ctxt:
            f = pytensor.function([], [s], mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        want = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        got = f()
        assert np.allclose(got, want)

        with pytest.raises(ValueError):
            join(-3, a, b)

        with impl_ctxt:
            utt.verify_grad(lambda a, b: join(-1, a, b), [v, 2 * v], mode=self.mode)

    def test_broadcastable_flag_assignment_mixed_otheraxes(self):
        # Test that the broadcastable flags for the output of
        # a join operation on non-join axes are True if one or
        # more inputs is broadcastable on that dimension.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((1, 4, 1)).astype(self.floatX)
        b_val = rng.random((1, 3, 1)).astype(self.floatX)

        a = self.shared(a_val, shape=(None, None, 1))
        b = self.shared(b_val, shape=(1, None, 1))
        c = self.join_op(1, a, b)
        assert c.type.shape == (1, None, 1)

        # Opt can remplace the int by an PyTensor constant
        c = self.join_op(constant(1), a, b)
        assert c.type.shape == (1, None, 1)

        # In case futur opt insert other useless stuff
        c = self.join_op(cast(constant(1), dtype="int32"), a, b)
        assert c.type.shape == (1, None, 1)

        f = function([], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        f()
        utt.verify_grad(
            (lambda a, b: join(1, a, b)), [a_val, b_val], rng=rng, mode=self.mode
        )

        # Should raise an error if dimension 0 does not match
        a.set_value(rng.random((2, 4, 1)).astype(self.floatX))
        with pytest.raises(ValueError):
            f()

    def test_broadcastable_flag_assignment_mixed_thisaxes(self):
        # Test that the broadcastable flag of the join axis
        # is False when some inputs are broadcastable on that
        # dimension.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((2, 4, 1)).astype(self.floatX)
        b_val = rng.random((1, 4, 1)).astype(self.floatX)

        a = self.shared(a_val, shape=(None, None, 1))
        b = self.shared(b_val, shape=(1, None, 1))
        c = self.join_op(0, a, b)
        assert c.type.shape[0] != 1

        f = function([], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        f()
        utt.verify_grad(
            (lambda a, b: join(0, a, b)), [a_val, b_val], rng=rng, mode=self.mode
        )
        # Should raise an error if b_val.shape[0] is not 1
        # We can't set the value|
        with pytest.raises(TypeError):
            b.set_value(rng.random((3, 4, 1)).astype(self.floatX))
        a = TensorType(dtype=self.floatX, shape=(None, None, 1))()
        b = TensorType(dtype=self.floatX, shape=(1, None, 1))()
        c = self.join_op(0, a, b)
        f = function([a, b], c, mode=self.mode)
        bad_b_val = rng.random((3, 4, 1)).astype(self.floatX)
        with pytest.raises(TypeError):
            f(a_val, bad_b_val)

    def test_broadcastable_flags_all_broadcastable_on_joinaxis(self):
        # Test that joining together several inputs which are all
        # broadcastable on the join dimension results in the output
        # being non-broadcastable on the join dimension.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((1, 4, 1)).astype(self.floatX)
        b_val = rng.random((1, 4, 1)).astype(self.floatX)

        a = self.shared(a_val, shape=(1, None, 1))
        b = self.shared(b_val, shape=(1, None, 1))
        c = self.join_op(0, a, b)
        assert c.type.shape[0] != 1

        f = function([], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        f()
        utt.verify_grad(
            (lambda a, b: join(0, a, b)), [a_val, b_val], rng=rng, mode=self.mode
        )

    def test_broadcastable_single_input_broadcastable_dimension(self):
        # Test that all broadcastable flags are preserved by a
        # single-input join.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((1, 4, 1)).astype(self.floatX)
        a = self.shared(a_val, shape=(1, None, 1))
        b = self.join_op(0, a)
        assert b.type.shape[0] == 1
        assert b.type.shape[2] == 1
        assert b.type.shape[1] != 1

        f = function([], b, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert not [
                True for node in topo if isinstance(node.op, type(self.join_op))
            ]

        f()
        utt.verify_grad((lambda a: join(0, a)), [a_val], rng=rng, mode=self.mode)
        # Should raise an error if length of dimension 0 is not 1
        with pytest.raises(TypeError):
            a.set_value(rng.random((2, 4, 1)).astype(self.floatX))
        # with pytest.raises(TypeError):
        #    f(bad_a_val)

    def test_broadcastable_flags_many_dims_and_inputs(self):
        # Test that the right broadcastable flags get set for a join
        # with many inputs and many input dimensions.
        a = TensorType(dtype=self.floatX, shape=(1, None, 1, None, None, None))()
        b = TensorType(dtype=self.floatX, shape=(1, 1, 1, None, None, None))()
        c = TensorType(dtype=self.floatX, shape=(1, None, None, None, None, None))()
        d = TensorType(dtype=self.floatX, shape=(1, None, 1, 1, None, 1))()
        e = TensorType(dtype=self.floatX, shape=(1, None, 1, None, None, 1))()

        f = self.join_op(0, a, b, c, d, e)
        fb = tuple(s == 1 for s in f.type.shape)
        assert f.type.shape == (5, 1, 1, 1, None, 1)
        assert fb == (False, True, True, True, False, True)

        g = self.join_op(1, a, b, c, d, e)
        gb = tuple(s == 1 for s in g.type.shape)
        assert g.type.shape == (1, None, 1, 1, None, 1)
        assert gb == (True, False, True, True, False, True)

        h = self.join_op(4, a, b, c, d, e)
        hb = tuple(s == 1 for s in h.type.shape)
        assert h.type.shape == (1, 1, 1, 1, None, 1)
        assert hb == (True, True, True, True, False, True)

        f = function([a, b, c, d, e], f, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert [True for node in topo if isinstance(node.op, type(self.join_op))]

        rng = np.random.default_rng(seed=utt.fetch_seed())
        a_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        b_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        c_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        d_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        e_val = rng.random((1, 1, 1, 1, 2, 1)).astype(self.floatX)
        f(a_val, b_val, c_val, d_val, e_val)
        utt.verify_grad(
            (lambda a, b, c, d, e: join(0, a, b, c, d, e)),
            [a_val, b_val, c_val, d_val, e_val],
            rng=rng,
            mode=self.mode,
        )
        # Should raise an error if length of dimension 0 is not 1
        bad_val = rng.random((2, 1, 1, 1, 2, 1)).astype(self.floatX)
        with pytest.raises(TypeError):
            f(bad_val, b_val, c_val, d_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, bad_val, c_val, d_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, b_val, bad_val, d_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, b_val, c_val, bad_val, e_val)
        with pytest.raises(TypeError):
            f(a_val, b_val, c_val, d_val, bad_val)
        # Should raise an error if any dimension other than 4 has length != 1
        bad_a_val = rng.random((1, 2, 1, 1, 2, 1)).astype(self.floatX)
        bad_b_val = rng.random((1, 1, 1, 1, 2, 2)).astype(self.floatX)
        bad_c_val = rng.random((1, 1, 2, 1, 2, 1)).astype(self.floatX)
        bad_d_val = rng.random((1, 2, 1, 1, 2, 1)).astype(self.floatX)
        bad_e_val = rng.random((1, 1, 1, 2, 2, 1)).astype(self.floatX)
        with pytest.raises(ValueError):
            f(bad_a_val, b_val, c_val, d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, bad_b_val, c_val, d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, b_val, bad_c_val, d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, b_val, c_val, bad_d_val, e_val)
        with pytest.raises(ValueError):
            f(a_val, b_val, c_val, d_val, bad_e_val)

    def test_infer_shape_join(self):
        def get_mat(s1, s2):
            return np.asarray(np.random.uniform(size=(s1, s2)), dtype=self.floatX)

        x1 = self.shared(get_mat(3, 4))
        x2 = self.shared(get_mat(2, 4))
        x3 = self.shared(get_mat(1, 4))

        # Test dim 0
        z = self.join_op(0, x1, x2, x3)
        f = pytensor.function([], z.shape, mode=self.mode)
        topo = f.maker.fgraph.toposort()

        out = f()
        assert (out == [6, 4]).all()

        if config.mode != "FAST_COMPILE":
            for node in f.maker.fgraph.toposort():
                assert not isinstance(node.op, type(self.join_op))

        # Test dim 1
        x1.set_value(get_mat(3, 4))
        x2.set_value(get_mat(3, 4))
        x3.set_value(get_mat(3, 5))
        z = self.join_op(1, x1, x2, x3)
        f = pytensor.function([], z.shape, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        out = f()
        assert (out == [3, 13]).all()

        if config.mode != "FAST_COMPILE":
            for node in topo:
                assert not isinstance(node.op, type(self.join_op))

    def test_rebroadcast(self):
        # Regression test for a crash that used to happen when rebroadcasting.
        x = TensorType(self.floatX, shape=(None, None, 1))()
        u = TensorType(self.floatX, shape=(None, None, 1))()
        # This line used to crash.
        ptb.concatenate([x, -u], axis=2)

    def test_concatenate_same(self):
        # Test that we can concatenate the same tensor multiple time.

        # In the past it was broken on the GPU.
        rng = np.random.default_rng(seed=utt.fetch_seed())
        T_shared = self.shared(rng.random((3, 4)).astype(self.floatX))
        Tout = ptb.concatenate([T_shared, T_shared])
        f = function([], Tout, mode=self.mode)
        out = f()
        if config.mode != "FAST_COMPILE":
            assert [
                True
                for node in f.maker.fgraph.toposort()
                if isinstance(node.op, type(self.join_op))
            ]
        assert np.allclose(
            out, np.concatenate([T_shared.get_value(), T_shared.get_value()])
        )

    def test_mixed_ndim_error(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        v = self.shared(rng.random(4).astype(self.floatX))
        m = self.shared(rng.random((4, 4)).astype(self.floatX))
        with pytest.raises(TypeError, match="same number of dimensions"):
            self.join_op(0, v, m)

    def test_static_shape_inference(self):
        a = ptb.tensor(dtype="int8", shape=(2, 3))
        b = ptb.tensor(dtype="int8", shape=(2, 5))

        res = ptb.join(1, a, b).type.shape
        assert res == (2, 8)
        assert all(isinstance(s, int) for s in res)

        res = ptb.join(-1, a, b).type.shape
        assert res == (2, 8)
        assert all(isinstance(s, int) for s in res)

        # Check early informative errors from static shape info
        with pytest.raises(ValueError, match="must match exactly"):
            ptb.join(0, ptb.ones((2, 3)), ptb.ones((2, 5)))

        # Check partial inference
        d = ptb.tensor(dtype="int8", shape=(2, None))
        res = ptb.join(1, a, b, d).type.shape
        assert res == (2, None)
        assert isinstance(res[0], int)

    def test_split_0elem(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        m = self.shared(rng.random((4, 6)).astype(self.floatX))
        o = self.split_op_class(2)(m, 0, [4, 0])
        f = function([], o, mode=self.mode)
        assert any(
            isinstance(node.op, self.split_op_class)
            for node in f.maker.fgraph.toposort()
        )
        o1, o2 = f()
        assert np.allclose(o1, m.get_value(borrow=True))
        assert np.allclose(o2, m.get_value(borrow=True)[4:])

    @config.change_flags(compute_test_value="off")
    def test_split_neg(self):
        rng = np.random.default_rng(seed=utt.fetch_seed())
        m = self.shared(rng.random((4, 6)).astype(self.floatX))
        o = self.split_op_class(2)(m, 0, [5, -1])
        f = function([], o, mode=self.mode)
        assert any(
            isinstance(node.op, self.split_op_class)
            for node in f.maker.fgraph.toposort()
        )
        with pytest.raises(ValueError):
            f()

    def test_split_static_shape(self):
        x = TensorType("floatX", shape=(5,))("x")
        s = iscalar("s")
        y = Split(2)(x, 0, [s, 5 - s])[0]
        assert y.type.shape == (None,)

    def test_join_oneInput(self):
        # Test join when only 1 input is given.
        #
        # This functions tests the case when concatenate is called
        # on an array of tensors but the array has only one element.
        # In this case, we would like to avoid the computational
        # overhead of concatenation of one element.
        x_0 = fmatrix()
        x_1 = fmatrix()
        x_2 = fvector()
        join_0 = ptb.concatenate([x_0], axis=1)
        join_1 = ptb.concatenate([x_0, x_1, shape_padright(x_2)], axis=1)

        assert join_0 is x_0
        assert join_1 is not x_0

    @pytest.mark.parametrize("linker", ("py", "c"))
    def test_split_view(self, linker):
        x = vector("x")
        axis = 0
        op = Split(len_splits=3)
        assert op.view_map == {0: [0], 1: [0], 2: [0]}
        splits = op(x, axis, [0, 3, 2])

        mode = Mode(linker)
        f = pytensor.function(
            [In(x, borrow=True)], [Out(s, borrow=True) for s in splits], mode=mode
        )
        x_test = np.arange(5, dtype=config.floatX)
        res = f(x_test)
        for r, expected in zip(res, ([], [0, 1, 2], [3, 4]), strict=True):
            assert np.allclose(r, expected)
            assert r.base is x_test

    @pytest.mark.parametrize("gc", (True, False), ids=lambda x: f"gc={x}")
    @pytest.mark.parametrize("memory_layout", ["C-contiguous", "F-contiguous", "Mixed"])
    @pytest.mark.parametrize("axis", (0, 1), ids=lambda x: f"axis={x}")
    @pytest.mark.parametrize("ndim", (1, 2), ids=["vector", "matrix"])
    @config.change_flags(cmodule__warn_no_version=False)
    def test_join_performance(self, ndim, axis, memory_layout, gc, benchmark):
        if ndim == 1 and not (memory_layout == "C-contiguous" and axis == 0):
            pytest.skip("Redundant parametrization")
        n = 64
        inputs = vectors("abcdef") if ndim == 1 else matrices("abcdef")
        out = join(axis, *inputs)
        fn = pytensor.function(inputs, Out(out, borrow=True), trust_input=True)
        fn.vm.allow_gc = gc
        test_values = [np.zeros((n, n)[:ndim], dtype=inputs[0].dtype) for _ in inputs]
        if memory_layout == "C-contiguous":
            pass
        elif memory_layout == "F-contiguous":
            test_values = [t.T for t in test_values]
        elif memory_layout == "Mixed":
            test_values = [t if i % 2 else t.T for i, t in enumerate(test_values)]
        else:
            raise ValueError

        assert fn(*test_values).shape == (n * 6, n)[:ndim] if axis == 0 else (n, n * 6)
        benchmark(fn, *test_values)

    def test_join_negative_axis_rewrite(self):
        """Test that constant negative axis is rewritten to positive axis in make_node."""
        v = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=self.floatX)
        a = self.shared(v)
        b = as_tensor_variable(v)

        assert equal_computations([join(-1, a, b)], [join(1, a, b)])
        assert equal_computations([join(-2, a, b)], [join(0, a, b)])


def test_TensorFromScalar():
    s = ps.constant(56)
    t = tensor_from_scalar(s)
    assert t.owner.op is tensor_from_scalar
    assert t.type.shape == ()
    assert t.type.ndim == 0
    assert t.type.dtype == s.type.dtype

    v = eval_outputs([t])

    assert v == 56, v
    assert isinstance(v, np.ndarray)
    assert v.shape == (), v.shape

    g = grad(t, s)
    assert eval_outputs([g]) == 0.0

    with pytest.raises(TypeError):
        tensor_from_scalar(vector())


@pytest.mark.parametrize(
    "cast_policy",
    [
        "custom",
        "numpy+floatX",
    ],
)
def test_ScalarFromTensor(cast_policy):
    with config.change_flags(cast_policy=cast_policy):
        tc = constant(56)  # ps.constant(56)
        ss = scalar_from_tensor(tc)
        assert ss.owner.op is scalar_from_tensor
        assert ss.type.dtype == tc.type.dtype

        v = eval_outputs([ss])

        assert v == 56
        assert v.shape == ()

        if cast_policy == "custom":
            assert isinstance(v, np.int8)
        elif cast_policy == "numpy+floatX":
            assert isinstance(v, np.int64)

        pts = lscalar()
        ss = scalar_from_tensor(pts)
        ss.owner.op.grad([pts], [ss])
        fff = function([pts], ss)
        v = fff(np.asarray(5))
        assert v == 5
        assert isinstance(v, np.int64)
        assert v.shape == ()

        with pytest.raises(TypeError):
            scalar_from_tensor(vector())


def test_op_cache():
    # TODO: What is this actually testing?
    # trigger bug in ticket #162
    v = matrix()
    v.name = "v"
    gv = fill(v / v, 1.0) / v - (fill(v / v, 1.0) * v) / (v * v)
    fn_py = inplace_func([v], gv)
    fn_c_or_py = inplace_func([v], gv)

    a = random(5, 2).astype(config.floatX)
    assert np.all(fn_py(a) == fn_c_or_py(a))


def test_dimshuffle():
    # The goal of the operation made by `b` is to ensure the second dimension
    # of the column matrix is broadcastable.
    a = dmatrix()
    b = a.reshape((a.shape[0],)).dimshuffle(0, "x")
    f = function([a], b)
    assert (f(np.zeros((3, 1))) + np.ones(2) == np.ones((3, 2))).all()


def test_flatten_ndim_default():
    a = dmatrix()
    c = flatten(a)
    f = inplace_func([a], c)
    a_val = np.asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")
    c_val = np.asarray([0, 1, 2, 3, 4, 5], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    utt.verify_grad(flatten, [a_val])


def test_flatten_scalar():
    a = dscalar()
    c = flatten(a)
    f = inplace_func([a], c)
    a_val = np.asarray(3.0, dtype="float64")
    c_val = np.asarray([3.0], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    # utt.verify_grad(flatten, [a_val]) #TODO: fix verify_grd to work on scalars


def test_flatten_ndim1():
    a = dmatrix()
    c = flatten(a, 1)
    f = inplace_func([a], c)
    a_val = np.asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")
    c_val = np.asarray([0, 1, 2, 3, 4, 5], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    utt.verify_grad(flatten, [a_val])


def test_flatten_ndim2():
    a = dmatrix()
    c = flatten(a, 2)
    f = inplace_func([a], c)
    a_val = np.asarray([[0, 1, 2], [3, 4, 5]], dtype="float64")
    assert np.all(f(a_val) == a_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == a_val)

    flatten_2 = partial(flatten, ndim=2)
    utt.verify_grad(flatten_2, [a_val])


def test_flatten_ndim2_of_3():
    a = TensorType("float64", shape=(None, None, None))()
    c = flatten(a, 2)
    f = inplace_func([a], c)
    a_val = np.asarray([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype="float64")
    c_val = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype="float64")
    assert np.all(f(a_val) == c_val)
    f = inplace_func([a], c)
    assert np.all(f(a_val) == c_val)

    flatten_2 = partial(flatten, ndim=2)
    utt.verify_grad(flatten_2, [a_val])


def test_flatten_broadcastable():
    # Ensure that the broadcastable pattern of the output is coherent with
    # that of the input

    inp = TensorType("float64", shape=(None, None, None, None))()
    out = flatten(inp, ndim=2)
    assert out.type.shape == (None, None)

    inp = TensorType("float64", shape=(None, None, None, 1))()
    out = flatten(inp, ndim=2)
    assert out.type.shape == (None, None)

    inp = TensorType("float64", shape=(None, 1, None, 1))()
    out = flatten(inp, ndim=2)
    assert out.type.shape == (None, None)

    inp = TensorType("float64", shape=(None, 1, 1, 1))()
    out = flatten(inp, ndim=2)
    assert out.type.shape == (None, 1)

    inp = TensorType("float64", shape=(1, None, 1, 1))()
    out = flatten(inp, ndim=3)
    assert out.type.shape == (1, None, 1)


def test_flatten_ndim_invalid():
    a = dmatrix()
    with pytest.raises(ValueError):
        flatten(a, 3)
    with pytest.raises(ValueError):
        flatten(a, 0)


def test_is_flat():
    # tests is_flat method for constant and symbolic variables,
    # as well as reshaped constant and symbolic variables on the
    # given `ndim`

    # Constant variable
    assert ptb.is_flat(ptb.as_tensor_variable(np.zeros(10)))
    assert ptb.is_flat(ptb.as_tensor_variable(np.zeros((10, 10, 10))), ndim=3)
    assert not ptb.is_flat(ptb.as_tensor_variable(np.zeros((10, 10, 10))))

    # Symbolic variable
    assert ptb.is_flat(vector())
    assert ptb.is_flat(tensor3(), ndim=3)
    assert not ptb.is_flat(tensor3())

    # Reshape with constant shape
    X = tensor4()
    assert ptb.is_flat(X.reshape((-1,)))
    assert ptb.is_flat(X.reshape((10, 10, -1)), ndim=3)
    assert not ptb.is_flat(X.reshape((10, 10, -1)))

    # Reshape with symbolic shape
    X = tensor4()
    assert ptb.is_flat(X.reshape((iscalar(),)))
    assert ptb.is_flat(X.reshape((iscalar(),) * 3), ndim=3)
    assert not ptb.is_flat(X.reshape((iscalar(),) * 3))


class TestTile:
    @pytest.mark.parametrize(
        "A_shape, reps_test",
        [
            ((), (2,)),
            ((5,), (2,)),
            ((2, 4), (2, 3)),
            ((2, 4), (2, 3, 4)),
            ((2, 4, 3), (2, 3)),
            ((2, 4, 3), (2, 3, 4)),
            ((2, 4, 3, 5), (2, 3, 4, 6)),
        ],
    )
    def test_tile_separate_reps_entries(self, A_shape, reps_test):
        rng = np.random.default_rng(2400)

        A = tensor("A", shape=(None,) * len(A_shape))
        reps = [iscalar(f"r{i}") for i in range(len(reps_test))]
        tile_out = tile(A, reps)

        tile_fn = function([A, *reps], tile_out)

        A_test = rng.standard_normal(A_shape).astype(config.floatX)
        np.testing.assert_array_equal(
            tile_fn(A_test, *reps_test),
            np.tile(A_test, reps_test),
            strict=True,
        )

    @pytest.mark.parametrize("reps", (2, np.array([2, 3, 4])))
    def test_combined_reps_entries(self, reps):
        rng = np.random.default_rng(2422)
        A_test = rng.standard_normal((2, 4, 3)).astype(config.floatX)
        expected_eval = np.tile(A_test, reps)

        A = tensor3("A")
        np.testing.assert_array_equal(
            tile(A, reps).eval({A: A_test}),
            expected_eval,
            strict=True,
        )

        sym_reps = as_tensor_variable(reps).type()
        np.testing.assert_array_equal(
            tile(A, sym_reps).eval({A: A_test, sym_reps: reps}),
            expected_eval,
            strict=True,
        )

    def test_mixed_reps_type(self):
        A = np.arange(9).reshape(3, 3)
        reps = [2, iscalar("3"), 4]
        np.testing.assert_array_equal(
            tile(A, reps).eval({"3": 3}),
            np.tile(A, [2, 3, 4]),
            strict=True,
        )

    def test_tensorlike_A(self):
        # Test when x is a list
        x_val = [[1.0, 2.0], [3.0, 4.0]]
        assert equal_computations(
            [tile(x_val, (2,))],
            [tile(as_tensor_variable(x_val), (2,))],
        )

    def test_error_unknown_reps_length(self):
        # error raising test: ndim not specified when reps is vector
        reps = ivector()
        with pytest.raises(ValueError, match="Use specify_shape to set the length"):
            tile(arange(3), reps)

        # fine with specify_shape
        out = tile(arange(3), specify_shape(reps, 2))
        np.testing.assert_array_equal(
            out.eval({reps: [2, 3]}),
            np.tile(np.arange(3), [2, 3]),
            strict=True,
        )

    def test_error_non_integer_reps(self):
        for reps in (
            2.5,
            fscalar(),
            vector(shape=(3,), dtype="float64"),
            [2, fscalar()],
        ):
            with pytest.raises(ValueError):
                tile(arange(3), reps)

    def test_error_reps_ndim(self):
        for reps in (
            matrix(shape=(3, 1), dtype=int),
            [2, vector(shape=(2,), dtype=int)],
        ):
            with pytest.raises(ValueError):
                tile(arange(3), reps)

    def test_tile_grad(self):
        A = tensor3("A")
        reps = vector("reps", shape=(3,), dtype=int)
        A_tile = tile(A, reps)
        grad_tile = grad(A_tile.sum(), A)

        # The gradient should be the product of the tiling dimensions
        # (since the gradients are additive through the tiling operation)
        rng = np.random.default_rng(2489)
        A_test = rng.normal(size=(2, 4, 3)).astype(config.floatX)
        reps_test = [3, 4, 5]
        np.testing.assert_array_equal(
            grad_tile.eval({A: A_test, reps: reps_test}),
            np.full(A_test.shape, np.prod(reps_test).astype(config.floatX)),
            strict=True,
        )


class TestARange:
    def test_Op_integers(self):
        # Test behaviour of ARange Op on integer inputs
        start, stop, step = iscalars("start", "stop", "step")
        out = ARange(start.type.dtype)(start, stop, step)
        f = function([start, stop, step], out)

        assert np.all(f(0, 5, 1) == np.arange(0, 5, 1))
        assert np.all(f(2, 11, 4) == np.arange(2, 11, 4))
        assert np.all(f(-5, 1, 1) == np.arange(-5, 1, 1))
        assert np.all(f(10, 2, -2) == np.arange(10, 2, -2))
        assert np.all(f(10, 2, 2) == np.arange(10, 2, 2))
        assert np.all(f(0, 0, 1) == np.arange(0, 0, 1))

    def test_grads(self):
        def f(start, stop, step):
            return ARange(start.type.dtype)(start, stop, step)

        rng = np.random.default_rng(utt.fetch_seed())
        # Due to the random projection, we should not use the exact
        # point that change the shape of the output.
        for start, stop, step in [(0, 4.9, 1), (5.1, 0, -0.5), (1, 5.1, 0.5)]:
            utt.verify_grad(
                f,
                [
                    np.asarray(start).astype(config.floatX),
                    np.asarray(stop).astype(config.floatX),
                    np.asarray(step).astype(config.floatX),
                ],
                rng=rng,
            )

    @pytest.mark.parametrize(
        "cast_policy",
        [
            "custom",
            "numpy+floatX",
        ],
    )
    def test_integers(self, cast_policy):
        """Test arange constructor, on integer outputs."""
        with config.change_flags(cast_policy=cast_policy):
            start, stop, step = iscalars("start", "stop", "step")
            out = arange(start, stop, step)
            f = function([start, stop, step], out)

            if cast_policy == "custom":
                assert out.dtype == "int64"
            elif cast_policy == "numpy+floatX":
                numpy_dtype = np.arange(np.array(1, dtype="int32")).dtype
                assert out.dtype == numpy_dtype

            assert np.all(f(0, 5, 1) == np.arange(0, 5, 1))
            assert np.all(f(2, 11, 4) == np.arange(2, 11, 4))
            assert np.all(f(-5, 1, 1) == np.arange(-5, 1, 1))
            assert np.all(f(10, 2, -2) == np.arange(10, 2, -2))
            assert np.all(f(10, 2, 2) == np.arange(10, 2, 2))
            assert np.all(f(0, 0, 1) == np.arange(0, 0, 1))

    @pytest.mark.parametrize(
        "cast_policy",
        [
            "custom",
            "numpy+floatX",
        ],
    )
    def test_float32(self, cast_policy):
        """Test arange constructor, on float32 outputs."""
        with config.change_flags(cast_policy=cast_policy):
            start, stop, step = fscalars("start", "stop", "step")
            out = arange(start, stop, step)
            f = function([start, stop, step], out)

            if config.cast_policy == "custom":
                assert out.dtype == start.type.dtype
            elif config.cast_policy == "numpy+floatX":
                assert out.dtype == config.floatX

            arg_vals = [
                (0, 5, 1),
                (2, 11, 4),
                (-5, 1.1, 1.2),
                (1.3, 2, -2.1),
                (10, 2, 2),
            ]
            for arg_v in arg_vals:
                start_v, stop_v, step_v = arg_v
                start_v_, stop_v_, step_v_ = np.asarray(arg_v, dtype=start.type.dtype)
                f_val = f(start_v_, stop_v_, step_v_)

                if config.cast_policy == "custom":
                    expected_val = np.arange(
                        start_v, stop_v, step_v, dtype=start.type.dtype
                    )
                elif config.cast_policy == "numpy+floatX":
                    expected_val = np.arange(
                        start_v_, stop_v_, step_v_, dtype=out.dtype
                    )

                assert np.all(f_val == expected_val)

    @pytest.mark.parametrize(
        "cast_policy",
        [
            "custom",
            "numpy+floatX",
        ],
    )
    def test_float64(self, cast_policy):
        """Test arange constructor, on float64 outputs."""
        with config.change_flags(cast_policy=cast_policy):
            start, stop, step = dscalars("start", "stop", "step")
            out = arange(start, stop, step)
            f = function([start, stop, step], out)

            assert out.dtype == start.type.dtype

            arg_vals = [
                (0, 5, 1),
                (2, 11, 4),
                (-5, 1.1, 1.2),
                (1.3, 2, -2.1),
                (10, 2, 2),
            ]
            for arg_v in arg_vals:
                start_v, stop_v, step_v = arg_v
                start_v_, stop_v_, step_v_ = np.asarray(arg_v, dtype=start.type.dtype)
                f_val = f(start_v_, stop_v_, step_v_)

                if config.cast_policy == "custom":
                    expected_val = np.arange(
                        start_v, stop_v, step_v, dtype=start.type.dtype
                    )
                elif config.cast_policy == "numpy+floatX":
                    expected_val = np.arange(start_v_, stop_v_, step_v_)

                assert np.all(f_val == expected_val)

    @pytest.mark.parametrize(
        "cast_policy",
        [
            "custom",
            "numpy+floatX",
        ],
    )
    def test_default_step(self, cast_policy):
        """Test that arange constructor uses the correct default step."""
        with config.change_flags(cast_policy=cast_policy):
            start, stop = iscalars("start", "stop")
            out = arange(start, stop)
            f = function([start, stop], out)

            if config.cast_policy == "custom":
                assert out.dtype == "int64"
            elif config.cast_policy == "numpy+floatX":
                assert out.dtype == np.arange(np.int32(0), np.int32(1)).dtype

            assert np.all(f(0, 5) == np.arange(0, 5))
            assert np.all(f(-5, 1) == np.arange(-5, 1))
            assert np.all(f(0, 0) == np.arange(0, 0))

            dstart, dstop = dscalars("start", "stop")
            dout = arange(dstart, dstop)
            df = function([dstart, dstop], dout)

            assert dout.dtype == dstart.type.dtype
            # print df(0.2, 5.3)
            # print np.arange(0.2, 5.3)
            assert np.all(df(0.2, 5.3) == np.arange(0.2, 5.3))
            assert np.all(df(0.8, 5.3) == np.arange(0.8, 5.3))
            assert np.all(df(-0.7, 5.3) == np.arange(-0.7, 5.3))

    @pytest.mark.parametrize(
        "cast_policy",
        [
            "custom",
            "numpy+floatX",
        ],
    )
    def test_default_start(self, cast_policy):
        """Test that arange constructor uses the correct default start."""
        with config.change_flags(cast_policy=cast_policy):
            stop = iscalar("stop")
            out = arange(stop)
            f = function([stop], out)

            if config.cast_policy == "custom":
                assert out.dtype == "int64"
            elif config.cast_policy == "numpy+floatX":
                assert out.dtype == np.arange(np.int32(1)).dtype

            assert np.all(f(8) == np.arange(8))
            assert np.all(f(-2) == np.arange(-2))

            fstop = fscalar("stop")
            fout = arange(fstop)
            ff = function([fstop], fout)

            if config.cast_policy == "custom":
                assert fout.dtype == fstop.type.dtype
            elif config.cast_policy == "numpy+floatX":
                if config.floatX == "float32":
                    assert fout.dtype == "float32"
                else:
                    assert fout.dtype == np.arange(np.float32(1)).dtype

            fstop_values = [0.2, -0.7, 8.5]
            for fstop_v in fstop_values:
                fstop_v32 = np.float32(fstop_v)
                assert np.all(ff(fstop_v32) == np.arange(fstop_v))

    @config.change_flags(cast_policy="custom")
    def test_upcast_custom(self):
        """Test that arange computes output type adequately."""
        assert arange(iscalar()).dtype == "int64"
        assert arange(fscalar()).dtype == fscalar().dtype
        assert arange(dscalar()).dtype == dscalar().dtype

        # int32 + float32 -> float64
        assert arange(iscalar(), fscalar()).dtype == dscalar().dtype
        assert arange(iscalar(), dscalar()).dtype == dscalar().dtype
        assert arange(fscalar(), dscalar()).dtype == dscalar().dtype

        assert arange(iscalar(), fscalar(), dscalar()).dtype == dscalar().dtype

    @pytest.mark.parametrize(
        "dtype", [dtype for dtype in ALL_DTYPES if not dtype.startswith("complex")]
    )
    @pytest.mark.parametrize(
        "stop_dtype", [dtype for dtype in ALL_DTYPES if not dtype.startswith("complex")]
    )
    @config.change_flags(cast_policy="numpy+floatX")
    def test_upcast_numpy(self, dtype, stop_dtype):
        """Make sure our `ARange` output dtypes match NumPy's under different casting policies."""
        # Test with a single argument.
        arange_dtype = arange(scalar(dtype=str(dtype))).dtype
        numpy_dtype = np.arange(np.array(1, dtype=dtype)).dtype
        if (
            dtype != "float64"
            and numpy_dtype == "float64"
            and config.cast_policy == "numpy+floatX"
            and config.floatX == "float32"
        ):
            # We want a float32 arange.
            assert arange_dtype == "float32"
        else:
            # Follow numpy.
            assert arange_dtype == numpy_dtype

        # Test with two arguments.
        arange_dtype = arange(
            start=scalar(dtype=str(dtype)),
            stop=scalar(dtype=str(stop_dtype)),
        ).dtype
        numpy_dtype = np.arange(
            start=np.array(0, dtype=dtype),
            stop=np.array(1, dtype=stop_dtype),
        ).dtype

        if (
            dtype != "float64"
            and stop_dtype != "float64"
            and numpy_dtype == "float64"
            and config.cast_policy == "numpy+floatX"
            and config.floatX == "float32"
        ):
            # We want a float32 arange.
            assert arange_dtype == "float32"
        else:
            # Follow numpy.
            assert arange_dtype == numpy_dtype

    def test_dtype_cache(self):
        """Check that the same `Op` is returned on repeated calls to `ARange` using the same dtype."""

        start, stop, step = iscalars("start", "stop", "step")
        out1 = arange(start, stop, step)
        out2 = arange(start, stop, step, dtype=out1.dtype)
        out3 = arange(start, stop, 2.0, dtype=out1.dtype)
        out4 = arange(start, stop, 2.0)

        assert out1.owner.op is out2.owner.op
        assert out2.owner.op is out3.owner.op
        assert out3.owner.op is not out4.owner.op

    @pytest.mark.parametrize(
        "cast_policy",
        [
            "custom",
            "numpy+floatX",
        ],
    )
    def test_infer_shape(self, cast_policy):
        with config.change_flags(cast_policy=cast_policy):
            start, stop, step = iscalars("start", "stop", "step")
            out = arange(start, stop, step)
            mode = config.mode
            if mode == "FAST_COMPILE":
                mode = "FAST_RUN"
            mode = compile.mode.get_mode(mode).excluding("fusion")
            f = function([start, stop, step], out.shape, mode=mode)
            assert len(f.maker.fgraph.toposort()) == 9

            if config.cast_policy == "custom":
                assert out.dtype == "int64"
            elif config.cast_policy == "numpy+floatX":
                numpy_dtype = np.arange(
                    np.array(0, dtype=start.dtype),
                    np.array(1, dtype=stop.dtype),
                    np.array(1, dtype=step.dtype),
                ).dtype
                assert out.dtype == numpy_dtype

            assert np.all(f(0, 5, 1) == len(np.arange(0, 5, 1)))
            assert np.all(f(2, 11, 4) == len(np.arange(2, 11, 4)))
            assert np.all(f(-5, 1, 1) == len(np.arange(-5, 1, 1)))
            assert np.all(f(10, 2, -2) == len(np.arange(10, 2, -2)))
            assert np.all(f(10, 2, 2) == len(np.arange(10, 2, 2)))
            assert np.all(f(0, 0, 1) == len(np.arange(0, 0, 1)))

            out = arange(start, stop, 1)
            f = function([start, stop], out.shape, mode=mode)
            assert len(f.maker.fgraph.toposort()) == 5
            # 4 [Elemwise{sub,no_inplace}(stop, start), Elemwise{Cast{int64}}(Elemwise{sub,no_inplace}.0), Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)](Elemwise{Cast{int64}}.0, 0), MakeVector(Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)].0)]
            if config.cast_policy == "custom":
                assert out.dtype == "int64"
            elif config.cast_policy == "numpy+floatX":
                assert (
                    out.dtype == np.arange(np.int32(0), np.int32(1), np.int32(1)).dtype
                )

            assert np.all(f(0, 5) == len(np.arange(0, 5)))
            assert np.all(f(2, 11) == len(np.arange(2, 11)))
            assert np.all(f(-5, 1) == len(np.arange(-5, 1)))
            assert np.all(f(10, 2) == len(np.arange(10, 2)))
            assert np.all(f(10, 2) == len(np.arange(10, 2)))
            assert np.all(f(0, 0) == len(np.arange(0, 0)))
            assert np.all(f(-64, 64) == len(np.arange(-64, 64)))
            assert arange(-64, 64).shape.eval() == [128]
            assert arange(-64, 64, 2).shape.eval() == [64]

            out = arange(0, stop, 1)
            f = function([stop], out.shape, mode=mode)
            assert len(f.maker.fgraph.toposort()) == 2
            # [Elemwise{Cast{int64}}(stop), MakeVector(Elemwise{Cast{int64}}.0)]

            if config.cast_policy == "custom":
                assert out.dtype == "int64"
            elif config.cast_policy == "numpy+floatX":
                numpy_dtype = np.arange(0, np.array(1, dtype=stop.dtype), 1).dtype
                assert out.dtype == numpy_dtype

            assert np.all(f(5) == len(np.arange(0, 5)))
            assert np.all(f(11) == len(np.arange(0, 11)))
            assert np.all(f(1) == len(np.arange(0, 1)))
            assert np.all(f(2) == len(np.arange(0, 2)))
            assert np.all(f(2) == len(np.arange(0, 2)))
            assert np.all(f(0) == len(np.arange(0, 0)))

    def test_static_shape(self):
        assert np.arange(1, 10).shape == arange(1, 10).type.shape
        assert np.arange(10, 1, -1).shape == arange(10, 1, -1).type.shape
        assert np.arange(1, -9, 2).shape == arange(1, -9, 2).type.shape
        assert np.arange(1.3, 17.48, 2.67).shape == arange(1.3, 17.48, 2.67).type.shape
        assert np.arange(-64, 64).shape == arange(-64, 64).type.shape

    def test_c_cache_bug(self):
        # Regression test for bug caused by issues in hash of `np.dtype()` objects
        # https://github.com/numpy/numpy/issues/17864
        end = iscalar("end")
        arange1 = ARange(np.dtype("float64"))(0, end, 1)
        arange2 = ARange("float64")(0, end + 1, 1)
        assert arange1.owner.op == arange2.owner.op
        assert hash(arange1.owner.op) == hash(arange2.owner.op)
        fn = function([end], [arange1, arange2])
        res1, res2 = fn(10)
        np.testing.assert_array_equal(res1, res2[:-1], strict=True)


class TestNdGrid:
    def setup_method(self):
        pass

    def test_mgrid_numpy_equiv(self):
        nmgrid = (
            [np.mgrid[0:1:0.1]],
            np.mgrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            np.mgrid[0:2:1, 1:10:1, 10:100:10],
        )
        tmgrid = (
            [mgrid[0:1:0.1]],
            mgrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            mgrid[0:2:1, 1:10:1, 10:100:10],
        )
        for n, t in zip(nmgrid, tmgrid, strict=True):
            for ng, tg in zip(n, t, strict=True):
                utt.assert_allclose(ng, tg.eval())

    def test_ogrid_numpy_equiv(self):
        nogrid = (
            [np.ogrid[0:1:0.1]],
            np.ogrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            np.ogrid[0:2:1, 1:10:1, 10:100:10],
        )
        togrid = (
            [ogrid[0:1:0.1]],
            ogrid[0:1:0.1, 1:10:1.0, 10:100:10.0],
            ogrid[0:2:1, 1:10:1, 10:100:10],
        )
        for n, t in zip(nogrid, togrid, strict=True):
            for ng, tg in zip(n, t, strict=True):
                utt.assert_allclose(ng, tg.eval())

    def test_mgrid_pytensor_variable_numpy_equiv(self):
        nfmgrid = np.mgrid[0:1:0.1, 1:10:1.0, 10:100:10.0]
        nimgrid = np.mgrid[0:2:1, 1:10:1, 10:100:10]
        i, j, k = dscalars("i", "j", "k")
        l, m, n = iscalars("l", "m", "n")
        tfmgrid = mgrid[i:1:0.1, 1:j:1.0, 10:100:k]
        timgrid = mgrid[l:2:1, 1:m:1, 10:100:n]
        ff = pytensor.function([i, j, k], tfmgrid)
        fi = pytensor.function([l, m, n], timgrid)
        for n, t in zip(
            (nfmgrid, nimgrid), (ff(0, 10, 10.0), fi(0, 10, 10)), strict=True
        ):
            for ng, tg in zip(n, t, strict=True):
                utt.assert_allclose(ng, tg)

    def test_ogrid_pytensor_variable_numpy_equiv(self):
        nfogrid = np.ogrid[0:1:0.1, 1:10:1.0, 10:100:10.0]
        niogrid = np.ogrid[0:2:1, 1:10:1, 10:100:10]
        i, j, k = dscalars("i", "j", "k")
        l, m, n = iscalars("l", "m", "n")
        tfogrid = ogrid[i:1:0.1, 1:j:1.0, 10:100:k]
        tiogrid = ogrid[l:2:1, 1:m:1, 10:100:n]
        ff = pytensor.function([i, j, k], tfogrid)
        fi = pytensor.function([l, m, n], tiogrid)
        for n, t in zip(
            (nfogrid, niogrid), (ff(0, 10, 10.0), fi(0, 10, 10)), strict=True
        ):
            for ng, tg in zip(n, t, strict=True):
                utt.assert_allclose(ng, tg)


class TestInversePermutation:
    def test_dim1(self):
        # Test the inversion of one permutation (int vector)
        p = ivector()
        inv = inverse_permutation(p)
        assert inv.dtype == p.dtype
        f_inverse = function([p], inv)

        # Generate a random permutation
        rng = np.random.default_rng(utt.fetch_seed())
        p_val = rng.permutation(10).astype("int32")
        inv_val = f_inverse(p_val)

        # Check that the inverse of the inverse is the original permutation
        assert np.all(f_inverse(inv_val) == p_val)
        # Check that permutation(inverse) == inverse(permutation) = identity
        assert np.all(p_val[inv_val] == np.arange(10))
        assert np.all(inv_val[p_val] == np.arange(10))

        # Test passing a list
        p = [2, 4, 3, 0, 1]
        inv = ptb.inverse_permutation(p)
        f = pytensor.function([], inv)
        assert np.array_equal(f(), np.array([3, 4, 0, 2, 1]))

    def test_dim2(self):
        # Test the inversion of several permutations at a time
        # Each row of p is a different permutation to inverse
        p = imatrix()
        inv = inverse_permutation(p)
        f_inverse = function([p], inv)

        rng = np.random.default_rng(utt.fetch_seed())
        # Generate 10 random permutations
        p_val = np.asarray([rng.permutation(10) for i in range(7)], dtype="int32")
        inv_val = f_inverse(p_val)

        # Check that the inverse of the inverse is the original permutation list
        assert np.all(f_inverse(inv_val) == p_val)
        # Check that, for each permutation,
        # permutation(inverse) == inverse(permutation) = identity
        for p_row, i_row in zip(p_val, inv_val, strict=True):
            assert np.all(p_row[i_row] == np.arange(10))
            assert np.all(i_row[p_row] == np.arange(10))


class TestPermuteRowElements:
    def test_1_1(self):
        # Test PermuteRowElements(vector, vector)
        input = dvector()
        p = ivector()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(5,))
        p_val = rng.permutation(5).astype("int32")
        out_val = permute(input_val, p_val)

        # Should be equivalent to advanced indexing
        out_bis = input_val[p_val]
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_2_1(self):
        # Test broadcasting in PermuteRowElements(matrix, vector)
        input = matrix()
        p = ivector()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(3, 5)).astype(config.floatX)
        p_val = rng.permutation(5).astype("int32")
        out_val = permute(input_val, p_val)

        # The same permutation should be applied to every row of the input matrix.
        out_bis = np.asarray([r[p_val] for r in input_val])
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_2_2(self):
        # Test PermuteRowElements(matrix, matrix)
        input = matrix()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(3, 5)).astype(config.floatX)
        p_val = np.asarray([rng.permutation(5) for i in range(3)], dtype="int32")
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to the corresponding
        # row of input
        out_bis = np.asarray(
            [i_row[p_row] for i_row, p_row in zip(input_val, p_val, strict=True)]
        )
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_1_2(self):
        # Test PermuteRowElements(vector, matrix)
        # Different permutations will be applied to the same input vector
        input = vector()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(5,)).astype(config.floatX)
        p_val = np.asarray([rng.permutation(5) for i in range(3)], dtype="int32")
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to the input vector
        out_bis = np.asarray([input_val[p_row] for p_row in p_val])
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])

    def test_3b_2(self):
        # Test permute_row_elements on a more complex broadcasting pattern:
        # input.type.shape = (None, 1, None),
        # p.type.shape = (None, None).

        input = TensorType("floatX", shape=(None, 1, None))()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = np.random.default_rng(utt.fetch_seed())
        input_val = rng.uniform(size=(4, 1, 5)).astype(config.floatX)
        p_val = np.asarray([rng.permutation(5) for i in range(3)], dtype="int32")
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to each row
        # of the input tensor
        out_bis = np.asarray(
            [[in_mat[0, p_row] for p_row in p_val] for in_mat in input_val]
        )
        assert np.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            # Auxiliary op defined to get rid of gradient wrt p_val
            return permute_row_elements(s_input, p_val)

        utt.verify_grad(permute_fixed, [input_val])


def test_stack():
    sx, sy = dscalar(), dscalar()

    rval = inplace_func([sx, sy], stack([sx, sy]))(-4.0, -2.0)
    assert type(rval) is np.ndarray
    assert [-4, -2] == list(rval)


@pytest.mark.skipif(
    isinstance(get_default_mode(), pytensor.compile.debugmode.DebugMode),
    reason="This test fails in DEBUG_MODE, but the generated code is OK. "
    "It is actually a problem of DEBUG_MODE, see #626.",
)
def test_default():
    x, y = scalars("xy")
    z = default(x, y)
    f = function([x, y], z)
    assert f(1, 2) == 1
    assert f(None, 2) == 2
    assert f(1, None) == 1

    with pytest.raises(TypeError, match=".*compatible types.*"):
        default(x, vector())


@pytest.mark.skipif(
    isinstance(get_default_mode(), pytensor.compile.debugmode.DebugMode),
    reason="This test fails in DEBUG_MODE, but the generated code is OK. "
    "It is actually a problem of DEBUG_MODE, see #626.",
)
def test_default_state():
    x, y = scalars("xy")
    # print config.floatX
    # print x.type
    # print y.type
    z = default(x, 3.8)
    new_x = y + z
    f = function([y, compile.In(x, update=new_x, value=12.0)], new_x)
    assert f(3) == 15
    f["x"] = None
    assert np.allclose(f(1), 4.8)
    assert np.allclose(f(np.asarray(2.2, dtype=config.floatX)), 7)


@config.change_flags(cast_policy="custom")
def test_autocast_custom():
    # Called from `test_autocast`.
    assert config.cast_policy == "custom"
    orig_autocast = autocast_float.dtypes

    # Test that autocast_float_as sets the autocast dtype correctly
    with autocast_float_as("float32"):
        assert autocast_float.dtypes == ("float32",)
    assert autocast_float.dtypes == orig_autocast

    with autocast_float_as("float64"):
        assert autocast_float.dtypes == ("float64",)
    assert autocast_float.dtypes == orig_autocast

    # Test that we can set it back to something, and nest it
    with autocast_float_as("float32"):
        assert autocast_float.dtypes == ("float32",)
        with autocast_float_as("float64"):
            assert autocast_float.dtypes == ("float64",)
        assert autocast_float.dtypes == ("float32",)
    assert autocast_float.dtypes == orig_autocast

    # Test that the autocasting dtype is used correctly in expression-building
    with autocast_float_as("float32"):
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.1).dtype == "float32"
        assert (fvector() + np.asarray(1.1, dtype="float64")).dtype == "float64"
        assert (fvector() + np.asarray(1.1, dtype="float32")).dtype == "float32"

        assert (dvector() + 1).dtype == "float64"
        assert (fvector() + 1).dtype == "float32"

    # Test that the autocasting dtype is used correctly in expression-building
    with autocast_float_as("float64"):
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.0).dtype == "float64"
        assert (fvector() + np.asarray(1.1, dtype="float64")).dtype == "float64"
        assert (fvector() + np.asarray(1.1, dtype="float32")).dtype == "float32"

        assert (dvector() + 1).dtype == "float64"
        assert (fvector() + 1).dtype == "float32"

    # Test that the autocasting dtype is used correctly in expression-building
    with autocast_float_as("float32", "float64"):
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + 1.1).dtype == config.floatX
        assert (fvector() + 1.0).dtype == "float32"
        assert (dvector() + np.float32(1.1)).dtype == "float64"
        assert (dvector() + np.float64(1.1)).dtype == "float64"
        assert (dvector() + 1.1).dtype == "float64"
        assert (fvector() + np.float32(1.1)).dtype == "float32"
        assert (fvector() + np.float64(1.1)).dtype == "float64"
        assert (lvector() + np.int64(1)).dtype == "int64"
        assert (lvector() + np.int32(1)).dtype == "int64"
        assert (lvector() + np.int16(1)).dtype == "int64"
        assert (lvector() + np.int8(1)).dtype == "int64"
        assert (ivector() + np.int8(1)).dtype == "int32"
        assert (wvector() + np.int8(1)).dtype == "int16"
        assert (bvector() + np.int8(1)).dtype == "int8"
        with autocast_float_as("float64"):
            assert (fvector() + 1.0).dtype == "float64"


@pytest.mark.skip(reason="Not implemented")
@config.change_flags(cast_policy="numpy")
def test_autocast_numpy():
    # Called from `test_autocast`.
    assert config.cast_policy == "numpy"
    # Go through some typical scalar values.

    def ok(z):
        assert constant(z).dtype == np.asarray(z).dtype

    for x in [2**i for i in range(63)] + [0, 0, 1, 2**63 - 1] + [0.0, 1.0, 1.1, 1.5]:
        n_x = np.asarray(x)
        # Make sure the data type is the same as the one found by numpy.
        ok(x)
        ok(-x)
        ok(x - 1)
        ok(-x + 1)
        ok(n_x)


@config.change_flags(cast_policy="numpy+floatX")
def test_autocast_numpy_floatX():
    # Called from `test_autocast`.
    assert config.cast_policy == "numpy+floatX"

    def ok(z, floatX):
        if isinstance(z, float) and floatX == "float32" and not hasattr(z, "dtype"):
            # Special case where we use 'float32' instead of 'float64'.
            assert constant(z).dtype == "float32"
        else:
            assert constant(z).dtype == np.asarray(z).dtype

    # Test with various values of `config.floatX`.
    for floatX in ("float32", "float64"):
        # Go through some typical scalar values.
        # We only consider 'int' and 'long' Python values that can fit
        # into int64, as that is the maximal integer type that PyTensor
        # supports, and that is the maximal type in Python indexing.
        for x in (
            [2**i - 1 for i in range(64)] + [0, 0, 1, 2**63 - 1] + [0.0, 1.0, 1.1, 1.5]
        ):
            with config.change_flags(floatX=floatX):
                ok(x, floatX)
                ok(-x, floatX)
                ok(x - 1, floatX)
                ok(-x + 1, floatX)
                ok(np.asarray(x), floatX)
                ok(np.float64(x), floatX)


class TestLongTensor:
    def test_fit_int64(self):
        bitwidth = PYTHON_INT_BITWIDTH
        for exponent in range(bitwidth):
            val = 2**exponent - 1
            scalar_ct = constant(val)

            assert scalar_ct.dtype in int_dtypes, (
                exponent,
                val,
                scalar_ct.dtype,
            )
            assert scalar_ct.value == val

            vector_ct = constant([val, val])
            # On Python 2, np.array() on a "long" returns int64,
            # but on Python 3, all integers are long, and np.asarray
            # will not force the upcasting, and return the native int width.
            if bitwidth == 32:
                assert vector_ct.dtype == "int32"
            else:
                assert vector_ct.dtype == "int64"
            assert np.all(vector_ct.value == val)

            matrix_ct = constant([[val, val]])
            # On Python 2, np.array() on a "long" returns int64,
            # but on Python 3, all integers are long, and np.asarray
            # will not force the upcasting, and return the native int width.
            if bitwidth == 32:
                assert matrix_ct.dtype == "int32"
            else:
                assert matrix_ct.dtype == "int64"
            assert np.all(matrix_ct.value == val)

    def test_too_big(self):
        val = 2**64
        # This fail for all NumPy version.
        with pytest.raises(Exception):
            constant(val)
        with pytest.raises(Exception):
            constant()[val, val]
        with pytest.raises(Exception):
            constant()[[val, val]]


def test_len():
    for shape_ in [(5,), (3, 4), (7, 4, 6)]:
        x = tensor(dtype="floatX", shape=(None,) * len(shape_))
        with pytest.raises(TypeError):
            len(x)


def test_unalign():
    if config.floatX == "float64":
        dtype = "b1,f8"
    else:
        dtype = "b1,f4"

    a = np.empty(10000, dtype=dtype)["f1"]
    b = np.empty(10000, dtype=dtype)["f1"]
    assert not a.flags.aligned
    assert not b.flags.aligned
    a[:] = random(len(a))
    b[:] = random(len(b))
    # out_numpy = 2 * a + 3 * b

    av, bv = vectors("ab")
    f = pytensor.function([av, bv], 2 * av + 3 * bv)
    f.maker.fgraph.toposort()

    with pytest.raises(TypeError):
        f(a, b)

    a = np.empty((), dtype=dtype)["f1"]
    b = np.empty((), dtype=dtype)["f1"]
    assert not a.flags.aligned
    assert not b.flags.aligned
    # out_numpy = 2 * a + 3 * b

    av, bv = scalars("ab")
    f = pytensor.function([av, bv], 2 * av + 3 * bv)
    f.maker.fgraph.toposort()
    with pytest.raises(TypeError):
        f(a, b)


def test_dimshuffle_duplicate():
    x = vector()
    with pytest.raises(ValueError, match="may not appear twice"):
        DimShuffle(input_ndim=1, new_order=(0, 0))(x)


class TestGetUnderlyingScalarConstantValue:
    def test_basic(self):
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(ps.int64())

        res = get_underlying_scalar_constant_value(ptb.as_tensor(10))
        assert res == 10
        assert isinstance(res, np.ndarray)

        res = get_underlying_scalar_constant_value(np.array(10))
        assert res == 10
        assert isinstance(res, np.ndarray)

        a = ptb.stack([1, 2, 3])
        assert get_underlying_scalar_constant_value(a[0]) == 1
        assert get_underlying_scalar_constant_value(a[1]) == 2
        assert get_underlying_scalar_constant_value(a[2]) == 3

        b = iscalar()
        a = ptb.stack([b, 2, 3])
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(a[0])
        assert get_underlying_scalar_constant_value(a[1]) == 2
        assert get_underlying_scalar_constant_value(a[2]) == 3

        # For now get_underlying_scalar_constant_value goes through only MakeVector and Join of
        # scalars.
        v = ivector()
        a = ptb.stack([v, [2], [3]])
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(a[0])
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(a[1])
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(a[2])

        # Test the case SubTensor(Shape(v)) when the dimensions
        # is broadcastable.
        v = row()
        assert get_underlying_scalar_constant_value(v.shape[0]) == 1

        res = ptb.get_underlying_scalar_constant_value(ptb.as_tensor([10, 20]).shape[0])
        assert isinstance(res, np.ndarray)
        assert 2 == res

        res = ptb.get_underlying_scalar_constant_value(
            9 + ptb.as_tensor([1.0]).shape[0],
            elemwise=True,
            only_process_constants=False,
            max_recur=9,
        )
        assert isinstance(res, np.ndarray)
        assert 10 == res

    @pytest.mark.xfail(reason="Incomplete implementation")
    def test_DimShufle(self):
        a = as_tensor_variable(1.0)[None][0]
        assert get_underlying_scalar_constant_value(a) == 1

    def test_subtensor_of_constant(self):
        c = constant(random(5))
        for i in range(c.value.shape[0]):
            assert get_underlying_scalar_constant_value(c[i]) == c.value[i]
        c = constant(random(5, 5))
        for i in range(c.value.shape[0]):
            for j in range(c.value.shape[1]):
                assert get_underlying_scalar_constant_value(c[i, j]) == c.value[i, j]

    def test_numpy_array(self):
        # Regression test for crash when called on a numpy array.
        assert get_underlying_scalar_constant_value(np.array(3)) == 3
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(np.array([0, 1]))
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(np.array([]))

    def test_make_vector(self):
        mv = make_vector(1, 2, 3)
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(mv)
        assert get_underlying_scalar_constant_value(mv[0]) == 1
        assert get_underlying_scalar_constant_value(mv[1]) == 2
        assert get_underlying_scalar_constant_value(mv[2]) == 3
        assert get_underlying_scalar_constant_value(mv[np.int32(0)]) == 1
        assert get_underlying_scalar_constant_value(mv[np.int64(1)]) == 2
        assert get_underlying_scalar_constant_value(mv[np.uint(2)]) == 3
        t = ps.ScalarType("int64")
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(mv[t()])

    def test_shape_i(self):
        c = constant(np.random.random((3, 4)))
        s = Shape_i(0)(c)
        assert get_underlying_scalar_constant_value(s) == 3
        s = Shape_i(1)(c)
        assert get_underlying_scalar_constant_value(s) == 4
        d = pytensor.shared(np.random.standard_normal((1, 1)), shape=(1, 1))
        f = ScalarFromTensor()(Shape_i(0)(d))
        assert get_underlying_scalar_constant_value(f) == 1

    def test_elemwise(self):
        # We test only for a few elemwise, the list of all supported
        # elemwise are in the fct.
        c = constant(np.random.random())
        s = c + 1
        assert np.allclose(get_underlying_scalar_constant_value(s), c.data + 1)
        s = c - 1
        assert np.allclose(get_underlying_scalar_constant_value(s), c.data - 1)
        s = c * 1.2
        assert np.allclose(get_underlying_scalar_constant_value(s), c.data * 1.2)
        s = c < 0.5
        assert np.allclose(get_underlying_scalar_constant_value(s), int(c.data < 0.5))
        s = ptb.second(c, 0.4)
        assert np.allclose(get_underlying_scalar_constant_value(s), 0.4)

    def test_assert(self):
        # Make sure we still get the constant value if it is wrapped in
        # an Assert.
        c = constant(2)
        x = scalar()

        # condition is always True
        a = Assert()(c, c > 1)
        assert get_underlying_scalar_constant_value(a) == 2

        with config.change_flags(compute_test_value="off"):
            # condition is always False
            a = Assert()(c, c > 2)
            with pytest.raises(NotScalarConstantError):
                get_underlying_scalar_constant_value(a)

        # condition is not constant
        a = Assert()(c, c > x)
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(a)

    def test_second(self):
        # Second should apply when the value is constant but not the shape
        c = constant(np.random.random())
        shp = vector()
        s = ptb.second(shp, c)
        assert get_underlying_scalar_constant_value(s) == c.data

    def test_copy(self):
        # Make sure we do not return a writeable internal storage of a constant,
        # so we cannot change the value of a constant by mistake.
        c = constant(3)
        d = get_scalar_constant_value(c)
        with pytest.raises(ValueError, match="output array is read-only"):
            d += 1
        e = get_scalar_constant_value(c)
        assert e == 3, (c, d, e)

    @pytest.mark.parametrize("only_process_constants", (True, False))
    def test_None_and_NoneConst(self, only_process_constants):
        with pytest.raises(NotScalarConstantError):
            get_underlying_scalar_constant_value(
                None, only_process_constants=only_process_constants
            )
        assert (
            get_underlying_scalar_constant_value(
                NoneConst, only_process_constants=only_process_constants
            )
            is None
        )


@pytest.mark.parametrize(
    ["valid_inp", "invalid_inp"],
    ((np.array(4), np.zeros(5)), (ptb.constant(4), ptb.constant(3, ndim=1))),
)
def test_get_scalar_constant_value(valid_inp, invalid_inp):
    with pytest.raises(NotScalarConstantError):
        get_scalar_constant_value(invalid_inp)
    assert get_scalar_constant_value(valid_inp) == 4


def test_complex_mod_failure():
    # Make sure % fails on complex numbers.
    x = vector(dtype="complex64")
    with pytest.raises(ps.ComplexError):
        x % 5


class TestSize:
    # Ensure the `size` attribute of tensors behaves as in numpy.
    def test_matrix(self):
        x = matrix()
        y = np.zeros((5, 7), dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_vector(self):
        x = vector()
        y = np.zeros(7, dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_scalar(self):
        x = scalar()
        y = np.array(7, dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_shared(self):
        # NB: we also test higher order tensors at the same time.
        y = np.zeros((1, 2, 3, 4), dtype=config.floatX)
        x = pytensor.shared(y)
        assert y.size == function([], x.size)()


class TestDiag:
    """
    Test that linalg.diag has the same behavior as numpy.diag.
    numpy.diag has two behaviors:
    (1) when given a vector, it returns a matrix with that vector as the diagonal.
    (2) when given a matrix, returns a vector which is the diagonal of the matrix.

    (1) and (2) are further tested by TestAllocDiag and TestExtractDiag, respectively.
    """

    def setup_method(self):
        self.mode = None
        self.shared = shared
        self.floatX = config.floatX
        self.type = TensorType

    def test_diag(self):
        """Makes sure that diag instantiates the right op based on the dimension of the input."""
        rng = np.random.default_rng(utt.fetch_seed())

        # test vector input
        x = vector()
        g = diag(x)
        f = pytensor.function([x], g)
        for shp in [5, 0, 1]:
            m = rng.random(shp).astype(self.floatX)
            v = np.diag(m)
            r = f(m)
            # The right matrix is created
            assert (r == v).all()

        # Test matrix input
        xx = self.shared(rng.random((3, 5)))
        g = diag(xx)
        assert isinstance(g.owner.op, ExtractDiag)
        f = pytensor.function([], g)
        for shp in [(5, 3), (3, 5), (5, 1), (1, 5), (5, 0), (0, 5), (1, 0), (0, 1)]:
            m = rng.random(shp).astype(self.floatX)
            xx.set_value(m)
            v = np.diag(m)
            r = f()
            # The right matrix is created
            assert (r == v).all()

        # Test passing a list
        xx = [[1, 2], [3, 4]]
        g = diag(xx)
        f = function([], g)
        assert np.array_equal(f(), np.diag(xx))

    @pytest.mark.parametrize("inp", (scalar, tensor3))
    def test_diag_invalid_input_ndim(self, inp):
        x = inp()
        with pytest.raises(ValueError, match="Input must be 1- or 2-d."):
            diag(x)


class TestExtractDiag:
    @pytest.mark.parametrize("axis1, axis2", [(0, 1), (1, 0)])
    @pytest.mark.parametrize("offset", (-1, 0, 2))
    def test_infer_shape(self, offset, axis1, axis2):
        rng = np.random.default_rng(utt.fetch_seed())

        x = matrix("x")
        g = ExtractDiag(offset=offset, axis1=axis1, axis2=axis2)(x)
        f = pytensor.function([x], g.shape)
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert sum(isinstance(node.op, ExtractDiag) for node in topo) == 0
        for shp in [(5, 3), (3, 5), (5, 1), (1, 5), (5, 0), (0, 5), (1, 0), (0, 1)]:
            m = rng.random(shp).astype(config.floatX)
            assert (
                f(m) == np.diagonal(m, offset=offset, axis1=axis1, axis2=axis2).shape
            ).all()

    @pytest.mark.parametrize("axis1, axis2", [(0, 1), (1, 0)])
    @pytest.mark.parametrize("offset", (0, 1, -1))
    def test_grad_2d(self, offset, axis1, axis2):
        diag_fn = ExtractDiag(offset=offset, axis1=axis1, axis2=axis2)
        rng = np.random.default_rng(utt.fetch_seed())
        x = rng.random((5, 3))
        utt.verify_grad(diag_fn, [x], rng=rng)

    @pytest.mark.parametrize(
        "axis1, axis2",
        [
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (0, 2),
            (2, 0),
        ],
    )
    @pytest.mark.parametrize("offset", (0, 1, -1))
    def test_grad_3d(self, offset, axis1, axis2):
        diag_fn = ExtractDiag(offset=offset, axis1=axis1, axis2=axis2)
        rng = np.random.default_rng(utt.fetch_seed())
        x = rng.random((5, 4, 3))
        utt.verify_grad(diag_fn, [x], rng=rng)


class TestAllocDiag:
    # TODO: Separate perform, grad and infer_shape tests

    def _generator(self):
        dims = 4
        shape = (5,) * dims
        xv = np.random.standard_normal(shape).astype(config.floatX)
        for d in range(1, dims + 1):
            # Create a TensorType of the same dimensions as
            # as the data we want to test.
            x = TensorType(dtype=config.floatX, shape=(None,) * d)("x")

            # Make a slice of the test data that has the
            # dimensions we need by doing xv[0,...,0]
            # For example, for an array of shape (5,), we
            # need to do xv[0, 0, 0, 0].
            test_val = xv[((0,) * (dims - d))]
            yield x, test_val

    def test_alloc_diag_values(self):
        for x, test_val in self._generator():
            for offset, axis1, axis2 in [
                (0, 0, 1),
                (0, 1, 2),
                (1, 0, 1),
                (0, 1, 3),
                (0, 2, 3),
                (1, 2, 3),
                (-1, 0, 1),
                (-2, 0, 1),
                (-1, 1, 2),
            ]:
                # Test perform
                if np.maximum(axis1, axis2) > len(test_val.shape):
                    continue
                diag_x = ptb.alloc_diag(x, offset=offset, axis1=axis1, axis2=axis2)
                f = pytensor.function([x], diag_x)
                # alloc_diag and extract the diagonal again to check for correctness
                diag_arr = f(test_val)
                rediag = np.diagonal(diag_arr, offset=offset, axis1=axis1, axis2=axis2)
                assert np.all(rediag == test_val)

                # Test infer_shape
                f_shape = pytensor.function([x], diag_x.shape, mode="FAST_RUN")

                output_shape = f_shape(test_val)
                rediag_shape = np.diagonal(
                    np.ones(output_shape), offset=offset, axis1=axis1, axis2=axis2
                ).shape
                assert np.all(rediag_shape == test_val.shape)

                # Test grad
                sum_diag_x = pt_sum(diag_x)
                grad_x = pytensor.grad(sum_diag_x, x)
                grad_diag_x = pytensor.grad(sum_diag_x, diag_x)
                f_grad_x = pytensor.function([x], grad_x)
                f_grad_diag_x = pytensor.function([x], grad_diag_x)
                grad_input = f_grad_x(test_val)
                grad_diag_input = f_grad_diag_x(test_val)
                true_grad_input = np.diagonal(
                    grad_diag_input, offset=offset, axis1=axis1, axis2=axis2
                )
                assert np.all(true_grad_input == grad_input)

    def test_multiple_ops_same_graph(self):
        """Regression test when AllocDiag OFG was given insufficient props, causing incompatible Ops to be merged."""
        v1 = vector("v1", shape=(2,), dtype="float64")
        v2 = vector("v2", shape=(3,), dtype="float64")
        a1 = alloc_diag(v1)
        a2 = alloc_diag(v2)

        fn = function([v1, v2], [a1, a2])
        res1, res2 = fn(v1=[np.e, np.e], v2=[np.pi, np.pi, np.pi])
        np.testing.assert_allclose(res1, np.eye(2) * np.e)
        np.testing.assert_allclose(res2, np.eye(3) * np.pi)


def test_diagonal_negative_axis():
    x = np.arange(2 * 3 * 3).reshape((2, 3, 3))
    np.testing.assert_allclose(
        ptb.diagonal(x, axis1=-1, axis2=-2).eval(),
        np.diagonal(x, axis1=-1, axis2=-2),
    )


def test_transpose():
    x1 = dvector("x1")
    x2 = dmatrix("x2")
    x3 = dtensor3("x3")

    x1v = np.arange(24)
    x2v = np.arange(24).reshape(2, 12)
    x3v = np.arange(24).reshape(2, 3, 4)

    f = pytensor.function(
        [x1, x2, x3],
        [
            ptb.transpose(x1),
            ptb.transpose(x2),
            ptb.transpose(x3),
            x1.transpose(),
            x2.transpose(),
            x3.transpose(),
            x2.transpose(0, 1),
            x3.transpose((0, 2, 1)),
            ptb.transpose(x2, [0, 1]),
            ptb.transpose(x3, [0, 2, 1]),
        ],
    )

    t1, t2, t3, t1b, t2b, t3b, t2c, t3c, t2d, t3d = f(x1v, x2v, x3v)

    assert t1.shape == np.transpose(x1v).shape
    assert t2.shape == np.transpose(x2v).shape
    assert t3.shape == np.transpose(x3v).shape
    assert np.all(t1 == np.transpose(x1v))
    assert np.all(t2 == np.transpose(x2v))
    assert np.all(t3 == np.transpose(x3v))
    assert np.all(t1b == x1v.transpose())
    assert np.all(t2b == x2v.transpose())
    assert np.all(t3b == x3v.transpose())
    assert t2c.shape == (2, 12)
    assert t3c.shape == (2, 4, 3)
    assert np.all(t2c == x2v.transpose([0, 1]))
    assert np.all(t3c == x3v.transpose([0, 2, 1]))
    assert t2d.shape == (2, 12)
    assert t3d.shape == (2, 4, 3)
    assert np.all(t2d == np.transpose(x2v, [0, 1]))
    assert np.all(t3d == np.transpose(x3v, [0, 2, 1]))

    # Check we don't introduce useless transpose
    assert ptb.transpose(x1) is x1

    # Check that we create a name.
    assert ptb.transpose(x2).name == "x2.T"
    assert ptb.transpose(x3).name == "x3.T"
    assert ptb.transpose(dmatrix()).name is None


def test_matrix_transpose():
    with pytest.raises(ValueError, match="Input array must be at least 2-dimensional"):
        ptb.matrix_transpose(dvector("x1"))

    x2 = dmatrix("x2")
    x3 = dtensor3("x3")

    var1 = ptb.matrix_transpose(x2)
    expected_var1 = swapaxes(x2, -1, -2)

    var2 = x3.mT
    expected_var2 = swapaxes(x3, -1, -2)

    assert equal_computations([var1], [expected_var1])
    assert equal_computations([var2], [expected_var2])


def test_stacklists():
    a, b, c, d = map(scalar, "abcd")
    X = stacklists([[a, b], [c, d]])
    f = function([a, b, c, d], X)
    result = f(1, 2, 3, 4)
    assert result.shape == (2, 2)
    assert np.allclose(f(1, 2, 3, 4), np.asarray([[1, 2], [3, 4]]))

    X = stacklists([a, b, c, d])
    f = function([a, b, c, d], X)
    result = f(1, 2, 3, 4)
    assert result.shape == (4,)
    assert np.allclose(f(1, 2, 3, 4), np.asarray([[1, 2, 3, 4]]))

    X = stacklists([[[a], [b]], [[c], [d]]])
    f = function([a, b, c, d], X)
    result = f(1, 2, 3, 4)
    assert result.shape == (2, 2, 1)

    a, b, c, d = (matrix(x) for x in "abcd")
    X = stacklists([[a, b], [c, d]])
    f = function([a, b, c, d], X)
    x = np.ones((4, 4), "float32")
    assert f(x, x, x, x).shape == (2, 2, 4, 4)


class TestInferShape(utt.InferShapeTester):
    def test_Flatten(self):
        atens3 = tensor3()
        atens3_val = random(4, 5, 3)
        for ndim in (2, 1):
            self._compile_and_check(
                [atens3],
                [flatten(atens3, ndim)],
                [atens3_val],
                Reshape,
            )

        amat = matrix()
        amat_val = random(4, 5)
        ndim = 1
        self._compile_and_check(
            [amat],
            [flatten(amat, ndim)],
            [amat_val],
            Reshape,
        )

    def test_Eye(self):
        aiscal = iscalar()
        biscal = iscalar()
        ciscal = iscalar()
        self._compile_and_check(
            [aiscal, biscal, ciscal], [Eye()(aiscal, biscal, ciscal)], [4, 4, 0], Eye
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Eye()(aiscal, biscal, ciscal)], [4, 5, 0], Eye
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Eye()(aiscal, biscal, ciscal)], [3, 5, 0], Eye
        )

    def test_Tri(self):
        aiscal = iscalar()
        biscal = iscalar()
        ciscal = iscalar()
        self._compile_and_check(
            [aiscal, biscal, ciscal], [Tri()(aiscal, biscal, ciscal)], [4, 4, 0], Tri
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Tri()(aiscal, biscal, ciscal)], [4, 5, 0], Tri
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal], [Tri()(aiscal, biscal, ciscal)], [3, 5, 0], Tri
        )

    def test_ExtractDiag(self):
        atens3 = tensor3()
        atens3_val = random(4, 5, 3)
        atens3_diag = ExtractDiag()(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(-1)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1, 0, 2)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1, 1, 2)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)
        atens3_diag = ExtractDiag(1, 2, 0)(atens3)
        self._compile_and_check([atens3], [atens3_diag], [atens3_val], ExtractDiag)

    def test_Split(self):
        aiscal = iscalar()
        aivec = ivector()
        adtens = tensor3()
        adtens_val = random(4, 10, 3)
        aivec_val = [2, 5, 3]
        for aiscal_val in [1, -2]:
            self._compile_and_check(
                [adtens, aiscal, aivec],
                [Split(3)(adtens, aiscal, aivec)[0]],
                [adtens_val, aiscal_val, aivec_val],
                (Split),
            )

    def test_Join(self):
        aiscal = iscalar()
        cdmat = dmatrix()
        admat_val = random(1, 3)
        bdmat_val = random(2, 3)
        cdmat_val = random(4, 3)
        admat = dmatrix()
        bdmat = dmatrix()
        for aiscal_val in [0, -2]:
            self._compile_and_check(
                [aiscal, admat, bdmat, cdmat],
                [Join()(aiscal, admat, bdmat, cdmat)],
                [aiscal_val, admat_val, bdmat_val, cdmat_val],
                Join,
            )

        admat_val = random(4, 1)
        bdmat_val = random(4, 3)
        cdmat_val = random(4, 2)
        for aiscal_val in [-1, 1]:
            self._compile_and_check(
                [aiscal, admat, bdmat, cdmat],
                [Join()(aiscal, admat, bdmat, cdmat)],
                [aiscal_val, admat_val, bdmat_val, cdmat_val],
                Join,
            )

    def test_PermuteRowElements(self):
        admat = dmatrix()
        advec = dvector()
        aivec = ivector()

        rng = np.random.default_rng(utt.fetch_seed())
        advec_val = random(5)
        aivec_val = rng.permutation(5).astype("int32")
        self._compile_and_check(
            [advec, aivec],
            [PermuteRowElements(inverse=True)(advec, aivec)],
            [advec_val, aivec_val],
            PermuteRowElements,
        )

        admat_val = random(3, 5)
        self._compile_and_check(
            [admat, aivec],
            [PermuteRowElements(inverse=False)(admat, aivec)],
            [admat_val, aivec_val],
            PermuteRowElements,
        )

        adtens3 = dtensor3()
        adtens3_val = random(3, 2, 5)
        self._compile_and_check(
            [adtens3, aivec],
            [PermuteRowElements(inverse=True)(adtens3, aivec)],
            [adtens3_val, aivec_val],
            PermuteRowElements,
        )

        aimat = imatrix()
        perma = rng.permutation(5).astype("int32")
        permb = rng.permutation(5).astype("int32")
        permc = rng.permutation(5).astype("int32")
        aimat_val = np.vstack((perma, permb, permc))
        admat_val = random(3, 5)
        self._compile_and_check(
            [admat, aimat],
            [PermuteRowElements(inverse=False)(admat, aimat)],
            [admat_val, aimat_val],
            PermuteRowElements,
        )

        aitens3 = itensor3()
        perma = rng.permutation(5).astype("int32")
        permb = rng.permutation(5).astype("int32")
        permc = rng.permutation(5).astype("int32")
        bimat_val = np.vstack((perma, permb, permc))
        aitens3_val = np.empty((2, 3, 5), "int32")
        aitens3_val[0, ::, ::] = aimat_val
        aitens3_val[1, ::, ::] = bimat_val
        self._compile_and_check(
            [admat, aitens3],
            [PermuteRowElements(inverse=True)(admat, aitens3)],
            [admat_val, aitens3_val],
            PermuteRowElements,
        )

    def test_ScalarFromTensor(self):
        aiscal = iscalar()
        self._compile_and_check(
            [aiscal],
            [TensorFromScalar()(ScalarFromTensor()(aiscal))],
            [45],
            ScalarFromTensor,
            excluding=["local_tensor_scalar_tensor"],
        )

    def test_TensorFromScalar(self):
        aiscal = ps.float64()

        self._compile_and_check(
            [aiscal], [TensorFromScalar()(aiscal)], [4.0], TensorFromScalar
        )

    def test_Alloc(self):
        integers = np.random.default_rng(utt.fetch_seed()).integers
        adscal = dscalar()
        aiscal = lscalar()
        biscal = lscalar()
        ciscal = lscalar()
        discal = lscalar()
        adscal_val = random()
        aiscal_val = integers(3, 6, size=())
        biscal_val = integers(3, 6, size=())
        ciscal_val = integers(3, 6, size=())
        discal_val = integers(3, 6, size=())
        self._compile_and_check(
            [adscal, aiscal, biscal, ciscal, discal],
            [Alloc()(adscal, aiscal, biscal, ciscal, discal)],
            [adscal_val, aiscal_val, biscal_val, ciscal_val, discal_val],
            Alloc,
        )

    def test_ARange(self):
        aiscal = lscalar()
        biscal = lscalar()
        ciscal = lscalar()

        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [0, 5, 1],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [2, 11, 4],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [-5, 1, 1],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [10, 2, -2],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [10, 2, 2],
            ARange,
        )
        self._compile_and_check(
            [aiscal, biscal, ciscal],
            [ARange("int64")(aiscal, biscal, ciscal)],
            [0, 0, 1],
            ARange,
        )


class TestSwapaxes:
    def test_no_dimensional_input(self):
        with pytest.raises(IndexError):
            swapaxes(2, 0, 1)

    def test_unidimensional_input(self):
        with pytest.raises(IndexError):
            swapaxes([2, 1], 0, 1)

    def test_not_enough_dimension(self):
        with pytest.raises(IndexError):
            swapaxes([[2, 1], [3, 4]], 3, 4)

    def test_doubleswap(self):
        y = matrix()
        n = swapaxes(y, 0, 1)
        f = function([y], n)
        testMatrix = [[2, 1], [3, 4]]
        assert np.array_equal(testMatrix, f(f(testMatrix)))

    def test_interface(self):
        x = matrix()
        x.swapaxes(0, 1)

    def test_numpy_compare(self):
        rng = np.random.default_rng(utt.fetch_seed())
        A = matrix("A", dtype=config.floatX)
        Q = swapaxes(A, 0, 1)
        fn = function([A], [Q])
        a = rng.random((4, 4)).astype(config.floatX)

        n_s = np.swapaxes(a, 0, 1)
        t_s = fn(a)
        assert np.allclose(n_s, t_s)


def test_moveaxis():
    x = ptb.zeros((3, 4, 5))
    tuple(moveaxis(x, 0, -1).shape.eval()) == (4, 5, 3)
    tuple(moveaxis(x, -1, 0).shape.eval()) == (5, 3, 4)
    tuple(moveaxis(x, [0, 1], [-1, -2]).shape.eval()) == (5, 4, 3)
    tuple(moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape.eval()) == (5, 4, 3)


def test_moveaxis_error():
    x = ptb.zeros((3, 4, 5))
    with pytest.raises(
        ValueError,
        match="`source` and `destination` arguments must have the same number of elements",
    ):
        moveaxis(x, [0, 1], 0)


class TestChoose(utt.InferShapeTester):
    op = staticmethod(choose)
    op_class = Choose
    modes = ["raise", "wrap", "clip"]
    rng = np.random.default_rng(utt.fetch_seed())

    def test_numpy_compare(self):
        a = vector(dtype="int32")
        b = matrix(dtype="float32")

        A = self.rng.integers(0, 4, 4).astype("int32")
        B = np.asarray(np.random.random((4, 4)), dtype="float32")

        for m in self.modes:
            f = function([a, b], choose(a, b, mode=m))
            t_c = f(A, B)
            n_c = np.choose(A, B, mode=m)
            assert np.allclose(t_c, n_c)

    def test_method(self):
        a = vector(dtype="int32")
        b = matrix(dtype="float32")

        A = self.rng.integers(0, 4, 4).astype("int32")
        B = np.asarray(np.random.random((4, 4)), dtype="float32")

        for m in self.modes:
            f = function([a, b], a.choose(b, mode=m))
            t_c = f(A, B)
            n_c = A.choose(B, mode=m)
            assert np.allclose(t_c, n_c)

    def test_broadcasted(self):
        a = scalar(dtype="int32")
        b = matrix(dtype="float32")

        # Test when a is broadcastable
        A = 3
        B = np.asarray(np.random.random((4, 4)), dtype="float32")

        for m in self.modes:
            f = function([a, b], choose(a, b, mode=m))
            t_c = f(A, B)
            n_c = np.choose(A, B, mode=m)
            assert np.allclose(t_c, n_c)

        # Test when the result should be broadcastable
        b = col(dtype="float32")
        B = np.asarray(np.random.random((4, 1)), dtype="float32")
        for m in self.modes:
            f = function([a, b], choose(a, b, mode=m))
            assert choose(a, b, mode=m).type.shape[0] == 1
            t_c = f(A, B)
            n_c = np.choose(A, B, mode=m)
            assert np.allclose(t_c, n_c)

    def test_dtype_error(self):
        a = scalar(dtype="float32")
        b = matrix(dtype="float32")

        with pytest.raises(TypeError):
            choose(a, b)

    @pytest.mark.parametrize(
        "test_input",
        [
            (
                tensor3(dtype="int32"),
                tensor3(dtype="float32"),
                tensor3(dtype="float32"),
                rng.integers(0, 2, (2, 1, 1)).astype("int32"),
                np.asarray(np.random.random((1, 6, 1)), dtype="float32"),
                np.asarray(np.random.random((1, 1, 5)), dtype="float32"),
            ),
            (
                vector(dtype="int32"),
                scalar(),
                scalar(),
                [0, 1, 1, 0],
                0.1,
                0.2,
            ),
        ],
    )
    def test_numpy_compare_tuple(self, test_input):
        """Test with list and tuples of scalars and 3d tensors."""
        a, b, c, A, B, C = test_input
        for m in self.modes:
            for ls in [list, tuple]:
                f = function([a, b, c], choose(a, ls([b, c]), mode=m))
                t_c = f(A, B, C)
                n_c = np.choose(A, ls([B, C]), mode=m)
                assert np.allclose(t_c, n_c)

    def test_infer_shape(self):
        for shp1, shp2 in [
            ((5, 4), (7, 4)),
            ((1, 4), (7, 4)),
            ((5, 1), (7, 4)),
            ((5, 4), (1, 4)),
            ((5, 4), (7, 1)),
            ((5, 4), (4,)),
            ((1, 4), (4,)),
            ((5, 1), (4,)),
            ((5, 4), (1,)),
            ((4,), (5, 4)),
            ((1,), (5, 4)),
            ((4,), (1, 4)),
            ((4,), (3, 1)),
            ((4,), (4,)),
            ((1,), (4,)),
            ((4,), (1,)),
            ((1,), (1,)),
        ]:
            a = tensor(dtype="int32", shape=tuple(1 if s == 1 else None for s in shp1))
            c = tensor(
                dtype="float32", shape=tuple(1 if s == 1 else None for s in shp2)
            )
            A = np.asarray(np.random.random(shp1) * shp2[0], dtype="int32")
            C = np.asarray(np.random.random(shp2) * shp2[0], dtype="float32")
            self._compile_and_check(
                [a, c],  # pytensor.function inputs
                [self.op(a, c)],  # pytensor.function outputs
                # Always use not square matrix!
                # inputs data
                [A, C],
                # Op that should be removed from the graph.
                self.op_class,
            )

    @pytest.mark.skip(reason="Not implemented")
    def test_infer_shape_tuple(self):
        a = tensor3(dtype="int32")
        b = tensor3(dtype="int32")
        c = tensor3(dtype="int32")

        A = np.asarray([1, 0], dtype="int32").reshape((2, 1, 1))
        B = np.asarray(np.random.random((1, 4, 1)), dtype="int32")
        C = np.asarray(np.random.random((1, 1, 7)), dtype="int32")

        f = function([a, b, c], choose(a, (b, c)))
        shape = (2, 4, 7)
        assert np.allclose(f(A, B, C).shape, shape)

        self._compile_and_check(
            [a, b, c],  # pytensor.function inputs
            [self.op(a, (b, c))],  # pytensor.function outputs
            # Always use not square matrix!
            # inputs data
            [A, B, C],
            # Op that should be removed from the graph.
            self.op_class,
        )


def test_empty():
    # Test that we allocated correctly
    f = pytensor.function([], AllocEmpty("float32")(2, 3))
    assert len(f.maker.fgraph.apply_nodes) == 1
    out = f()

    assert out.shape == (2, 3)
    assert out.dtype == "float32"

    empty_pt = ptb.empty(3)
    res = pytensor.function([], empty_pt)()
    assert res.shape == (3,)

    empty_pt = ptb.empty((2, 3), dtype=None)
    res = pytensor.function([], empty_pt)()
    assert res.shape == (2, 3)

    empty_pt = ptb.empty((2, 3), dtype="int64")
    res = pytensor.function([], empty_pt)()
    assert res.shape == (2, 3)
    assert res.dtype == "int64"

    empty_pt = ptb.empty_like(empty_pt)
    res = pytensor.function([], empty_pt)()
    assert res.shape == (2, 3)
    assert res.dtype == "int64"


def test_identity_like_dtype():
    # Test that we allocate eye correctly via identity_like
    m = matrix(dtype="int64")
    m_out = identity_like(m)
    assert m_out.dtype == m.dtype
    m_out_float = identity_like(m, dtype=np.float64)
    assert m_out_float.dtype == "float64"

    # Test passing list
    m = [[0, 1], [1, 3]]
    out = ptb.identity_like(m)
    f = pytensor.function([], out)
    assert np.array_equal(f(), np.eye(2))


def test_atleast_Nd():
    ary1 = dscalar()
    res_ary1 = atleast_Nd(ary1, n=1)
    assert res_ary1.ndim == 1

    for n in range(1, 3):
        ary1, ary2 = dscalar(), dvector()
        res_ary1 = atleast_Nd(ary1, n=n)
        res_ary2 = atleast_Nd(ary2, n=n)

        assert res_ary1.ndim == n
        if n == ary2.ndim:
            assert ary2 is res_ary2
        else:
            assert res_ary2.ndim == n

        ary1_val = np.array(1.0, dtype=np.float64)
        ary2_val = np.array([1.0, 2.0], dtype=np.float64)
        res_ary1_val, res_ary2_val = pytensor.function(
            [ary1, ary2], [res_ary1, res_ary2]
        )(ary1_val, ary2_val)

        np_fn = np.atleast_1d if n == 1 else np.atleast_2d
        assert np.array_equal(res_ary1_val, np_fn(ary1_val))
        assert np.array_equal(res_ary2_val, np_fn(ary2_val))


def test_expand_dims():
    x_pt = dscalar()
    res_pt = expand_dims(x_pt, 0)
    x_val = np.array(1.0, dtype=np.float64)
    exp_res = np.expand_dims(x_val, 0)
    res_val = pytensor.function([x_pt], res_pt)(x_val)
    assert np.array_equal(exp_res, res_val)

    x_pt = dscalar()
    res_pt = expand_dims(x_pt, (0, 1))
    x_val = np.array(1.0, dtype=np.float64)
    exp_res = np.expand_dims(x_val, (0, 1))
    res_val = pytensor.function([x_pt], res_pt)(x_val)
    assert np.array_equal(exp_res, res_val)

    x_pt = dmatrix()
    res_pt = expand_dims(x_pt, (2, 1))
    x_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    exp_res = np.expand_dims(x_val, (2, 1))
    res_val = pytensor.function([x_pt], res_pt)(x_val)
    assert np.array_equal(exp_res, res_val)


class TestTakeAlongAxis:
    @pytest.mark.parametrize(
        ["shape", "axis", "samples"],
        (
            ((1,), None, 1),
            ((1,), -1, 10),
            ((3, 2, 1), -1, 1),
            ((3, 2, 1), 0, 10),
            ((3, 2, 1), -1, 10),
        ),
        ids=str,
    )
    def test_take_along_axis(self, shape, axis, samples):
        rng = np.random.default_rng()
        arr = rng.normal(size=shape).astype(config.floatX)
        indices_size = list(shape)
        indices_size[axis or 0] = samples
        indices = rng.integers(low=0, high=shape[axis or 0], size=indices_size)

        arr_in = ptb.tensor(
            dtype=config.floatX, shape=tuple(1 if s == 1 else None for s in arr.shape)
        )
        indices_in = ptb.tensor(
            dtype=np.int64, shape=tuple(1 if s == 1 else None for s in indices.shape)
        )

        out = ptb.take_along_axis(arr_in, indices_in, axis)

        func = pytensor.function([arr_in, indices_in], out)

        assert np.allclose(
            np.take_along_axis(arr, indices, axis=axis), func(arr, indices)
        )

    def test_ndim_dtype_failures(self):
        arr = ptb.tensor(dtype=config.floatX, shape=(None,) * 2)
        indices = ptb.tensor(dtype=np.int64, shape=(None,) * 3)
        with pytest.raises(ValueError):
            ptb.take_along_axis(arr, indices)

        indices = ptb.tensor(dtype=np.float64, shape=(None,) * 2)
        with pytest.raises(IndexError):
            ptb.take_along_axis(arr, indices)


@pytest.mark.parametrize(
    "inp, shape",
    [(scalar, ()), (vector, 3), (matrix, (3, 4))],
)
def test_full_like(inp, shape):
    fill_value = 5
    dtype = config.floatX

    x = inp("x")
    y = full_like(x, fill_value, dtype=dtype)

    np.testing.assert_array_equal(
        y.eval({x: np.zeros(shape, dtype=dtype)}),
        np.full(shape, fill_value, dtype=dtype),
    )


@pytest.mark.parametrize("func", [horizontal_stack, vertical_stack])
def test_oriented_stack_functions(func):
    with pytest.raises(ValueError):
        func()

    a = ptb.tensor(dtype=np.float64, shape=(None, None, None))

    with pytest.raises(ValueError):
        func(a, a)


def test_trace():
    x_val = np.ones((5, 4, 2))
    x = ptb.as_tensor(x_val)

    np.testing.assert_allclose(
        trace(x).eval(),
        np.trace(x_val),
    )

    np.testing.assert_allclose(
        trace(x, offset=1, axis1=1, axis2=2).eval(),
        np.trace(x_val, offset=1, axis1=1, axis2=2),
    )

    np.testing.assert_allclose(
        trace(x, offset=-1, axis1=0, axis2=-1).eval(),
        np.trace(x_val, offset=-1, axis1=0, axis2=-1),
    )


def test_vectorize_extract_diag():
    signature = "(a1,b,a2)->(b,a)"

    def core_pt(x):
        return ptb.diagonal(x, offset=1, axis1=0, axis2=2)

    def core_np(x):
        return np.diagonal(x, offset=1, axis1=0, axis2=2)

    x = tensor(shape=(5, 5, 5, 5))
    vectorize_pt = function([x], vectorize(core_pt, signature=signature)(x))
    assert not any(
        isinstance(node.op, Blockwise) for node in vectorize_pt.maker.fgraph.apply_nodes
    )

    x_test = np.random.normal(size=x.type.shape).astype(x.type.dtype)
    vectorize_np = np.vectorize(core_np, signature=signature)
    np.testing.assert_allclose(
        vectorize_pt(x_test),
        vectorize_np(x_test),
    )


@pytest.mark.parametrize(
    "batch_shapes",
    [
        ((3,),),  # edge case of make_vector with a single input
        ((), (), ()),  # Useless
        ((3,), (3,), (3,)),  # No broadcasting needed
        ((3,), (5, 3), ()),  # Broadcasting needed
    ],
)
def test_vectorize_make_vector(batch_shapes):
    n_inputs = len(batch_shapes)
    input_sig = ",".join(["()"] * n_inputs)
    signature = f"{input_sig}->({n_inputs})"  # Something like "(),(),()->(3)"

    def core_pt(*scalars):
        out = stack(scalars)
        out.dprint()
        return out

    def core_np(*scalars):
        return np.stack(scalars)

    tensors = [tensor(shape=shape) for shape in batch_shapes]

    vectorize_pt = function(tensors, vectorize(core_pt, signature=signature)(*tensors))
    assert not any(
        isinstance(node.op, Blockwise) for node in vectorize_pt.maker.fgraph.apply_nodes
    )

    test_values = [
        np.random.normal(size=tensor.type.shape).astype(tensor.type.dtype)
        for tensor in tensors
    ]

    np.testing.assert_allclose(
        vectorize_pt(*test_values),
        np.vectorize(core_np, signature=signature)(*test_values),
    )


@pytest.mark.parametrize("axis", [constant(1), constant(-2), shared(1)])
@pytest.mark.parametrize("broadcasting_y", ["none", "implicit", "explicit"])
@config.change_flags(cxx="")  # C code not needed
def test_vectorize_join(axis, broadcasting_y):
    # Signature for join along intermediate axis
    signature = "(a,b1,c),(a,b2,c)->(a,b,c)"

    def core_pt(x, y):
        return join(axis, x, y)

    def core_np(x, y):
        return np.concatenate([x, y], axis=axis.eval())

    x = tensor(shape=(4, 2, 3, 5))
    y_shape = {"none": (4, 2, 3, 5), "implicit": (2, 3, 5), "explicit": (1, 2, 3, 5)}
    y = tensor(shape=y_shape[broadcasting_y])

    vectorize_pt = function([x, y], vectorize(core_pt, signature=signature)(x, y))

    blockwise_needed = isinstance(axis, SharedVariable) or broadcasting_y != "none"
    has_blockwise = any(
        isinstance(node.op, Blockwise) for node in vectorize_pt.maker.fgraph.apply_nodes
    )
    assert has_blockwise == blockwise_needed

    x_test = np.random.normal(size=x.type.shape).astype(x.type.dtype)
    y_test = np.random.normal(size=y.type.shape).astype(y.type.dtype)
    vectorize_np = np.vectorize(core_np, signature=signature)
    np.testing.assert_allclose(
        vectorize_pt(x_test, y_test),
        vectorize_np(x_test, y_test),
    )


def test_where():
    a = np.arange(10)
    cond = a < 5
    ift = np.pi
    iff = np.e
    # Test for all 3 inputs
    np.testing.assert_allclose(np.where(cond, ift, iff), where(cond, ift, iff).eval())

    # Test for only condition input
    for np_output, pt_output in zip(np.where(cond), where(cond), strict=True):
        np.testing.assert_allclose(np_output, pt_output.eval())

    # Test for error
    with pytest.raises(ValueError, match="either both"):
        where(cond, ift)
