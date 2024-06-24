import contextlib
import inspect
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import pytest

from tests.tensor.test_math_scipy import scipy


numba = pytest.importorskip("numba")

import pytensor.scalar as ps
import pytensor.scalar.math as psm
import pytensor.tensor as pt
import pytensor.tensor.math as ptm
from pytensor import config, shared
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.compile.ops import ViewOp
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Apply, Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op, get_test_value
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.type import Type
from pytensor.ifelse import ifelse
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.linker import NumbaLinker
from pytensor.raise_op import assert_op
from pytensor.scalar.basic import ScalarOp, as_scalar
from pytensor.tensor import blas
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape


if TYPE_CHECKING:
    from pytensor.graph.basic import Variable
    from pytensor.tensor import TensorLike


class MyType(Type):
    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)

    def __hash__(self):
        return hash(MyType)


class MyOp(Op):
    def perform(self, *args):
        pass


class MySingleOut(Op):
    def make_node(self, a, b):
        return Apply(self, [a, b], [a.type()])

    def perform(self, node, inputs, outputs):
        res = (inputs[0] + inputs[1]).astype(inputs[0][0].dtype)
        outputs[0][0] = res


class ScalarMyMultiOut(ScalarOp):
    nin = 2
    nout = 2

    @staticmethod
    def impl(a, b):
        res1 = 2 * a
        res2 = 2 * b
        return [res1, res2]

    def make_node(self, a, b):
        a = as_scalar(a)
        b = as_scalar(b)
        return Apply(self, [a, b], [a.type(), b.type()])

    def perform(self, node, inputs, outputs):
        res1, res2 = self.impl(inputs[0], inputs[1])
        outputs[0][0] = res1
        outputs[1][0] = res2


scalar_my_multi_out = Elemwise(ScalarMyMultiOut())
scalar_my_multi_out.ufunc = ScalarMyMultiOut.impl
scalar_my_multi_out.ufunc.nin = 2
scalar_my_multi_out.ufunc.nout = 2


class MyMultiOut(Op):
    nin = 2
    nout = 2

    @staticmethod
    def impl(a, b):
        res1 = 2 * a
        res2 = 2 * b
        return [res1, res2]

    def make_node(self, a, b):
        return Apply(self, [a, b], [a.type(), b.type()])

    def perform(self, node, inputs, outputs):
        res1, res2 = self.impl(inputs[0], inputs[1])
        outputs[0][0] = res1
        outputs[1][0] = res2


my_multi_out = Elemwise(MyMultiOut())
my_multi_out.ufunc = MyMultiOut.impl
my_multi_out.ufunc.nin = 2
my_multi_out.ufunc.nout = 2
opts = RewriteDatabaseQuery(
    include=[None], exclude=["cxx_only", "BlasOpt", "local_careduce_fusion"]
)
numba_mode = Mode(
    NumbaLinker(), opts.including("numba", "local_useless_unbatched_blockwise")
)
py_mode = Mode("py", opts)

rng = np.random.default_rng(42849)


def set_test_value(x, v):
    x.tag.test_value = v
    return x


def compare_shape_dtype(x, y):
    return x.shape == y.shape and x.dtype == y.dtype


def eval_python_only(fn_inputs, fn_outputs, inputs, mode=numba_mode):
    """Evaluate the Numba implementation in pure Python for coverage purposes."""

    def py_tuple_setitem(t, i, v):
        ll = list(t)
        ll[i] = v
        return tuple(ll)

    def py_to_scalar(x):
        if isinstance(x, np.ndarray):
            return x.item()
        else:
            return x

    def njit_noop(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        else:
            return lambda x: x

    def vectorize_noop(*args, **kwargs):
        def wrap(fn):
            # `numba.vectorize` allows an `out` positional argument.  We need
            # to account for that
            sig = inspect.signature(fn)
            nparams = len(sig.parameters)

            def inner_vec(*args):
                if len(args) > nparams:
                    # An `out` argument has been specified for an in-place
                    # operation
                    out = args[-1]
                    out[...] = np.vectorize(fn)(*args[:nparams])
                    return out
                else:
                    return np.vectorize(fn)(*args)

            return inner_vec

        if len(args) == 1 and callable(args[0]):
            return wrap(args[0], **kwargs)
        else:
            return wrap

    def py_global_numba_func(func):
        if hasattr(func, "py_func"):
            return func.py_func
        return func

    mocks = [
        mock.patch("numba.njit", njit_noop),
        mock.patch("numba.vectorize", vectorize_noop),
        mock.patch(
            "pytensor.link.numba.dispatch.basic.global_numba_func",
            py_global_numba_func,
        ),
        mock.patch(
            "pytensor.link.numba.dispatch.basic.tuple_setitem", py_tuple_setitem
        ),
        mock.patch("pytensor.link.numba.dispatch.basic.numba_njit", njit_noop),
        mock.patch(
            "pytensor.link.numba.dispatch.basic.numba_vectorize", vectorize_noop
        ),
        mock.patch(
            "pytensor.link.numba.dispatch.basic.direct_cast", lambda x, dtype: x
        ),
        mock.patch("pytensor.link.numba.dispatch.basic.to_scalar", py_to_scalar),
        mock.patch(
            "pytensor.link.numba.dispatch.basic.numba.np.numpy_support.from_dtype",
            lambda dtype: dtype,
        ),
        mock.patch("numba.np.unsafe.ndarray.to_fixed_tuple", lambda x, n: tuple(x)),
    ]

    with contextlib.ExitStack() as stack:
        for ctx in mocks:
            stack.enter_context(ctx)

        pytensor_numba_fn = function(
            fn_inputs,
            fn_outputs,
            mode=mode,
            accept_inplace=True,
        )
        _ = pytensor_numba_fn(*inputs)


def compare_numba_and_py(
    fgraph: FunctionGraph | tuple[Sequence["Variable"], Sequence["Variable"]],
    inputs: Sequence["TensorLike"],
    assert_fn: Callable | None = None,
    numba_mode=numba_mode,
    py_mode=py_mode,
    updates=None,
    eval_obj_mode: bool = True,
) -> tuple[Callable, Any]:
    """Function to compare python graph output and Numba compiled output for testing equality

    In the tests below computational graphs are defined in PyTensor. These graphs are then passed to
    this function which then compiles the graphs in both Numba and python, runs the calculation
    in both and checks if the results are the same

    Parameters
    ----------
    fgraph
        `FunctionGraph` or inputs to compare.
    inputs
        Numeric inputs to be passed to the compiled graphs.
    assert_fn
        Assert function used to check for equality between python and Numba. If not
        provided uses `np.testing.assert_allclose`.
    updates
        Updates to be passed to `pytensor.function`.
    eval_obj_mode : bool, default True
        Whether to do an isolated call in object mode. Used for test coverage

    Returns
    -------
    The compiled PyTensor function and its last computed result.

    """
    if assert_fn is None:

        def assert_fn(x, y):
            return np.testing.assert_allclose(x, y, rtol=1e-4) and compare_shape_dtype(
                x, y
            )

    if isinstance(fgraph, tuple):
        fn_inputs, fn_outputs = fgraph
    else:
        fn_inputs = fgraph.inputs
        fn_outputs = fgraph.outputs

    fn_inputs = [i for i in fn_inputs if not isinstance(i, SharedVariable)]

    pytensor_py_fn = function(
        fn_inputs, fn_outputs, mode=py_mode, accept_inplace=True, updates=updates
    )
    py_res = pytensor_py_fn(*inputs)

    pytensor_numba_fn = function(
        fn_inputs,
        fn_outputs,
        mode=numba_mode,
        accept_inplace=True,
        updates=updates,
    )
    numba_res = pytensor_numba_fn(*inputs)

    # Get some coverage
    if eval_obj_mode:
        eval_python_only(fn_inputs, fn_outputs, inputs, mode=numba_mode)

    if len(fn_outputs) > 1:
        for j, p in zip(numba_res, py_res, strict=True):
            assert_fn(j, p)
    else:
        assert_fn(numba_res[0], py_res[0])

    return pytensor_numba_fn, numba_res


@pytest.mark.parametrize(
    "v, expected, force_scalar, not_implemented",
    [
        (MyType(), None, False, True),
        (ps.float32, numba.types.float32, False, False),
        (pt.fscalar, numba.types.Array(numba.types.float32, 0, "A"), False, False),
        (pt.fscalar, numba.types.float32, True, False),
        (pt.lvector, numba.types.int64[:], False, False),
        (pt.dmatrix, numba.types.float64[:, :], False, False),
        (pt.dmatrix, numba.types.float64, True, False),
    ],
)
def test_get_numba_type(v, expected, force_scalar, not_implemented):
    cm = (
        contextlib.suppress()
        if not not_implemented
        else pytest.raises(NotImplementedError)
    )
    with cm:
        res = numba_basic.get_numba_type(v, force_scalar=force_scalar)
        assert res == expected


@pytest.mark.parametrize(
    "v, expected, force_scalar",
    [
        (Apply(MyOp(), [], []), numba.types.void(), False),
        (Apply(MyOp(), [], []), numba.types.void(), True),
        (
            Apply(MyOp(), [pt.lvector()], []),
            numba.types.void(numba.types.int64[:]),
            False,
        ),
        (Apply(MyOp(), [pt.lvector()], []), numba.types.void(numba.types.int64), True),
        (
            Apply(MyOp(), [pt.dmatrix(), ps.float32()], [pt.dmatrix()]),
            numba.types.float64[:, :](numba.types.float64[:, :], numba.types.float32),
            False,
        ),
        (
            Apply(MyOp(), [pt.dmatrix(), ps.float32()], [pt.dmatrix()]),
            numba.types.float64(numba.types.float64, numba.types.float32),
            True,
        ),
        (
            Apply(MyOp(), [pt.dmatrix(), ps.float32()], [pt.dmatrix(), ps.int32()]),
            numba.types.Tuple([numba.types.float64[:, :], numba.types.int32])(
                numba.types.float64[:, :], numba.types.float32
            ),
            False,
        ),
        (
            Apply(MyOp(), [pt.dmatrix(), ps.float32()], [pt.dmatrix(), ps.int32()]),
            numba.types.Tuple([numba.types.float64, numba.types.int32])(
                numba.types.float64, numba.types.float32
            ),
            True,
        ),
    ],
)
def test_create_numba_signature(v, expected, force_scalar):
    res = numba_basic.create_numba_signature(v, force_scalar=force_scalar)
    assert res == expected


@pytest.mark.parametrize(
    "x, i",
    [
        (np.zeros((20, 3)), 1),
    ],
)
def test_Shape(x, i):
    g = Shape()(pt.as_tensor_variable(x))
    g_fg = FunctionGraph([], [g])

    compare_numba_and_py(g_fg, [])

    g = Shape_i(i)(pt.as_tensor_variable(x))
    g_fg = FunctionGraph([], [g])

    compare_numba_and_py(g_fg, [])


@pytest.mark.parametrize(
    "v, shape, ndim",
    [
        (set_test_value(pt.vector(), np.array([4], dtype=config.floatX)), (), 0),
        (set_test_value(pt.vector(), np.arange(4, dtype=config.floatX)), (2, 2), 2),
        (
            set_test_value(pt.vector(), np.arange(4, dtype=config.floatX)),
            set_test_value(pt.lvector(), np.array([2, 2], dtype="int64")),
            2,
        ),
    ],
)
def test_Reshape(v, shape, ndim):
    g = Reshape(ndim)(v, shape)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


def test_Reshape_scalar():
    v = pt.vector()
    v.tag.test_value = np.array([1.0], dtype=config.floatX)
    g = Reshape(1)(v[0], (1,))
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


@pytest.mark.parametrize(
    "v, shape, fails",
    [
        (
            set_test_value(pt.matrix(), np.array([[1.0]], dtype=config.floatX)),
            (1, 1),
            False,
        ),
        (
            set_test_value(pt.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            (1, 1),
            True,
        ),
        (
            set_test_value(pt.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            (1, None),
            False,
        ),
    ],
)
def test_SpecifyShape(v, shape, fails):
    g = SpecifyShape()(v, *shape)
    g_fg = FunctionGraph(outputs=[g])
    cm = contextlib.suppress() if not fails else pytest.raises(AssertionError)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "v",
    [
        set_test_value(pt.vector(), np.arange(4, dtype=config.floatX)),
    ],
)
def test_ViewOp(v):
    g = ViewOp()(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, SharedVariable | Constant)
        ],
    )


@pytest.mark.parametrize(
    "inputs, op, exc",
    [
        (
            [
                set_test_value(
                    pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)
                ),
                set_test_value(pt.lmatrix(), rng.poisson(size=(2, 3))),
            ],
            MySingleOut,
            UserWarning,
        ),
        (
            [
                set_test_value(
                    pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)
                ),
                set_test_value(pt.lmatrix(), rng.poisson(size=(2, 3))),
            ],
            MyMultiOut,
            UserWarning,
        ),
    ],
)
def test_perform(inputs, op, exc):
    g = op()(*inputs)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


def test_perform_params():
    """This tests for `Op.perform` implementations that require the `params` arguments."""

    x = pt.vector()
    x.tag.test_value = np.array([1.0, 2.0], dtype=config.floatX)

    out = assert_op(x, np.array(True))

    if not isinstance(out, list | tuple):
        out = [out]

    out_fg = FunctionGraph([x], out)
    compare_numba_and_py(out_fg, [get_test_value(i) for i in out_fg.inputs])


def test_perform_type_convert():
    """This tests the use of `Type.filter` in `objmode`.

    The `Op.perform` takes a single input that it returns as-is, but it gets a
    native scalar and it's supposed to return an `np.ndarray`.
    """

    x = pt.vector()
    x.tag.test_value = np.array([1.0, 2.0], dtype=config.floatX)

    out = assert_op(x.sum(), np.array(True))

    if not isinstance(out, list | tuple):
        out = [out]

    out_fg = FunctionGraph([x], out)
    compare_numba_and_py(out_fg, [get_test_value(i) for i in out_fg.inputs])


@pytest.mark.parametrize(
    "x, y, exc",
    [
        (
            set_test_value(pt.matrix(), rng.random(size=(3, 2)).astype(config.floatX)),
            set_test_value(pt.vector(), rng.random(size=(2,)).astype(config.floatX)),
            None,
        ),
        (
            set_test_value(
                pt.matrix(dtype="float64"), rng.random(size=(3, 2)).astype("float64")
            ),
            set_test_value(
                pt.vector(dtype="float32"), rng.random(size=(2,)).astype("float32")
            ),
            None,
        ),
        (
            set_test_value(pt.lmatrix(), rng.poisson(size=(3, 2))),
            set_test_value(pt.fvector(), rng.random(size=(2,)).astype("float32")),
            None,
        ),
        (
            set_test_value(pt.lvector(), rng.random(size=(2,)).astype(np.int64)),
            set_test_value(pt.lvector(), rng.random(size=(2,)).astype(np.int64)),
            None,
        ),
    ],
)
def test_Dot(x, y, exc):
    g = ptm.Dot()(x, y)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "x, exc",
    [
        (
            set_test_value(ps.float64(), np.array(0.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(ps.float64(), np.array(-32.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(ps.float64(), np.array(-40.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(ps.float64(), np.array(32.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(ps.float64(), np.array(40.0, dtype="float64")),
            None,
        ),
        (
            set_test_value(ps.int64(), np.array(32, dtype="int64")),
            None,
        ),
    ],
)
def test_Softplus(x, exc):
    g = psm.Softplus(ps.upgrade_to_float)(x)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


@pytest.mark.parametrize(
    "x, y, exc",
    [
        (
            set_test_value(
                pt.dtensor3(),
                rng.random(size=(2, 3, 3)).astype("float64"),
            ),
            set_test_value(
                pt.dtensor3(),
                rng.random(size=(2, 3, 3)).astype("float64"),
            ),
            None,
        ),
        (
            set_test_value(
                pt.dtensor3(),
                rng.random(size=(2, 3, 3)).astype("float64"),
            ),
            set_test_value(
                pt.ltensor3(),
                rng.poisson(size=(2, 3, 3)).astype("int64"),
            ),
            None,
        ),
    ],
)
def test_BatchedDot(x, y, exc):
    g = blas.BatchedDot()(x, y)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, SharedVariable | Constant)
            ],
        )


def test_shared():
    a = shared(np.array([1, 2, 3], dtype=config.floatX))

    pytensor_numba_fn = function([], a, mode="NUMBA")
    numba_res = pytensor_numba_fn()

    np.testing.assert_allclose(numba_res, a.get_value())

    pytensor_numba_fn = function([], a * 2, mode="NUMBA")
    numba_res = pytensor_numba_fn()

    np.testing.assert_allclose(numba_res, a.get_value() * 2)

    # Changed the shared value and make sure that the Numba-compiled function
    # also changes.
    new_a_value = np.array([3, 4, 5], dtype=config.floatX)
    a.set_value(new_a_value)

    numba_res = pytensor_numba_fn()
    np.testing.assert_allclose(numba_res, new_a_value * 2)


def test_shared_updates():
    a = shared(0)

    pytensor_numba_fn = function([], a, updates={a: a + 1}, mode="NUMBA")
    res1, res2 = pytensor_numba_fn(), pytensor_numba_fn()
    assert res1 == 0
    assert res2 == 1
    assert a.get_value() == 2

    a.set_value(5)
    res1, res2 = pytensor_numba_fn(), pytensor_numba_fn()
    assert res1 == 5
    assert res2 == 6
    assert a.get_value() == 7


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
    "inputs, cond_fn, true_vals, false_vals",
    [
        ([], lambda: np.array(True), np.r_[1, 2, 3], np.r_[-1, -2, -3]),
        (
            [set_test_value(pt.dscalar(), np.array(0.2, dtype=np.float64))],
            lambda x: x < 0.5,
            np.r_[1, 2, 3],
            np.r_[-1, -2, -3],
        ),
        (
            [
                set_test_value(pt.dscalar(), np.array(0.3, dtype=np.float64)),
                set_test_value(pt.dscalar(), np.array(0.5, dtype=np.float64)),
            ],
            lambda x, y: x > y,
            x,
            y,
        ),
        (
            [
                set_test_value(pt.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
                set_test_value(pt.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
            ],
            lambda x, y: pt.all(x > y),
            x,
            y,
        ),
        (
            [
                set_test_value(pt.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
                set_test_value(pt.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
            ],
            lambda x, y: pt.all(x > y),
            [x, 2 * x],
            [y, 3 * y],
        ),
        (
            [
                set_test_value(pt.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
                set_test_value(pt.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
            ],
            lambda x, y: pt.all(x > y),
            [x, 2 * x],
            [y, 3 * y],
        ),
    ],
)
def test_IfElse(inputs, cond_fn, true_vals, false_vals):
    out = ifelse(cond_fn(*inputs), true_vals, false_vals)

    if not isinstance(out, list):
        out = [out]

    out_fg = FunctionGraph(inputs, out)

    compare_numba_and_py(out_fg, [get_test_value(i) for i in out_fg.inputs])


@pytest.mark.xfail(reason="https://github.com/numba/numba/issues/7409")
def test_config_options_parallel():
    x = pt.dvector()

    with config.change_flags(numba__vectorize_target="parallel"):
        pytensor_numba_fn = function([x], pt.sum(x), mode=numba_mode)
        numba_mul_fn = pytensor_numba_fn.vm.jit_fn.py_func.__globals__["impl_sum"]
        assert numba_mul_fn.targetoptions["parallel"] is True


def test_config_options_fastmath():
    x = pt.dvector()

    with config.change_flags(numba__fastmath=True):
        pytensor_numba_fn = function([x], pt.sum(x), mode=numba_mode)
        print(list(pytensor_numba_fn.vm.jit_fn.py_func.__globals__))
        numba_mul_fn = pytensor_numba_fn.vm.jit_fn.py_func.__globals__["impl_sum"]
        assert numba_mul_fn.targetoptions["fastmath"] is True


def test_config_options_cached():
    x = pt.dvector()

    with config.change_flags(numba__cache=True):
        pytensor_numba_fn = function([x], pt.sum(x), mode=numba_mode)
        numba_mul_fn = pytensor_numba_fn.vm.jit_fn.py_func.__globals__["impl_sum"]
        assert not isinstance(numba_mul_fn._cache, numba.core.caching.NullCache)

    with config.change_flags(numba__cache=False):
        pytensor_numba_fn = function([x], pt.sum(x), mode=numba_mode)
        numba_mul_fn = pytensor_numba_fn.vm.jit_fn.py_func.__globals__["impl_sum"]
        assert isinstance(numba_mul_fn._cache, numba.core.caching.NullCache)


def test_scalar_return_value_conversion():
    r"""Make sure that we convert \"native\" scalars to `ndarray`\s in the graph outputs."""
    x = pt.scalar(name="x")
    x_fn = function(
        [x],
        2 * x,
        mode=numba_mode,
    )
    assert isinstance(x_fn(1.0), np.ndarray)


def test_OpFromGraph():
    x, y, z = pt.matrices("xyz")
    ofg_1 = OpFromGraph([x, y], [x + y], inline=False)
    ofg_2 = OpFromGraph([x, y], [x * y, x - y], inline=False)

    o1, o2 = ofg_2(y, z)
    out = ofg_1(x, o1) + o2

    xv = np.ones((2, 2), dtype=config.floatX)
    yv = np.ones((2, 2), dtype=config.floatX) * 3
    zv = np.ones((2, 2), dtype=config.floatX) * 5

    compare_numba_and_py(((x, y, z), (out,)), [xv, yv, zv])


@pytest.mark.filterwarnings("error")
def test_cache_warning_suppressed():
    x = pt.vector("x", shape=(5,), dtype="float64")
    out = pt.psi(x) * 2
    fn = function([x], out, mode="NUMBA")

    x_test = np.random.uniform(size=5)
    np.testing.assert_allclose(fn(x_test), scipy.special.psi(x_test) * 2)
