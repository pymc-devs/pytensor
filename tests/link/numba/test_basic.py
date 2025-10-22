import contextlib
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import pytest
import scipy

from pytensor.compile import SymbolicInput


numba = pytest.importorskip("numba")

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor import config, shared
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.compile.ops import ViewOp
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.type import Type
from pytensor.ifelse import ifelse
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.linker import NumbaLinker
from pytensor.scalar.basic import ScalarOp, as_scalar
from pytensor.tensor.elemwise import Elemwise


if TYPE_CHECKING:
    from pytensor.graph.basic import Variable


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
numba_inplace_mode = numba_mode.including("inplace")
py_mode = Mode("py", opts)

rng = np.random.default_rng(42849)


def compare_shape_dtype(x, y):
    return x.shape == y.shape and x.dtype == y.dtype


def eval_python_only(fn_inputs, fn_outputs, inputs, mode=numba_mode):
    """Evaluate the Numba implementation in pure Python for coverage purposes."""

    def py_tuple_setitem(t, i, v):
        ll = list(t)
        ll[i] = v
        return tuple(ll)

    def njit_noop(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        else:
            return lambda x: x

    mocks = [
        mock.patch("numba.njit", njit_noop),
        mock.patch(
            "pytensor.link.numba.dispatch.basic.tuple_setitem", py_tuple_setitem
        ),
        mock.patch("pytensor.link.numba.dispatch.basic.numba_njit", njit_noop),
        mock.patch(
            "pytensor.link.numba.dispatch.basic.direct_cast", lambda x, dtype: x
        ),
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
    graph_inputs: Iterable[Variable],
    graph_outputs: Variable | Iterable[Variable],
    test_inputs: Iterable,
    *,
    assert_fn: Callable | None = None,
    numba_mode=numba_mode,
    py_mode=py_mode,
    updates=None,
    inplace: bool = False,
    eval_obj_mode: bool = True,
) -> tuple[Callable, Any]:
    """Function to compare python function output and Numba compiled output for testing equality

    The inputs and outputs are then passed to this function which then compiles the given function in both
    numba and python, runs the calculation in both and checks if the results are the same

    Parameters
    ----------
    graph_inputs:
        Symbolic inputs to the graph
    graph_outputs:
        Symbolic outputs of the graph
    test_inputs
        Numerical inputs with which to evaluate the graph.
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
            np.testing.assert_allclose(x, y, rtol=1e-4, strict=True)
            # Make sure we don't have one input be a np.ndarray while the other is not
            if isinstance(x, np.ndarray):
                assert isinstance(y, np.ndarray), "y is not a NumPy array, but x is"
            else:
                assert not isinstance(y, np.ndarray), "y is a NumPy array, but x is not"

    if any(
        inp.owner is not None
        for inp in graph_inputs
        if not isinstance(inp, SymbolicInput)
    ):
        raise ValueError("Inputs must be root variables")

    pytensor_py_fn = function(
        graph_inputs, graph_outputs, mode=py_mode, accept_inplace=True, updates=updates
    )

    test_inputs_copy = (inp.copy() for inp in test_inputs) if inplace else test_inputs
    py_res = pytensor_py_fn(*test_inputs_copy)

    # Get some coverage (and catch errors in python mode before unreadable numba ones)
    if eval_obj_mode:
        test_inputs_copy = (
            (inp.copy() for inp in test_inputs) if inplace else test_inputs
        )
        eval_python_only(graph_inputs, graph_outputs, test_inputs_copy, mode=numba_mode)

    pytensor_numba_fn = function(
        graph_inputs,
        graph_outputs,
        mode=numba_mode,
        accept_inplace=True,
        updates=updates,
    )
    test_inputs_copy = (inp.copy() for inp in test_inputs) if inplace else test_inputs
    numba_res = pytensor_numba_fn(*test_inputs_copy)
    if isinstance(graph_outputs, tuple | list):
        for numba_res_i, python_res_i in zip(numba_res, py_res, strict=True):
            assert_fn(numba_res_i, python_res_i)
    else:
        assert_fn(numba_res, py_res)

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


def test_ViewOp():
    v = pt.vector()
    v_test_value = np.arange(4, dtype=config.floatX)
    g = ViewOp()(v)

    compare_numba_and_py(
        [v],
        [g],
        [v_test_value],
    )


@pytest.mark.parametrize(
    "inputs, op, exc",
    [
        (
            [
                (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
                (pt.lmatrix(), rng.poisson(size=(2, 3))),
            ],
            MySingleOut,
            UserWarning,
        ),
        (
            [
                (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
                (pt.lmatrix(), rng.poisson(size=(2, 3))),
            ],
            MyMultiOut,
            UserWarning,
        ),
    ],
)
def test_perform(inputs, op, exc):
    inputs, test_values = zip(*inputs, strict=True)
    g = op()(*inputs)

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
            [(pt.dscalar(), np.array(0.2, dtype=np.float64))],
            lambda x: x < 0.5,
            np.r_[1, 2, 3],
            np.r_[-1, -2, -3],
        ),
        (
            [
                (pt.dscalar(), np.array(0.3, dtype=np.float64)),
                (pt.dscalar(), np.array(0.5, dtype=np.float64)),
            ],
            lambda x, y: x > y,
            x,
            y,
        ),
        (
            [
                (pt.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
                (pt.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
            ],
            lambda x, y: pt.all(x > y),
            x,
            y,
        ),
        (
            [
                (pt.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
                (pt.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
            ],
            lambda x, y: pt.all(x > y),
            [x, 2 * x],
            [y, 3 * y],
        ),
        (
            [
                (pt.dvector(), np.array([0.5, 0.9], dtype=np.float64)),
                (pt.dvector(), np.array([0.3, 0.1], dtype=np.float64)),
            ],
            lambda x, y: pt.all(x > y),
            [x, 2 * x],
            [y, 3 * y],
        ),
    ],
)
def test_IfElse(inputs, cond_fn, true_vals, false_vals):
    inputs, test_values = zip(*inputs, strict=True) if inputs else ([], [])
    out = ifelse(cond_fn(*inputs), true_vals, false_vals)
    compare_numba_and_py(inputs, out, test_values)


def test_config_options_fastmath():
    x = pt.dvector()

    with config.change_flags(numba__fastmath=True):
        pytensor_numba_fn = function([x], pt.sum(x), mode=numba_mode)
        numba_mul_fn = pytensor_numba_fn.vm.jit_fn.py_func.__globals__["impl_sum"]
        assert numba_mul_fn.targetoptions["fastmath"] == {
            "afn",
            "arcp",
            "contract",
            "nsz",
            "reassoc",
        }


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

    compare_numba_and_py([x, y, z], [out], [xv, yv, zv])


@pytest.mark.filterwarnings("error")
def test_ofg_inner_inplace():
    x = pt.vector("x")
    set0 = x[0].set(1)  # SetSubtensor should not inplace on x
    exp_x = pt.exp(x)
    set1 = exp_x[0].set(1)  # SetSubtensor should inplace on exp_x
    ofg0 = OpFromGraph([x], [set0])
    ofg1 = OpFromGraph([x], [set1])

    y, z = pt.vectors("y", "z")
    fn = function([y, z], [ofg0(y), ofg1(z)], mode="NUMBA")

    fn_ofg0 = fn.maker.fgraph.outputs[0].owner.op
    assert isinstance(fn_ofg0, OpFromGraph)
    fn_set0 = fn_ofg0.fgraph.outputs[0]
    assert fn_set0.owner.op.destroy_map == {}

    fn_ofg1 = fn.maker.fgraph.outputs[1].owner.op
    assert isinstance(fn_ofg1, OpFromGraph)
    fn_set1 = fn_ofg1.fgraph.outputs[0]
    assert fn_set1.owner.op.destroy_map == {0: [0]}

    x_test = np.array([0, 1, 1], dtype=config.floatX)
    y_test = np.array([0, 1, 1], dtype=config.floatX)
    res0, res1 = fn(x_test, y_test)
    # Check inputs were not mutated
    np.testing.assert_allclose(x_test, [0, 1, 1])
    np.testing.assert_allclose(y_test, [0, 1, 1])
    # Check outputs are correct
    np.testing.assert_allclose(res0, [1, 1, 1])
    np.testing.assert_allclose(res1, [1, np.e, np.e])


@pytest.mark.filterwarnings("error")
def test_cache_warning_suppressed():
    x = pt.vector("x", shape=(5,), dtype="float64")
    out = pt.psi(x) * 2
    fn = function([x], out, mode="NUMBA")

    x_test = np.random.uniform(size=5)
    np.testing.assert_allclose(fn(x_test), scipy.special.psi(x_test) * 2)


@pytest.mark.parametrize("mode", ("default", "trust_input", "direct"))
def test_function_overhead(mode, benchmark):
    x = pt.vector("x")
    out = pt.exp(x)

    fn = function([x], out, mode="NUMBA")
    if mode == "trust_input":
        fn.trust_input = True
    elif mode == "direct":
        fn = fn.vm.jit_fn

    test_x = np.zeros(1000)
    assert np.sum(fn(test_x)) == 1000

    benchmark(fn, test_x)
