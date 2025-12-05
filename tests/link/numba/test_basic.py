import contextlib
import copy
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any
from unittest import mock

import numpy as np
import pytest
import scipy

from pytensor.tensor import scalar_from_tensor


numba = pytest.importorskip("numba")

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor import config, shared
from pytensor.compile import SymbolicInput
from pytensor.compile.function import function
from pytensor.compile.mode import Mode
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.type import Type
from pytensor.link.numba.dispatch import basic as numba_basic
from pytensor.link.numba.dispatch.basic import (
    _filter_numba_warnings,
    cache_key_for_constant,
    numba_funcify_and_cache_key,
)
from pytensor.link.numba.linker import NumbaLinker
from pytensor.scalar.basic import Composite, ScalarOp, as_scalar
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.utils import hash_from_ndarray


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
        def add_py_func_attr(x):
            x.py_func = x
            return x

        if len(args) == 1 and callable(args[0]):
            return add_py_func_attr(args[0])
        else:
            return lambda x: add_py_func_attr(x)

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

    test_input_deepcopy = None
    if not inplace:
        test_input_deepcopy = [
            i.copy() if isinstance(i, np.ndarray) else copy.deepcopy(i)
            for i in test_inputs
        ]

    pytensor_py_fn = function(
        graph_inputs,
        graph_outputs,
        mode=py_mode,
        accept_inplace=inplace,
        updates=updates,
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
        accept_inplace=inplace,
        updates=updates,
    )
    test_inputs_copy = (inp.copy() for inp in test_inputs) if inplace else test_inputs
    numba_res = pytensor_numba_fn(*test_inputs_copy)

    if not inplace:
        # Check we did not accidentally modify the inputs inplace
        for test_input, test_input_copy in zip(test_inputs, test_input_deepcopy):
            try:
                assert_fn(test_input, test_input_copy)
            except AssertionError as e:
                raise AssertionError("Inputs were modified inplace") from e

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
    with pytest.warns(FutureWarning, match="deprecated"):
        res = numba_basic.create_numba_signature(v, force_scalar=force_scalar)
    assert res == expected


@pytest.mark.parametrize(
    "inputs, op",
    [
        (
            [
                (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
                (pt.lmatrix(), rng.poisson(size=(2, 3))),
            ],
            MySingleOut,
        ),
        (
            [
                (pt.matrix(), rng.random(size=(2, 3)).astype(config.floatX)),
                (pt.lmatrix(), rng.poisson(size=(2, 3))),
            ],
            MyMultiOut,
        ),
    ],
)
def test_fallback_perform(inputs, op):
    inputs, test_values = zip(*inputs, strict=True)
    g = op()(*inputs)

    if isinstance(g, list):
        outputs = g
    else:
        outputs = [g]

    with pytest.warns(UserWarning):
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


def test_config_options_fastmath():
    x = pt.dvector()

    with config.change_flags(numba__fastmath=True):
        pytensor_numba_fn = function([x], pt.sum(x), mode=numba_mode)
        numba_sum_fn = pytensor_numba_fn.vm.jit_fn.py_func.__globals__[
            "jitable_func"
        ].py_func.__globals__["impl_sum"]
        assert numba_sum_fn.targetoptions["fastmath"] == {
            "afn",
            "arcp",
            "contract",
            "nsz",
            "reassoc",
        }

    with config.change_flags(numba__fastmath=False):
        pytensor_numba_fn = function([x], pt.sum(x), mode=numba_mode)
        numba_sum_fn = pytensor_numba_fn.vm.jit_fn.py_func.__globals__[
            "jitable_func"
        ].py_func.__globals__["impl_sum"]
        assert numba_sum_fn.targetoptions["fastmath"] is False


def test_config_options_cached():
    x = pt.dvector()

    with config.change_flags(numba__cache=True):
        pytensor_numba_fn = function([x], pt.sum(x), mode=numba_mode)
        numba_sum_fn = pytensor_numba_fn.vm.jit_fn.py_func.__globals__[
            "jitable_func"
        ].py_func.__globals__["impl_sum"]
        assert not isinstance(numba_sum_fn._cache, numba.core.caching.NullCache)

    with config.change_flags(numba__cache=False):
        pytensor_numba_fn = function([x], pt.sum(x), mode=numba_mode)
        # Without caching we don't wrap the function in jitable_func
        numba_sum_fn = pytensor_numba_fn.vm.jit_fn.py_func.__globals__["impl_sum"]
        assert isinstance(numba_sum_fn._cache, numba.core.caching.NullCache)


def test_scalar_return_value_conversion():
    r"""Make sure that we convert \"native\" scalars to `ndarray`\s in the graph outputs."""
    x = pt.scalar(name="x")
    x_fn = function(
        [x],
        2 * x,
        mode=numba_mode,
    )
    assert isinstance(x_fn(1.0), np.ndarray)


class TestNumbaWarnings:
    def setup_method(self, method):
        # Pytest messes up with the package filters, reenable here for testing
        _filter_numba_warnings()

    @pytest.mark.filterwarnings("error")
    def test_cache_pointer_func_warning_suppressed(self):
        x = pt.vector("x", shape=(5,), dtype="float64")
        out = pt.psi(x) * 2
        fn = function([x], out, mode="NUMBA")

        x_test = np.random.uniform(size=5)
        np.testing.assert_allclose(fn(x_test), scipy.special.psi(x_test) * 2)

    @pytest.mark.filterwarnings("error")
    def test_cache_large_global_array_warning_suppressed(self):
        rng = np.random.default_rng(458)
        large_constant = rng.normal(size=(100000, 5))

        x = pt.vector("x", shape=(5,), dtype="float64")
        out = x * large_constant
        fn = function([x], out, mode="NUMBA")

        x_test = rng.uniform(size=5)
        np.testing.assert_allclose(fn(x_test), x_test * large_constant)

    @pytest.mark.filterwarnings("error")
    def test_contiguous_array_dot_warning_suppressed(self):
        A = pt.matrix("A")
        b = pt.vector("b")
        out = pt.dot(A, b[:, None])
        # Cached functions won't reemit the warning, so we have to disable it
        with config.change_flags(numba__cache=False):
            fn = function([A, b], out, mode="NUMBA")

        A_test = np.ones((5, 5))
        # Numba actually warns even on contiguous arrays: https://github.com/numba/numba/issues/10086
        # But either way we don't want this warning for users as they have little control over strides
        b_test = np.ones((10,))[::2]
        np.testing.assert_allclose(fn(A_test, b_test), np.dot(A_test, b_test[:, None]))


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


class ComplexType:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class TestKeyForConstant:
    def test_numpy_scalars(self):
        key_float64_0 = cache_key_for_constant(np.float64(0))
        key_float64_0_again = cache_key_for_constant(np.float64(0))
        key_int64_0 = cache_key_for_constant(np.float32(0))
        assert key_float64_0 == key_float64_0_again
        assert key_float64_0 != key_int64_0

    def test_None(self):
        key_none_1 = cache_key_for_constant(None)
        key_none_2 = cache_key_for_constant(None)
        assert key_none_1 == key_none_2

    def test_python_scalars(self):
        key_int_0 = cache_key_for_constant(0)
        key_int_0_again = cache_key_for_constant(0)
        key_float_0 = cache_key_for_constant(0.0)
        assert key_int_0 == key_int_0_again
        assert key_int_0 != key_float_0

    def test_numpy_arrays(self):
        # Jest check we are using hash_from_ndarary and trust that is working
        # If we change our implementation we may need more exhaustive tests here
        arr1 = np.array([1, 2, 3], dtype=np.float32)
        arr2 = np.array([1, 3, 2], dtype=np.float32)

        key_arr1 = cache_key_for_constant(arr1)
        expected_key_arr1 = hash_from_ndarray(arr1)

        key_arr2 = cache_key_for_constant(arr2)
        expected_key_arr2 = hash_from_ndarray(arr2)

        assert key_arr1 == expected_key_arr1
        assert key_arr2 == expected_key_arr2
        assert key_arr1 != key_arr2

    def test_complex_types(self):
        obj1 = ComplexType(1, 2)
        ob1_again = ComplexType(1, 2)
        obj2 = ComplexType(3, 4)

        key_obj1 = cache_key_for_constant(obj1)
        key_obj1_again = cache_key_for_constant(ob1_again)
        key_obj2 = cache_key_for_constant(obj2)

        assert key_obj1 == key_obj1_again
        assert key_obj1 != key_obj2


def test_funcify_dispatch_interop():
    """Test that the different funcify registration decorators work together as expected."""

    class BaseOp(Op):
        itypes = [pt.dscalar]
        otypes = [pt.dscalar]

    class FuncifiedOp(BaseOp):
        def perform(self, node, inputs, outputs):
            outputs[0][0] = inputs[0] + 1

    class FuncifiedAndCachedOp(BaseOp):
        def perform(self, node, inputs, outputs):
            outputs[0][0] = inputs[0] * 2

    class FuncifiedAndDefaultCachedOp(BaseOp):
        __props__ = ()

        def perform(self, node, inputs, outputs):
            outputs[0][0] = inputs[0] - 3

    @numba_basic.numba_funcify.register(FuncifiedOp)
    def _(op, node, **kwargs):
        @numba_basic.numba_njit
        def impl(x):
            return x + 1

        return impl

    @numba_basic.register_funcify_and_cache_key(FuncifiedAndCachedOp)
    def _(op, node, **kwargs):
        @numba_basic.numba_njit
        def impl(x):
            return x * 2

        return impl, "sushi-hash"

    @numba_basic.register_funcify_default_op_cache_key(FuncifiedAndDefaultCachedOp)
    def _(op, node, **kwargs):
        @numba_basic.numba_njit
        def impl(x):
            return x - 3

        return impl

    x = pt.scalar("x", dtype="float64")
    outs = [
        FuncifiedOp()(x),
        FuncifiedAndCachedOp()(x),
        FuncifiedAndDefaultCachedOp()(x),
    ]
    test_x = np.array(5.0)

    compare_numba_and_py(
        [x],
        outs,
        [test_x],
    )

    # Test we can use numba_funcify_ensure_cache
    fn0, cache0 = numba_basic.numba_funcify_ensure_cache(
        outs[0].owner.op, outs[0].owner
    )
    assert cache0 is None
    assert numba.njit(lambda x: fn0(x))(test_x) == 6
    fn1, cache1 = numba_basic.numba_funcify_ensure_cache(
        outs[1].owner.op, outs[1].owner
    )
    assert cache1 == "sushi-hash"
    assert numba.njit(lambda x: fn1(x))(test_x) == 10
    fn2, cache2 = numba_basic.numba_funcify_ensure_cache(
        outs[2].owner.op, outs[2].owner
    )
    assert cache2 is not None
    assert numba.njit(lambda x: fn2(x))(test_x) == 2
    fn2_again, cache2_again = numba_basic.numba_funcify_ensure_cache(
        outs[2].owner.op, outs[2].owner
    )
    assert cache2 == cache2_again
    assert numba.njit(lambda x: fn2_again(x))(test_x) == 2

    # Test we can use numba_funcify directly
    fn0 = numba_basic.numba_funcify(outs[0].owner.op, outs[0].owner)
    assert numba.njit(lambda x: fn0(x))(test_x) == 6
    fn1 = numba_basic.numba_funcify(outs[1].owner.op, outs[1].owner)
    assert numba.njit(lambda x: fn1(x))(test_x) == 10
    fn2 = numba_basic.numba_funcify(outs[2].owner.op, outs[2].owner)
    assert numba.njit(lambda x: fn2(x))(test_x) == 2

    # Test we can use numba_funcify_and_cache_key directly
    fn0, cache0 = numba_basic.numba_funcify_and_cache_key(
        outs[0].owner.op, outs[0].owner
    )
    assert cache0 is None
    assert numba.njit(lambda x: fn0(x))(test_x) == 6
    fn1, cache1 = numba_basic.numba_funcify_and_cache_key(
        outs[1].owner.op, outs[1].owner
    )
    assert cache1 == "sushi-hash"
    assert numba.njit(lambda x: fn1(x))(test_x) == 10
    fn2, cache2 = numba_basic.numba_funcify_and_cache_key(
        outs[2].owner.op, outs[2].owner
    )
    assert cache2 is not None
    assert numba.njit(lambda x: fn2(x))(test_x) == 2
    fn2_again, cache2_again = numba_basic.numba_funcify_and_cache_key(
        outs[2].owner.op, outs[2].owner
    )
    assert cache2 == cache2_again
    assert numba.njit(lambda x: fn2_again(x))(test_x) == 2

    # Test numba_funcify_default_op_cache_key works as expected
    with pytest.raises(NotImplementedError):
        numba_basic.numba_funcify_default_op_cache_key(outs[0].owner.op, outs[0].owner)
    with pytest.raises(NotImplementedError):
        numba_basic.numba_funcify_default_op_cache_key(outs[1].owner.op, outs[1].owner)
    fn2_def_cached = numba_basic.numba_funcify_default_op_cache_key(
        outs[2].owner.op, outs[2].owner
    )
    assert numba.njit(lambda x: fn2_def_cached(x))(test_x) == 2


class TestFgraphCacheKey:
    @staticmethod
    def generate_and_validate_key(fg):
        _, key = numba_funcify_and_cache_key(fg)
        assert key is not None
        _, key_again = numba_funcify_and_cache_key(fg)
        assert key == key_again  # Check its stable
        return key

    def test_node_order(self):
        x = pt.scalar("x")
        log_x = pt.log(x)
        graphs = [
            pt.exp(x) / log_x,
            log_x / pt.exp(x),
            pt.exp(log_x) / x,
            x / pt.exp(log_x),
            pt.exp(log_x) / log_x,
            log_x / pt.exp(log_x),
        ]

        keys = []
        for graph in graphs:
            fg = FunctionGraph([x], [graph], clone=False)
            keys.append(self.generate_and_validate_key(fg))
        # Check keys are unique
        assert len(set(keys)) == len(graphs)

        # Extra unused input should alter the key, because it changes the function signature
        y = pt.scalar("y")
        for inputs in [[x, y], [y, x]]:
            fg = FunctionGraph(inputs, [graphs[0]], clone=False)
            keys.append(self.generate_and_validate_key(fg))
        assert len(set(keys)) == len(graphs) + 2

        # Adding an input as an output should also change the key
        for outputs in [
            [graphs[0], x],
            [x, graphs[0]],
            [x, x, graphs[0]],
            [x, graphs[0], x],
            [graphs[0], x, x],
        ]:
            fg = FunctionGraph([x], outputs, clone=False)
            keys.append(self.generate_and_validate_key(fg))
        assert len(set(keys)) == len(graphs) + 2 + 5

    def test_multi_output(self):
        x = pt.scalar("x")

        xs = scalar_from_tensor(x)
        out0, out1 = Elemwise(Composite([xs], [xs * 2, xs - 2]))(x)

        test_outs = [
            [out0],
            [out1],
            [out0, out1],
            [out1, out0],
        ]
        keys = []
        for test_out in test_outs:
            fg = FunctionGraph([x], test_out, clone=False)
            keys.append(self.generate_and_validate_key(fg))
        assert len(set(keys)) == len(test_outs)

    def test_constant_output(self):
        fg_pi = FunctionGraph([], [pt.constant(np.pi)])
        fg_e = FunctionGraph([], [pt.constant(np.e)])

        assert self.generate_and_validate_key(fg_pi) != self.generate_and_validate_key(
            fg_e
        )
