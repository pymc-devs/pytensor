from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.maker import function
from pytensor.compile.mode import JAX, Mode
from pytensor.compile.sharedvalue import shared
from pytensor.conditional.op import ifelse
from pytensor.configdefaults import config
from pytensor.graph import RewriteDatabaseQuery
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.link.jax import JAXLinker
from pytensor.raise_op import assert_op
from pytensor.tensor.type import dscalar, matrices, scalar, vector
from tests.conditional.utils import IfElseCompatibilityTests


@pytest.fixture(scope="module", autouse=True)
def set_pytensor_flags():
    with config.change_flags(cxx=""):
        yield


jax = pytest.importorskip("jax")

from pytensor.link.jax.dispatch import jax_funcify
from pytensor.link.jax.dispatch.shape import JAXShapeTuple


optimizer = RewriteDatabaseQuery(include=["jax"], exclude=JAX._optimizer.exclude)
jax_mode = Mode(linker=JAXLinker(), optimizer=optimizer)
py_mode = Mode(linker="py", optimizer=None)


def compare_jax_and_py(
    graph_inputs: Iterable[Variable],
    graph_outputs: Variable | Iterable[Variable],
    test_inputs: Iterable,
    *,
    assert_fn: Callable | None = None,
    must_be_device_array: bool = True,
    jax_mode=jax_mode,
    py_mode=py_mode,
):
    """Function to compare python function output and jax compiled output for testing equality

    The inputs and outputs are then passed to this function which then compiles the given function in both
    jax and python, runs the calculation in both and checks if the results are the same

    Parameters
    ----------
    graph_inputs:
        Symbolic inputs to the graph
    outputs:
        Symbolic outputs of the graph
    test_inputs: iter
        Numerical inputs for testing the function.
    assert_fn: func, opt
        Assert function used to check for equality between python and jax. If not
        provided uses np.testing.assert_allclose
    must_be_device_array: Bool
        Checks for instance of jax.interpreters.xla.DeviceArray. For testing purposes
        if this device array is found it indicates if the result was computed by jax

    Returns
    -------
    jax_res

    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    if any(inp.owner is not None for inp in graph_inputs):
        raise ValueError("Inputs must be root variables")

    pytensor_jax_fn = function(graph_inputs, graph_outputs, mode=jax_mode)
    jax_res = pytensor_jax_fn(*test_inputs)

    if must_be_device_array:
        if isinstance(jax_res, list):
            assert all(isinstance(res, jax.Array) for res in jax_res)
        else:
            assert isinstance(jax_res, jax.Array)

    pytensor_py_fn = function(graph_inputs, graph_outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    if isinstance(graph_outputs, list | tuple):
        for j, p in zip(jax_res, py_res, strict=True):
            assert_fn(j, p)
    else:
        assert_fn(jax_res, py_res)

    return pytensor_jax_fn, jax_res


def test_jax_FunctionGraph_once():
    """Make sure that an output is only computed once when it's referenced multiple times."""
    x = vector("x")
    y = vector("y")

    class TestOp(Op):
        def __init__(self):
            self.called = 0

        def make_node(self, *args):
            return Apply(self, list(args), [x.type() for x in args])

        def perform(self, inputs, outputs):
            for i, inp in enumerate(inputs):
                outputs[i][0] = inp[0]

    @jax_funcify.register(TestOp)
    def jax_funcify_TestOp(op, **kwargs):
        def func(*args, op=op):
            op.called += 1
            return list(args)

        return func

    op1 = TestOp()
    op2 = TestOp()

    q, r = op1(x, y)
    outs = op2(q + r, q + r)

    out_fg = FunctionGraph([x, y], outs, clone=False)
    assert len(out_fg.outputs) == 2

    out_jx = jax_funcify(out_fg)

    x_val = np.r_[1, 2].astype(config.floatX)
    y_val = np.r_[2, 3].astype(config.floatX)

    res = out_jx(x_val, y_val)
    assert len(res) == 2
    assert op1.called == 1
    assert op2.called == 1

    res = out_jx(x_val, y_val)
    assert len(res) == 2
    assert op1.called == 2
    assert op2.called == 2


def test_shared():
    a = shared(np.array([1, 2, 3], dtype=config.floatX))

    pytensor_jax_fn = function([], a, mode="JAX")
    jax_res = pytensor_jax_fn()

    assert isinstance(jax_res, jax.Array)
    np.testing.assert_allclose(jax_res, a.get_value())

    pytensor_jax_fn = function([], a * 2, mode="JAX")
    jax_res = pytensor_jax_fn()

    assert isinstance(jax_res, jax.Array)
    np.testing.assert_allclose(jax_res, a.get_value() * 2)

    # Changed the shared value and make sure that the JAX-compiled
    # function also changes.
    new_a_value = np.array([3, 4, 5], dtype=config.floatX)
    a.set_value(new_a_value)

    jax_res = pytensor_jax_fn()
    assert isinstance(jax_res, jax.Array)
    np.testing.assert_allclose(jax_res, new_a_value * 2)


def test_shared_updates():
    a = shared(0)

    pytensor_jax_fn = function([], a, updates={a: a + 1}, mode="JAX")
    res1, res2 = pytensor_jax_fn(), pytensor_jax_fn()
    assert res1 == 0
    assert res2 == 1
    assert a.get_value() == 2

    a.set_value(5)
    res1, res2 = pytensor_jax_fn(), pytensor_jax_fn()
    assert res1 == 5
    assert res2 == 6
    assert a.get_value() == 7


def test_jax_ifelse():
    true_vals = np.r_[1, 2, 3]
    false_vals = np.r_[-1, -2, -3]

    x = ifelse(np.array(True), true_vals, false_vals)

    compare_jax_and_py([], [x], [])

    a = dscalar("a")
    a_test = np.array(0.2, dtype=config.floatX)
    x = ifelse(a < 0.5, true_vals, false_vals)

    compare_jax_and_py([a], [x], [a_test])


class TestLazyIfElse(IfElseCompatibilityTests):
    mode = "JAX"
    check_aliasing = False  # JAX arrays are immutable and can safely share buffers.

    @staticmethod
    def check(x, message):
        # sqrt gives invalid inputs NaN derivatives, exposing eager execution of
        # unused branches through contamination of the selected branch's gradient.
        return pt.switch(pt.all(x > 0), x, pt.sqrt(x))

    @staticmethod
    def assert_invalid(fn, *args, message):
        assert np.isnan(fn(*args)).any()

    @staticmethod
    def gradient(fn, argnums):
        return jax.jit(
            jax.grad(lambda *args: fn.vm.jit_fn(*args)[0].sum(), argnums=argnums)
        )

    @pytest.mark.parametrize("shape_kind", ["scalar", "vector", "tuple"])
    def test_shared_shape_input(self, shape_kind):
        c = pt.iscalar("c")
        x, y = pt.vectors("x", "y")
        z = pt.matrix("z")
        if shape_kind == "scalar":
            shape = z.shape[0]
            reshape_shape = (shape, -1)
        else:
            dims = [z.shape[0], z.shape[1]]
            shape = pt.stack(dims) if shape_kind == "vector" else JAXShapeTuple()(*dims)
            reshape_shape = shape
        out = ifelse(c, x.reshape(reshape_shape, ndim=2).sum(1), y)
        # Returning shape keeps its computation outside the conditional branches.
        fn = function([c, x, y, z], [out, shape], mode=self.mode)
        x_value = np.arange(6, dtype=x.dtype)
        for rows in (2, 3):
            z_shape = (rows, 6 // rows)
            y_value = np.full(rows, 10, dtype=y.dtype)
            z_value = np.zeros(z_shape, dtype=z.dtype)
            for condition in (False, True):
                result, result_shape = fn(condition, x_value, y_value, z_value)
                expected = x_value.reshape(z_shape).sum(1) if condition else y_value
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(
                    result_shape, rows if shape_kind == "scalar" else z_shape
                )

    def test_shared_shape_arithmetic(self):
        c = pt.iscalar("c")
        x, y = pt.vectors("x", "y")
        z = pt.matrix("z")
        n = z.shape[0] * 2
        out = ifelse(c, x.reshape((n, -1)).sum(1), y)
        fn = function([c, x, y, z], [out, n], mode=self.mode)
        x_value = np.arange(12, dtype=x.dtype)
        y_value = np.full(4, 5, dtype=y.dtype)
        z_value = np.zeros((2, 3), dtype=z.dtype)
        for condition in (False, True):
            res, res_n = fn(condition, x_value, y_value, z_value)
            expected = x_value.reshape((4, 3)).sum(1) if condition else y_value
            np.testing.assert_array_equal(res, expected)
            np.testing.assert_array_equal(res_n, 4)

    def test_independent_scalar_shape_input(self):
        c = pt.iscalar("c")
        x, y = pt.vectors("x", "y")
        n = pt.lscalar("n")
        out = ifelse(c, x.reshape((n, -1)).sum(1), y)
        fn = function([c, x, y, n], out, mode=self.mode)
        x_value = np.arange(6, dtype=x.dtype)
        res_t = fn(True, x_value, np.zeros(2, dtype=y.dtype), 2)
        np.testing.assert_array_equal(res_t, x_value.reshape((2, 3)).sum(1))
        res_f = fn(False, x_value, np.full(2, 7, dtype=y.dtype), 2)
        np.testing.assert_array_equal(res_f, np.full(2, 7, dtype=y.dtype))
        # Recompilation with new static n value
        res_recompile = fn(True, x_value, np.zeros(3, dtype=y.dtype), 3)
        np.testing.assert_array_equal(res_recompile, x_value.reshape((3, 2)).sum(1))

    def test_alloc_shape_input(self):
        c = pt.iscalar("c")
        z = pt.matrix("z")
        n = z.shape[0]
        out_zeros = ifelse(c, pt.zeros((n, 3)), pt.ones((2, 3)))
        fn_zeros = function([c, z], [out_zeros, n], mode=self.mode)
        res_zeros, res_n = fn_zeros(True, np.zeros((2, 3)))
        assert res_zeros.shape == (2, 3)
        np.testing.assert_array_equal(res_n, 2)

    def test_user_static_argnums(self):
        from pytensor.compile.mode import Mode
        from pytensor.link.jax.linker import JAXLinker

        c = pt.iscalar("c")
        x, y = pt.vectors("x", "y")
        n = pt.lscalar("n")
        out = ifelse(c, x.reshape((n, -1)).sum(1), y)
        linker = JAXLinker(static_argnums=(3,))
        mode = Mode(linker=linker)
        fn = function([c, x, y, n], out, mode=mode)
        res = fn(True, np.arange(6.0), np.zeros(2), 2)
        np.testing.assert_array_equal(res, [3.0, 12.0])


def test_jax_checkandraise():
    p = scalar()

    res = assert_op(p, p < 1.0)

    with pytest.warns(UserWarning):
        function((p,), res, mode=jax_mode)


def test_OpFromGraph():
    x, y, z = matrices("xyz")
    ofg_1 = OpFromGraph([x, y], [x + y], inline=False)
    ofg_2 = OpFromGraph([x, y], [x * y, x - y], inline=False)

    o1, o2 = ofg_2(y, z)
    out = ofg_1(x, o1) + o2

    xv = np.ones((2, 2), dtype=config.floatX)
    yv = np.ones((2, 2), dtype=config.floatX) * 3
    zv = np.ones((2, 2), dtype=config.floatX) * 5

    compare_jax_and_py([x, y, z], [out], [xv, yv, zv])
