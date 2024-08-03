from collections.abc import Callable, Iterable
from functools import partial

import numpy as np
import pytest

from pytensor.compile.builders import OpFromGraph
from pytensor.compile.function import function
from pytensor.compile.mode import get_mode
from pytensor.compile.sharedvalue import SharedVariable, shared
from pytensor.configdefaults import config
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op, get_test_value
from pytensor.ifelse import ifelse
from pytensor.raise_op import assert_op
from pytensor.tensor.type import dscalar, matrices, scalar, vector


@pytest.fixture(scope="module", autouse=True)
def set_pytensor_flags():
    with config.change_flags(cxx="", compute_test_value="ignore"):
        yield


jax = pytest.importorskip("jax")


# We assume that the JAX mode includes all the rewrites needed to transpile JAX graphs
jax_mode = get_mode("JAX")
py_mode = get_mode("FAST_COMPILE")


def compare_jax_and_py(
    fgraph: FunctionGraph,
    test_inputs: Iterable,
    assert_fn: Callable | None = None,
    must_be_device_array: bool = True,
    jax_mode=jax_mode,
    py_mode=py_mode,
):
    """Function to compare python graph output and jax compiled output for testing equality

    In the tests below computational graphs are defined in PyTensor. These graphs are then passed to
    this function which then compiles the graphs in both jax and python, runs the calculation
    in both and checks if the results are the same

    Parameters
    ----------
    fgraph: FunctionGraph
        PyTensor function Graph object
    test_inputs: iter
        Numerical inputs for testing the function graph
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

    fn_inputs = [i for i in fgraph.inputs if not isinstance(i, SharedVariable)]
    pytensor_jax_fn = function(fn_inputs, fgraph.outputs, mode=jax_mode)
    jax_res = pytensor_jax_fn(*test_inputs)

    if must_be_device_array:
        if isinstance(jax_res, list):
            assert all(isinstance(res, jax.Array) for res in jax_res)
        else:
            assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)

    pytensor_py_fn = function(fn_inputs, fgraph.outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(jax_res, py_res, strict=True):
            assert_fn(j, p)
    else:
        assert_fn(jax_res, py_res)

    return pytensor_jax_fn, jax_res


def test_jax_FunctionGraph_once():
    """Make sure that an output is only computed once when it's referenced multiple times."""
    from pytensor.link.jax.dispatch import jax_funcify

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
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    a = dscalar("a")
    a.tag.test_value = np.array(0.2, dtype=config.floatX)
    x = ifelse(a < 0.5, true_vals, false_vals)
    x_fg = FunctionGraph([a], [x])  # I.e. False

    compare_jax_and_py(x_fg, [get_test_value(i) for i in x_fg.inputs])


def test_jax_checkandraise():
    p = scalar()
    p.tag.test_value = 0

    res = assert_op(p, p < 1.0)

    with pytest.warns(UserWarning):
        function((p,), res, mode=jax_mode)


def set_test_value(x, v):
    x.tag.test_value = v
    return x


def test_OpFromGraph():
    x, y, z = matrices("xyz")
    ofg_1 = OpFromGraph([x, y], [x + y], inline=False)
    ofg_2 = OpFromGraph([x, y], [x * y, x - y], inline=False)

    o1, o2 = ofg_2(y, z)
    out = ofg_1(x, o1) + o2
    out_fg = FunctionGraph([x, y, z], [out])

    xv = np.ones((2, 2), dtype=config.floatX)
    yv = np.ones((2, 2), dtype=config.floatX) * 3
    zv = np.ones((2, 2), dtype=config.floatX) * 5

    compare_jax_and_py(out_fg, [xv, yv, zv])
