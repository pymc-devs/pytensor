import subprocess
import sys

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode, predefined_linkers
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.ifelse import ifelse
from pytensor.link.python.dispatch.basic import python_funcify
from pytensor.link.python.linker import PythonLinker
from pytensor.link.vm import VMLinker
from pytensor.raise_op import Assert
from pytensor.scalar.basic import Composite
from pytensor.tensor.blas import Gemv
from pytensor.tensor.blas_c import CGemv
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.type import matrix, vector


def python_function(inputs, outputs, **kwargs):
    return pytensor.function(inputs, outputs, mode="PYTHON", **kwargs)


def compare_py_and_pyjit(graph_inputs, graph_outputs, test_inputs, assert_fn=None):
    """Compare the per-node ``py`` (VM) backend against the whole-graph ``pyjit``.

    Both run the same ``python_funcify`` dispatches, so this checks that the VM
    per-node wiring and the JIT whole-graph composition agree. Only valid for
    graphs without lazy ops, which the JIT cannot compose.
    """
    if assert_fn is None:

        def assert_fn(a, b):
            np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-10)

    py_fn = pytensor.function(graph_inputs, graph_outputs, mode="PYTHON")
    pyjit_fn = pytensor.function(graph_inputs, graph_outputs, mode="PYJIT")

    py_res = py_fn(*test_inputs)
    pyjit_res = pyjit_fn(*test_inputs)
    for py_out, pyjit_out in zip(
        py_res if isinstance(py_res, list) else [py_res],
        pyjit_res if isinstance(pyjit_res, list) else [pyjit_res],
    ):
        assert_fn(py_out, pyjit_out)
    return py_fn, py_res


def test_mode_and_linker_registered():
    # "py" is the robust per-node VM backend; "pyjit" the whole-graph JIT.
    assert isinstance(predefined_linkers["py"], VMLinker)
    assert isinstance(predefined_linkers["pyjit"], PythonLinker)
    assert isinstance(get_mode("PYTHON").linker, VMLinker)
    assert isinstance(get_mode("PYJIT").linker, PythonLinker)


@pytest.mark.parametrize(
    "build, values, expected",
    [
        pytest.param(
            lambda x, y: pt.exp(x) + y * 2.0,
            (np.arange(4.0), np.arange(4.0) + 1),
            lambda xv, yv: np.exp(xv) + yv * 2.0,
            id="elemwise",
        ),
        pytest.param(
            lambda x, y: x.sum() + y.mean(),
            (np.arange(4.0), np.arange(4.0) + 1),
            lambda xv, yv: xv.sum() + yv.mean(),
            id="reduction",
        ),
        pytest.param(
            lambda x, y: x[1:3] - y[::-1][1:3],
            (np.arange(4.0), np.arange(4.0) + 1),
            lambda xv, yv: xv[1:3] - yv[::-1][1:3],
            id="subtensor",
        ),
    ],
)
def test_vector_graphs(build, values, expected):
    x = vector("x")
    y = vector("y")
    fn = python_function([x, y], build(x, y))
    np.testing.assert_allclose(fn(*values), expected(*values))


def test_matmul():
    A = matrix("A")
    B = matrix("B")
    Av = np.arange(6.0).reshape(2, 3)
    Bv = np.arange(6.0).reshape(3, 2)
    fn = python_function([A, B], A @ B)
    np.testing.assert_allclose(fn(Av, Bv), Av @ Bv)


def test_multiple_outputs():
    x = vector("x")
    y = vector("y")
    fn = python_function([x, y], [x + y, x - y])
    xv, yv = np.arange(4.0), np.arange(4.0) + 1
    out_add, out_sub = fn(xv, yv)
    np.testing.assert_allclose(out_add, xv + yv)
    np.testing.assert_allclose(out_sub, xv - yv)


def test_constant_in_graph():
    x = vector("x")
    fn = python_function([x], x + pt.constant(np.ones(4)))
    xv = np.arange(4.0)
    np.testing.assert_allclose(fn(xv), xv + 1.0)


def test_constant_only_output():
    # An output with no owner (a bare constant) must still be returned.
    fn = python_function([], pt.constant(5.0))
    np.testing.assert_allclose(fn(), 5.0)


def test_shared_input():
    x = vector("x")
    s = pytensor.shared(2.0)
    fn = python_function([x], x * s)
    xv = np.arange(4.0)
    np.testing.assert_allclose(fn(xv), xv * 2.0)
    s.set_value(3.0)
    np.testing.assert_allclose(fn(xv), xv * 3.0)


def test_no_outputs():
    x = vector("x")
    fn = python_function([x], [], on_unused_input="ignore")
    assert fn(np.arange(4.0)) == []


def test_fusion_excluded():
    x = vector("x")
    y = vector("y")
    fn = python_function([x, y], pt.exp(x) * y + pt.log(x) - y**2)
    elemwise_nodes = [
        node for node in fn.maker.fgraph.apply_nodes if isinstance(node.op, Elemwise)
    ]
    # Without fusion every scalar op stays its own vectorized Elemwise node,
    # rather than collapsing into a single Composite.
    assert len(elemwise_nodes) > 1
    assert not any(isinstance(node.op.scalar_op, Composite) for node in elemwise_nodes)


def test_cxx_only_excluded():
    # The `use_c_blas` rewrite (tagged cxx_only) would turn Gemv into the
    # C-only CGemv, which has no perform and would strand a pure-Python linker.
    # Excluding cxx_only keeps the perform-backed Gemv.
    A = matrix("A")
    x = vector("x")
    y = vector("y")
    fn = python_function([A, x, y], 2.0 * y + 3.0 * (A @ x))
    ops = {type(node.op) for node in fn.maker.fgraph.apply_nodes}
    assert Gemv in ops
    assert CGemv not in ops
    Av, xv, yv = np.arange(6.0).reshape(2, 3), np.arange(3.0), np.arange(2.0)
    np.testing.assert_allclose(fn(Av, xv, yv), 2.0 * yv + 3.0 * (Av @ xv))


def test_ifelse_lazy():
    # IfElse has no perform (only a lazy make_thunk). The py (VM) backend runs it
    # via the fallback AND short-circuits it: the unused branch (which raises if
    # evaluated) must not run. The whole-graph pyjit backend cannot do this.
    c = pt.scalar("c")
    x = vector("x")
    boom = Assert("unused branch must not run")(x, pt.eq(x.sum(), -999.0))
    fn = python_function([c, x], ifelse(c > 0, x * 2.0, boom))
    np.testing.assert_allclose(fn(1.0, np.ones(3)), np.full(3, 2.0))


class _PerformOnlyOp(Op):
    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = np.square(inputs[0])


def test_default_dispatch_uses_perform():
    x = vector("x")
    fn = python_function([x], _PerformOnlyOp()(x))
    xv = np.arange(4.0)
    np.testing.assert_allclose(fn(xv), xv**2)


class _DispatchedOp(Op):
    __props__ = ()

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = inputs[0]


@python_funcify.register(_DispatchedOp)
def _python_funcify_dispatched_op(op, node=None, **kwargs):
    # A fast path distinguishable from the identity perform above.
    def impl(x):
        return x + 100.0

    return impl


def test_registered_dispatch_overrides_perform():
    x = vector("x")
    fn = python_function([x], _DispatchedOp()(x))
    xv = np.arange(4.0)
    np.testing.assert_allclose(fn(xv), xv + 100.0)


def test_dispatch_loaded_lazily():
    # Importing pytensor must not pull in the dispatch package; it should only
    # load on the first PYTHON compile.
    script = (
        "import sys, pytensor, pytensor.tensor as pt;"
        "mod='pytensor.link.python.dispatch.basic';"
        "assert mod not in sys.modules, 'loaded too early';"
        "x=pt.vector('x');"
        "pytensor.function([x], x+1, mode='PYTHON');"
        "assert mod in sys.modules, 'not loaded after compile'"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
