import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.inner_function import HasInnerFunction
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.tensor.optimize import minimize, root


pytest.importorskip("numba")


@pytest.fixture
def linkers_spy(monkeypatch):
    """Record the linker of every inner function built via `HasInnerFunction.link_fgraph`."""
    linkers = []
    orig = HasInnerFunction.link_fgraph

    def spy(self, fgraph, mode):
        linkers.append(type(mode.linker).__name__)
        return orig(self, fgraph, mode)

    monkeypatch.setattr(HasInnerFunction, "link_fgraph", spy)
    return linkers


class ToyInnerFunctionOp(HasInnerFunction, Op):
    """Stand-in for an out-of-tree `HasInnerFunction` op with no numba dispatch."""

    def __init__(self, inputs, outputs):
        self.fgraph = FunctionGraph(inputs, outputs, clone=True).freeze()

    def make_node(self, x):
        x = pt.as_tensor(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = self.fn(*inputs)[0]


def _minimize_graph():
    x = pt.vector("x", shape=(10,))
    a = pt.vector("a", shape=(10,))
    objective = pt.sum(a[1:] * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    x_star, _ = minimize(objective, x, method="trust-exact", jac=True, hess=True)
    return [x, a], x_star, (np.zeros(10), np.full(10, 100.0))


def test_minimize_inner_fn_linked_with_numba(linkers_spy):
    inputs, out, vals = _minimize_graph()
    with pytest.warns(UserWarning, match="object mode"):
        res = pytensor.function(inputs, out, mode="NUMBA")(*vals)
    assert linkers_spy == ["NumbaLinker"]
    ref = pytensor.function(inputs, out, mode="FAST_COMPILE")(*vals)
    np.testing.assert_allclose(res, ref, rtol=1e-5)


def test_root_inner_fn_linked_with_numba(linkers_spy):
    y = pt.vector("y", shape=(3,))
    sol, _ = root(y**3 + y - 1.0, y)
    with pytest.warns(UserWarning, match="object mode"):
        res = pytensor.function([y], sol, mode="NUMBA")(np.zeros(3))
    assert linkers_spy == ["NumbaLinker"]
    np.testing.assert_allclose(res**3 + res - 1.0, 0, atol=1e-6)


@pytest.mark.parametrize("later_mode", ["FAST_COMPILE", "CVM"])
def test_numba_inner_fn_does_not_leak(linkers_spy, later_mode):
    inputs, out, vals = _minimize_graph()
    with pytest.warns(UserWarning, match="object mode"):
        pytensor.function(inputs, out, mode="NUMBA")(*vals)
    linkers_spy.clear()
    pytensor.function(inputs, out, mode=later_mode)(*vals)
    assert linkers_spy
    assert "NumbaLinker" not in linkers_spy


def test_generic_has_inner_function_op(linkers_spy):
    xi = pt.vector("xi")
    op = ToyInnerFunctionOp([xi], [pt.exp(xi) + 1])
    x = pt.vector("x")
    with pytest.warns(UserWarning, match="object mode"):
        fn = pytensor.function([x], op(x), mode="NUMBA")
    np.testing.assert_allclose(fn(np.arange(3.0)), np.exp(np.arange(3.0)) + 1)
    assert linkers_spy == ["NumbaLinker"]


def test_inner_fn_not_compiled_at_dispatch(linkers_spy):
    inputs, out, vals = _minimize_graph()
    with pytest.warns(UserWarning, match="object mode"):
        fn = pytensor.function(inputs, out, mode="NUMBA")
    assert linkers_spy == []
    fn(*vals)
    assert linkers_spy == ["NumbaLinker"]
