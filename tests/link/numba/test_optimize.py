import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.inner_function import HasInnerFunction
from pytensor.tensor.optimize import minimize


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


def _build_minimize():
    x = pt.vector("x", shape=(10,))
    a = pt.vector("a", shape=(10,))
    objective = pt.sum(a[1:] * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    x_star, _ = minimize(objective, x, method="trust-exact", jac=True, hess=True)
    return x, a, x_star


def test_minimize_inner_fn_linked_with_numba(linkers_spy):
    """Regression test for #2668: inner function must follow the outer numba mode."""
    x, a, x_star = _build_minimize()
    x0, a_val = np.zeros(10), np.full(10, 100.0)

    with pytest.warns(UserWarning, match="object mode"):
        fn = pytensor.function([x, a], x_star, mode="NUMBA")
        res_numba = fn(x0, a_val)

    assert linkers_spy
    assert set(linkers_spy) == {"NumbaLinker"}

    res_default = pytensor.function([x, a], x_star, mode="FAST_COMPILE")(x0, a_val)
    np.testing.assert_allclose(res_numba, res_default, rtol=1e-5)


def test_numba_inner_linker_does_not_leak_to_other_modes(linkers_spy):
    """Compiling with NUMBA must not change how a later default-mode compile links."""
    x, a, x_star = _build_minimize()
    x0, a_val = np.zeros(10), np.full(10, 100.0)

    with pytest.warns(UserWarning, match="object mode"):
        pytensor.function([x, a], x_star, mode="NUMBA")(x0, a_val)

    linkers_spy.clear()
    res = pytensor.function([x, a], x_star, mode="FAST_COMPILE")(x0, a_val)

    assert "NumbaLinker" not in linkers_spy
    assert np.all(np.isfinite(res))
