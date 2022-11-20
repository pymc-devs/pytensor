import pytest

import pytensor
from pytensor.graph.fg import FunctionGraph
from pytensor.link.c.basic import CLinker
from pytensor.scalar.basic import floats
from pytensor.scalar.basic_sympy import SymPyCCode
from tests.link.test_link import make_function


sympy = pytest.importorskip("sympy")


xs = sympy.Symbol("x")
ys = sympy.Symbol("y")

xt, yt = floats("xy")


@pytest.mark.skipif(not pytensor.config.cxx, reason="Need cxx for this test")
def test_SymPyCCode():
    op = SymPyCCode([xs, ys], xs + ys)
    e = op(xt, yt)
    g = FunctionGraph([xt, yt], [e])
    fn = make_function(CLinker().accept(g))
    assert fn(1.0, 2.0) == 3.0


def test_grad():
    op = SymPyCCode([xs], xs**2)
    zt = op(xt)
    ztprime = pytensor.grad(zt, xt)
    assert ztprime.owner.op.expr == 2 * xs


def test_multivar_grad():
    op = SymPyCCode([xs, ys], xs**2 + ys**3)
    zt = op(xt, yt)
    dzdx, dzdy = pytensor.grad(zt, [xt, yt])
    assert dzdx.owner.op.expr == 2 * xs
    assert dzdy.owner.op.expr == 3 * ys**2
