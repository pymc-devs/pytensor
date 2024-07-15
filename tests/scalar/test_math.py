import itertools

import numpy as np
import pytest
import scipy.special as sp

import pytensor.tensor as pt
from pytensor import function
from pytensor.compile.mode import Mode
from pytensor.graph import ancestors
from pytensor.graph.fg import FunctionGraph
from pytensor.link.c.basic import CLinker
from pytensor.scalar import ScalarLoop, float32, float64, int32
from pytensor.scalar.math import (
    betainc,
    betainc_grad,
    gammainc,
    gammaincc,
    gammal,
    gammau,
    hyp2f1,
)
from tests.link.test_link import make_function


def test_gammainc_python():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = gammainc(x1, x2)
    test_func = function([x1, x2], y, mode=Mode("py"))
    assert np.isclose(test_func(1, 2), sp.gammainc(1, 2))


def test_gammainc_nan_c():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = gammainc(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammainc_inf_c():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = gammainc(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isclose(test_func(np.inf, 1), sp.gammainc(np.inf, 1))
    assert np.isclose(test_func(1, np.inf), sp.gammainc(1, np.inf))
    assert np.isnan(test_func(np.inf, np.inf))


def test_gammaincc_python():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = gammaincc(x1, x2)
    test_func = function([x1, x2], y, mode=Mode("py"))
    assert np.isclose(test_func(1, 2), sp.gammaincc(1, 2))


def test_gammaincc_nan_c():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = gammaincc(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammaincc_inf_c():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = gammaincc(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isclose(test_func(np.inf, 1), sp.gammaincc(np.inf, 1))
    assert np.isclose(test_func(1, np.inf), sp.gammaincc(1, np.inf))
    assert np.isnan(test_func(np.inf, np.inf))


def test_gammal_nan_c():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = gammal(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammau_nan_c():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = gammau(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


@pytest.mark.parametrize("linker", ["py", "c"])
def test_betainc(linker):
    a, b, x = pt.scalars("a", "b", "x")
    res = betainc(a, b, x)
    test_func = function([a, b, x], res, mode=Mode(linker=linker, optimizer="fast_run"))
    assert np.isclose(test_func(15, 10, 0.7), sp.betainc(15, 10, 0.7))

    # Regression test for https://github.com/pymc-devs/pytensor/issues/906
    if res.dtype == "float64":
        assert test_func(100, 1.0, 0.1) > 0


def test_betainc_derivative_nan():
    a, b, x = pt.scalars("a", "b", "x")
    res = betainc_grad(a, b, x, True)
    test_func = function([a, b, x], res, mode=Mode("py"))
    assert not np.isnan(test_func(1, 1, 1))
    assert np.isnan(test_func(1, 1, -1))
    assert np.isnan(test_func(1, 1, 2))
    assert np.isnan(test_func(1, -1, 1))
    assert np.isnan(test_func(1, 1, -1))


@pytest.mark.parametrize(
    "op, scalar_loop_grads",
    [
        (gammainc, [0]),
        (gammaincc, [0]),
        (betainc, [0, 1]),
        (hyp2f1, [0, 1, 2]),
    ],
)
def test_scalarloop_grad_mixed_dtypes(op, scalar_loop_grads):
    nin = op.nin
    for types in itertools.product((float32, float64, int32), repeat=nin):
        inputs = [type() for type in types]
        out = op(*inputs)
        wrt = [
            inp
            for idx, inp in enumerate(inputs)
            if idx in scalar_loop_grads and inp.type.dtype.startswith("float")
        ]
        if not wrt:
            continue
        # The ScalarLoop in the graph will fail if the input types are different from the updates
        grad = pt.grad(out, wrt=wrt)
        assert any(
            (var.owner and isinstance(var.owner.op, ScalarLoop))
            for var in ancestors(grad)
        )
