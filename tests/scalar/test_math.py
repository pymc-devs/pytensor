import itertools

import numpy as np
import pytest
import scipy
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
    hyp2f1,
    psi,
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

    for a, x in [
        (0, 1),
        (np.nan, 0),
        (np.nan, 1),
        (1, np.nan),
        (np.nan, np.inf),
        (np.inf, np.nan),
    ]:
        assert np.isnan(test_func(a, x))


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

    for a, x in [
        (0, 1),
        (np.nan, 0),
        (np.nan, 1),
        (1, np.nan),
        (np.nan, np.inf),
        (np.inf, np.nan),
    ]:
        assert np.isnan(test_func(a, x))


def test_gammaincc_inf_c():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = gammaincc(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isclose(test_func(np.inf, 1), sp.gammaincc(np.inf, 1))
    assert np.isclose(test_func(1, np.inf), sp.gammaincc(1, np.inf))
    assert np.isnan(test_func(np.inf, np.inf))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gammainc_large_c(dtype):
    a, x = (pt.vector(name, dtype=dtype) for name in ("a", "x"))
    f = function(
        [a, x],
        [pt.gammainc(a, x), pt.gammaincc(a, x)],
        mode=Mode(linker="c", optimizer=None),
    )
    a_vals = np.array([1000, 10001, 100001, 300001, 1000001, 1e8, 1e12])
    offsets = np.array([-4, -1, 0, 1, 4])
    x_vals = a_vals[:, None] + np.sqrt(a_vals[:, None]) * offsets
    # Include the reported Poisson CDF case and adjacent floating-point values.
    a_vals = np.broadcast_to(a_vals[:, None], x_vals.shape).astype(dtype).ravel()
    x_vals = x_vals.astype(dtype).ravel()
    x_vals = np.concatenate(
        [x_vals, a_vals - 1, np.nextafter(a_vals, 0), np.nextafter(a_vals, np.inf)]
    )
    a_vals = np.tile(a_vals, 4)
    p, q = f(a_vals, x_vals)
    rtol = 2e-6 if dtype == "float32" else 5e-14
    np.testing.assert_allclose(p, sp.gammainc(a_vals, x_vals), rtol=rtol, atol=0)
    np.testing.assert_allclose(q, sp.gammaincc(a_vals, x_vals), rtol=rtol, atol=0)
    np.testing.assert_allclose(p + q, 1, rtol=0, atol=np.finfo(dtype).eps)


def test_gammainc_asymptotic_boundaries_c():
    a, x = pt.dvectors("a", "x")
    f = function(
        [a, x],
        [pt.gammainc(a, x), pt.gammaincc(a, x)],
        mode=Mode(linker="c", optimizer=None),
    )
    a_vals = np.array([np.nextafter(1000.0, 0), 1000, np.nextafter(1000.0, np.inf)])
    ratios = np.array([0.7 - 1e-12, 0.7, 0.7 + 1e-12, 1, 1.3 - 1e-12, 1.3, 1.3 + 1e-12])
    x_vals = a_vals[:, None] * ratios
    a_vals = np.broadcast_to(a_vals[:, None], x_vals.shape).ravel()
    x_vals = x_vals.ravel()
    p, q = f(a_vals, x_vals)
    np.testing.assert_allclose(p, sp.gammainc(a_vals, x_vals), rtol=3e-12, atol=0)
    np.testing.assert_allclose(q, sp.gammaincc(a_vals, x_vals), rtol=3e-12, atol=0)


def test_gammainc_large_tails_c():
    a, x = pt.dscalars("a", "x")
    f = function(
        [a, x],
        [pt.gammainc(a, x), pt.gammaincc(a, x)],
        mode=Mode(linker="c", optimizer=None),
    )
    # Independent 70-digit mpmath quadrature of the gamma density, rescaled
    # by sqrt(a). SciPy's series can also lose accuracy in large-a lower tails.
    for a_val, x_val, tail, expected in [
        (1000, 700, 0, 1.0158583345333216374579116938985359e-26),
        (1000, 1300, 1, 1.8736155715785551242055681977145791e-18),
        (1e6, 995000, 0, 2.7495803592700707538279083912739602e-7),
        (1e6, 1005000, 1, 2.9874901401146348544408764692820645e-7),
        (1e8, 99950000, 0, 2.8546421399586261429767424887672704e-7),
        (1e12, 999995000000, 0, 2.8663967832502037179692080057750576e-7),
    ]:
        np.testing.assert_allclose(f(a_val, x_val)[tail], expected, rtol=5e-14, atol=0)
    for a_val in [1e100, 1e308]:
        np.testing.assert_array_equal(f(a_val, a_val), [0.5, 0.5])
        np.testing.assert_array_equal(f(a_val, 0.9 * a_val), [0, 1])
        np.testing.assert_array_equal(f(a_val, 1.1 * a_val), [1, 0])


def test_gammal_nan_c():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = pt.gammal(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


def test_gammau_nan_c():
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = pt.gammau(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    assert np.isnan(test_func(-1, 1))
    assert np.isnan(test_func(1, -1))
    assert np.isnan(test_func(-1, -1))


@pytest.mark.parametrize(
    "pt_func, sp_reg_func, points",
    [
        (pt.gammau, sp.gammaincc, [(2, 1), (10, 1), (20, 5)]),
        (pt.gammal, sp.gammainc, [(2, 16), (10, 80), (50, 200)]),
    ],
)
def test_unregularized_gamma_c_domain(pt_func, sp_reg_func, points):
    # Regression test for the removed upperGamma/lowerGamma C kernels, which used a
    # single approximation method for the whole domain and returned garbage for
    # gammau with x < k + 1 (NaN at k=2, x=1) and inaccurate gammal for x >> k
    x1 = pt.dscalar()
    x2 = pt.dscalar()
    y = pt_func(x1, x2)
    test_func = make_function(CLinker().accept(FunctionGraph([x1, x2], [y])))
    for k, x in points:
        expected = sp_reg_func(k, x) * sp.gamma(k)
        np.testing.assert_allclose(test_func(k, x), expected, rtol=1e-12)


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


@pytest.mark.parametrize(
    "linker",
    ["py", "cvm"],
)
def test_psi(linker):
    x = float64("x")
    out = psi(x)

    fn = function([x], out, mode=Mode(linker=linker, optimizer="fast_run"))
    fn.dprint()

    x_test = np.float64(0.7)

    np.testing.assert_allclose(fn(x_test), scipy.special.psi(x_test))
    np.testing.assert_allclose(fn(-x_test), scipy.special.psi(-x_test))
