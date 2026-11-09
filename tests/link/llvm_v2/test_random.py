"""Random variables: draws from the function state's random state."""

import ctypes

import numpy as np
import pytest
from scipy import stats

import pytensor
import pytensor.tensor as pt
import pytensor.tensor.random.basic as rb
from pytensor.compile.mode import get_mode


pytest.importorskip("pytensor_llvm")

MODE = get_mode("LLVM_V2").clone(link_kwargs={"numba_fallback": False})
N = 200_000


def draw(rv_fn, *params, size=(N,), seed=1):
    rng = pytensor.shared(np.random.default_rng(0))
    x = rv_fn(*params, size=size, rng=rng)
    f = pytensor.function([], x, mode=MODE)
    f.maker.linker.compiled.seed(seed)
    return f()


@pytest.mark.parametrize(
    "rv, params, dist",
    [
        (rb.normal, (1.5, 2.0), stats.norm(1.5, 2.0)),
        (rb.halfnormal, (0.5, 2.0), stats.halfnorm(0.5, 2.0)),
        (rb.lognormal, (0.2, 0.5), stats.lognorm(0.5, scale=np.exp(0.2))),
        (rb.uniform, (-1.0, 3.0), stats.uniform(-1.0, 4.0)),
        (rb.exponential, (2.0,), stats.expon(scale=2.0)),
        (rb.gamma, (3.0, 2.0), stats.gamma(3.0, scale=0.5)),  # (shape, rate)
        (rb.invgamma, (4.0, 3.0), stats.invgamma(4.0, scale=3.0)),
        (rb.beta, (2.0, 5.0), stats.beta(2.0, 5.0)),
        (rb.bernoulli, (0.3,), stats.bernoulli(0.3)),
        (rb.poisson, (4.5,), stats.poisson(4.5)),
        (rb.binomial, (10, 0.35), stats.binom(10, 0.35)),
        (rb.integers, (2, 7), stats.randint(2, 7)),
    ],
)
def test_moments(rv, params, dist):
    x = draw(rv, *params)
    assert x.shape == (N,)
    mean, var = dist.mean(), dist.var()
    np.testing.assert_allclose(x.mean(), mean, atol=5 * np.sqrt(var / N))
    np.testing.assert_allclose(x.var(), var, rtol=0.05)
    if rv in (rb.bernoulli, rb.poisson, rb.binomial, rb.integers):
        assert x.dtype == np.int64
        lo, hi = dist.support()
        assert x.min() >= lo and x.max() <= hi


def test_seeded_draws_are_reproducible():
    rng = pytensor.shared(np.random.default_rng(0))
    x = rb.normal(0, 1, size=(5,), rng=rng)
    f = pytensor.function([], x, mode=MODE)
    p = f.maker.linker.compiled
    p.seed(42)
    a, b = f(), f()
    assert not np.array_equal(a, b)  # the state advances
    p.seed(42)
    np.testing.assert_array_equal(f(), a)
    np.testing.assert_array_equal(f(), b)
    p.seed(43)
    assert not np.array_equal(f(), a)


def test_broadcast_parameters_and_size():
    mu = pt.dvector("mu")
    sigma = pt.dmatrix("sigma")
    rng = pytensor.shared(np.random.default_rng(0))
    x = rb.normal(mu, sigma, rng=rng)  # shape from the parameters
    y = rb.normal(mu, 1.0, size=(4000, 3), rng=rng)
    f = pytensor.function([mu, sigma], [x, y], mode=MODE)
    mv = np.array([0.0, 10.0, -10.0])
    rx, ry = f(mv, np.full((4000, 3), 1e-12))
    np.testing.assert_allclose(rx, np.broadcast_to(mv, (4000, 3)), atol=1e-9)
    np.testing.assert_allclose(ry.mean(axis=0), mv, atol=0.1)


def test_parameter_checks():
    sigma = pt.dscalar("sigma")
    rng = pytensor.shared(np.random.default_rng(0))
    f = pytensor.function([sigma], rb.normal(0, sigma, size=(3,), rng=rng), mode=MODE)
    f(1.0)
    with pytest.raises(Exception, match="scale < 0"):
        f(-1.0)
    p = pt.dscalar("p")
    g = pytensor.function([p], rb.bernoulli(p, size=(3,), rng=rng), mode=MODE)
    with pytest.raises(Exception, match="p < 0"):
        g(1.5)


def test_shared_generator_updates_pass_through():
    rng = pytensor.shared(np.random.default_rng(0))
    next_rng, x = rb.normal(0, 1, size=(3,), rng=rng).owner.outputs
    f = pytensor.function([], x, updates={rng: next_rng}, mode=MODE)
    gen = rng.get_value(borrow=True)
    a, b = f(), f()
    assert not np.array_equal(a, b)
    # the generator isn't used: draws come from the function's own state
    assert rng.get_value(borrow=True) is gen


def test_generator_as_explicit_input():
    rng = pt.random.type.random_generator_type("rng")
    next_rng, x = rb.uniform(0, 1, size=(4,), rng=rng).owner.outputs
    f = pytensor.function([rng], [next_rng, x], mode=MODE)
    gen = np.random.default_rng(3)
    out_rng, draws = f(gen)
    assert out_rng is gen
    assert draws.shape == (4,) and np.all((draws >= 0) & (draws < 1))


def test_nested_program_draws_from_the_callers_state():
    from pytensor.compile.builders import OpFromGraph, UnitKind

    a = pt.dscalar("a")
    r = pt.random.type.random_generator_type("r")
    inner = OpFromGraph([a, r], [a + rb.normal(0, 1, size=(4,), rng=r)], unit=UnitKind.INTERPRETED)
    x = pt.dscalar("x")
    rng = pytensor.shared(np.random.default_rng(0))
    f = pytensor.function([x], inner(x, rng), mode=MODE)
    assert "program:" in f.maker.linker.compiled.dump()
    p = f.maker.linker.compiled
    p.seed(7)
    r1 = f(0.0)
    p.seed(7)
    np.testing.assert_array_equal(f(0.0), r1)


def test_c_api_seed():
    from tests.link.llvm.test_c_api import Api, PtArray, as_ptarrays

    rng = pytensor.shared(np.random.default_rng(0))
    f = pytensor.function([], rb.normal(0, 1, size=(6,), rng=rng), mode=MODE)
    p = f.maker.linker.compiled
    fn = p.c_function()
    a = Api.from_address(p.c_api_table())
    new = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(a.session_new)
    free = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.session_free)
    seed = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64)(a.seed)
    call = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(PtArray), ctypes.POINTER(PtArray))(a.call)

    def run(st):
        out = np.empty(6)
        ins, k1 = as_ptarrays([])
        outs, k2 = as_ptarrays([out])
        assert call(st, ins, outs) == 0
        return out

    s1, s2 = new(fn), new(fn)
    seed(s1, 5)
    seed(s2, 5)
    np.testing.assert_array_equal(run(s1), run(s2))  # one state per caller (chain)
    seed(s2, 6)
    assert not np.array_equal(run(s1), run(s2))
    free(s1)
    free(s2)
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(a.function_decref)(fn)
