import pytest


pytest.importorskip("xarray")

import numpy as np

from pytensor.compile.function import function
from pytensor.tensor import tensor
from pytensor.tensor.random.type import random_generator_type
from pytensor.tensor.random.variable import RandomGeneratorVariable
from pytensor.xtensor.random import (
    XRandomGeneratorSharedVariable,
    XRandomGeneratorVariable,
    shared_xrng,
)
from pytensor.xtensor.random import (
    rng as xrng_fn,
)
from pytensor.xtensor.random.type import rng_to_xrng, xrng_to_rng
from pytensor.xtensor.type import XTensorVariable, as_xtensor


@pytest.fixture
def xrng():
    return xrng_fn("test_xrng")


def assert_xrng_and_draw(result):
    """Assert result is a (XRandomGeneratorVariable, XTensorVariable) tuple."""
    assert isinstance(result, (tuple, list)) and len(result) == 2
    next_rng, draw = result
    assert isinstance(next_rng, XRandomGeneratorVariable)
    assert isinstance(draw, XTensorVariable)
    return next_rng, draw


class TestXRandomGeneratorVariableMethods:
    """Test that all ptx.random.rng() methods return (XRandomGeneratorVariable, XTensorVariable) tuples."""

    def test_normal(self, xrng):
        assert_xrng_and_draw(xrng.normal(0, 1, extra_dims={"a": 3}))

    def test_uniform(self, xrng):
        assert_xrng_and_draw(xrng.uniform(0, 1, extra_dims={"a": 3}))

    def test_triangular(self, xrng):
        assert_xrng_and_draw(xrng.triangular(0, 0.5, 1, extra_dims={"a": 3}))

    def test_beta(self, xrng):
        assert_xrng_and_draw(xrng.beta(1.0, 2.0, extra_dims={"a": 3}))

    def test_halfnormal(self, xrng):
        assert_xrng_and_draw(xrng.halfnormal(0, 1, extra_dims={"a": 3}))

    def test_lognormal(self, xrng):
        assert_xrng_and_draw(xrng.lognormal(0, 1, extra_dims={"a": 3}))

    def test_gamma(self, xrng):
        assert_xrng_and_draw(xrng.gamma(1.0, 1.0, extra_dims={"a": 3}))

    def test_exponential(self, xrng):
        assert_xrng_and_draw(xrng.exponential(1.0, extra_dims={"a": 3}))

    def test_weibull(self, xrng):
        assert_xrng_and_draw(xrng.weibull(1.5, extra_dims={"a": 3}))

    def test_logistic(self, xrng):
        assert_xrng_and_draw(xrng.logistic(0, 1, extra_dims={"a": 3}))

    def test_vonmises(self, xrng):
        assert_xrng_and_draw(xrng.vonmises(0, 1, extra_dims={"a": 3}))

    def test_pareto(self, xrng):
        assert_xrng_and_draw(xrng.pareto(2.0, 1.0, extra_dims={"a": 3}))

    def test_gumbel(self, xrng):
        assert_xrng_and_draw(xrng.gumbel(0, 1, extra_dims={"a": 3}))

    def test_cauchy(self, xrng):
        assert_xrng_and_draw(xrng.cauchy(0, 1, extra_dims={"a": 3}))

    def test_halfcauchy(self, xrng):
        assert_xrng_and_draw(xrng.halfcauchy(0, 1, extra_dims={"a": 3}))

    def test_invgamma(self, xrng):
        assert_xrng_and_draw(xrng.invgamma(2.0, 1.0, extra_dims={"a": 3}))

    def test_wald(self, xrng):
        assert_xrng_and_draw(xrng.wald(1.0, 1.0, extra_dims={"a": 3}))

    def test_truncexpon(self, xrng):
        assert_xrng_and_draw(xrng.truncexpon(2.0, 0, 1, extra_dims={"a": 3}))

    def test_laplace(self, xrng):
        assert_xrng_and_draw(xrng.laplace(0, 1, extra_dims={"a": 3}))

    def test_t(self, xrng):
        assert_xrng_and_draw(xrng.t(3.0, 0, 1, extra_dims={"a": 3}))

    def test_gengamma(self, xrng):
        assert_xrng_and_draw(xrng.gengamma(1.0, 1.0, 1.0, extra_dims={"a": 3}))

    def test_multivariate_normal(self, xrng):
        mu = as_xtensor(tensor("mu", shape=(2,)), dims=("rows",))
        cov = as_xtensor(tensor("cov", shape=(2, 2)), dims=("rows", "cols"))
        assert_xrng_and_draw(
            xrng.multivariate_normal(mu, cov, core_dims=("rows", "cols"))
        )

    def test_dirichlet(self, xrng):
        alphas = as_xtensor(tensor("alphas", shape=(3,)), dims=("k",))
        assert_xrng_and_draw(xrng.dirichlet(alphas, core_dims=("k",)))

    def test_poisson(self, xrng):
        assert_xrng_and_draw(xrng.poisson(5.0, extra_dims={"a": 3}))

    def test_geometric(self, xrng):
        assert_xrng_and_draw(xrng.geometric(0.5, extra_dims={"a": 3}))

    def test_hypergeometric(self, xrng):
        assert_xrng_and_draw(xrng.hypergeometric(10, 5, 3, extra_dims={"a": 3}))

    def test_bernoulli(self, xrng):
        assert_xrng_and_draw(xrng.bernoulli(0.5, extra_dims={"a": 3}))

    def test_binomial(self, xrng):
        assert_xrng_and_draw(xrng.binomial(10, 0.5, extra_dims={"a": 3}))

    def test_negative_binomial(self, xrng):
        assert_xrng_and_draw(xrng.negative_binomial(10, 0.5, extra_dims={"a": 3}))

    def test_betabinom(self, xrng):
        assert_xrng_and_draw(xrng.betabinom(10, 2.0, 3.0, extra_dims={"a": 3}))

    def test_multinomial(self, xrng):
        p = as_xtensor(tensor("p", shape=(3,)), dims=("k",))
        assert_xrng_and_draw(xrng.multinomial(10, p, core_dims=("k",)))

    def test_categorical(self, xrng):
        p = as_xtensor(tensor("p", shape=(3,)), dims=("k",))
        assert_xrng_and_draw(xrng.categorical(p, core_dims=("k",)))

    def test_integers(self, xrng):
        assert_xrng_and_draw(xrng.integers(0, 10, extra_dims={"a": 3}))

    def test_standard_normal(self, xrng):
        assert_xrng_and_draw(xrng.standard_normal(extra_dims={"a": 3}))

    def test_chisquare(self, xrng):
        assert_xrng_and_draw(xrng.chisquare(3.0, extra_dims={"a": 3}))

    def test_rayleigh(self, xrng):
        assert_xrng_and_draw(xrng.rayleigh(1.0, extra_dims={"a": 3}))


class TestXRandomGeneratorVariableChaining:
    """Test chaining xtensor RNG methods."""

    def test_chain_with_named_dims(self, xrng):
        mu = as_xtensor(tensor("mu", shape=(3,)), dims=("city",))
        next_rng, x = xrng.normal(mu, 1.0)
        next_rng, y = next_rng.uniform(0, 1, extra_dims={"sample": 5})

        assert x.type.dims == ("city",)
        assert y.type.dims == ("sample",)
        assert isinstance(next_rng, XRandomGeneratorVariable)

    def test_chain_compiles_and_runs(self):
        xrng = xrng_fn("rng")
        next_rng, x = xrng.normal(0, 1, extra_dims={"a": 3})
        next_rng, y = next_rng.uniform(0, 1, extra_dims={"b": 5})

        fn = function([xrng], [x, y])
        result = fn(np.random.default_rng(42))
        assert result[0].shape == (3,)
        assert result[1].shape == (5,)

    def test_is_variable(self, xrng):
        """Test that xrng is a proper PyTensor Variable."""
        from pytensor.graph.basic import Variable

        assert isinstance(xrng, Variable)


class TestRNGTypeCasting:
    """Test casting between RandomGeneratorType and XRandomGeneratorType."""

    def test_rng_to_xrng(self):
        rng_var = random_generator_type("rng")
        assert isinstance(rng_var, RandomGeneratorVariable)
        xrng_var = rng_to_xrng(rng_var)
        assert isinstance(xrng_var, XRandomGeneratorVariable)

    def test_xrng_to_rng(self):
        xrng_var = xrng_fn("xrng")
        assert isinstance(xrng_var, XRandomGeneratorVariable)
        rng_var = xrng_to_rng(xrng_var)
        assert isinstance(rng_var, RandomGeneratorVariable)

    def test_roundtrip(self):
        rng_var = random_generator_type("rng")
        xrng_var = rng_to_xrng(rng_var)
        rng_back = xrng_to_rng(xrng_var)
        assert isinstance(rng_back, RandomGeneratorVariable)
        assert not isinstance(rng_back, XRandomGeneratorVariable)

    def test_rng_to_xrng_eval(self):
        """Cast tensor rng to xrng, draw, compile and eval."""
        rng_var = random_generator_type("rng")
        xrng_var = rng_to_xrng(rng_var)
        next_rng, x = xrng_var.normal(0, 1, extra_dims={"a": 4})

        fn = function([rng_var], [x])
        result = fn(np.random.default_rng(123))
        assert result[0].shape == (4,)

    def test_xrng_to_rng_eval(self):
        """Cast xrng to tensor rng, draw with tensor op, compile and eval."""

        xrng_var = xrng_fn("xrng")
        rng_var = xrng_to_rng(xrng_var)
        next_rng, x = rng_var.normal(0, 1, size=(4,))

        fn = function([xrng_var], [x])
        result = fn(np.random.default_rng(123))
        assert result[0].shape == (4,)

    def test_roundtrip_eval(self):
        """rng -> xrng -> rng roundtrip, compile and eval."""
        rng_var = random_generator_type("rng")
        xrng_var = rng_to_xrng(rng_var)
        rng_back = xrng_to_rng(xrng_var)
        next_rng, x = rng_back.normal(0, 1, size=(3,))

        fn = function([rng_var], [x])
        result = fn(np.random.default_rng(42))
        assert result[0].shape == (3,)


class TestXRandomGeneratorSharedVariable:
    """Test shared xtensor RNG variables."""

    def test_shared_xrng_default(self):
        sv = shared_xrng(name="xrng")
        assert isinstance(sv, XRandomGeneratorSharedVariable)
        assert isinstance(sv, XRandomGeneratorVariable)
        assert isinstance(sv.get_value(), np.random.Generator)

    def test_shared_xrng_with_value(self):
        gen = np.random.default_rng(42)
        sv = shared_xrng(gen, name="xrng42")
        assert isinstance(sv, XRandomGeneratorSharedVariable)

    def test_shared_xrng_methods(self):
        sv = shared_xrng(name="xrng")
        next_rng, draw = sv.normal(0, 1, extra_dims={"a": 3})
        assert isinstance(next_rng, XRandomGeneratorVariable)
        assert isinstance(draw, XTensorVariable)

    def test_shared_xrng_compiles(self):
        sv = shared_xrng(np.random.default_rng(42), name="xrng")
        next_rng, x = sv.normal(0, 1, extra_dims={"a": 5})
        fn = function([], [x])
        result = fn()
        assert result[0].shape == (5,)

    def test_shared_xrng_chain_compiles(self):
        sv = shared_xrng(np.random.default_rng(42), name="xrng")
        next_rng, x = sv.normal(0, 1, extra_dims={"a": 3})
        next_rng, y = next_rng.uniform(0, 1, extra_dims={"b": 5})
        fn = function([], [x, y])
        result = fn()
        assert result[0].shape == (3,)
        assert result[1].shape == (5,)

    def test_shared_xrng_rejects_non_generator(self):
        with pytest.raises(TypeError, match="Expected numpy.random.Generator"):
            shared_xrng(42)
