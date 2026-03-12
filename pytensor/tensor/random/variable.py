import copy
import warnings
from functools import wraps
from typing import TypeAlias

import numpy as np

import pytensor.tensor.random.basic as ptrb
from pytensor import config
from pytensor.compile.sharedvalue import SharedVariable, shared_constructor
from pytensor.graph.basic import OptionalApplyType, Variable
from pytensor.tensor.random.type import RandomGeneratorType, random_generator_type
from pytensor.tensor.variable import TensorVariable


RNG_AND_DRAW: TypeAlias = tuple["RandomGeneratorVariable", TensorVariable]


def warn_reuse(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self.tag, "used", False) and config.warn_rng.reuse:
            warnings.warn(
                f"RandomGeneratorVariable {self} has already been used. "
                "You probably want to use the new RandomGeneratorVariable that was returned when you used it.",
                UserWarning,
            )
        self.tag.used = True
        return func(self, *args, **kwargs)

    return wrapper


class _random_generator_py_operators:
    # Continuous distributions

    @warn_reuse
    def normal(self, loc=0.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.normal(loc, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def uniform(self, low=0.0, high=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.uniform(low, high, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def triangular(self, left, mode, right, size=None) -> RNG_AND_DRAW:
        return ptrb.triangular(
            left, mode, right, size=size, rng=self, return_next_rng=True
        )

    @warn_reuse
    def beta(self, alpha, beta, size=None) -> RNG_AND_DRAW:
        return ptrb.beta(alpha, beta, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def halfnormal(self, loc=0.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.halfnormal(loc, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def lognormal(self, mean=0.0, sigma=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.lognormal(mean, sigma, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def gamma(self, shape, rate=None, scale=None, size=None) -> RNG_AND_DRAW:
        return ptrb.gamma(
            shape, rate=rate, scale=scale, size=size, rng=self, return_next_rng=True
        )

    @warn_reuse
    def exponential(self, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.exponential(scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def weibull(self, shape, size=None) -> RNG_AND_DRAW:
        return ptrb.weibull(shape, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def logistic(self, loc=0, scale=1, size=None) -> RNG_AND_DRAW:
        return ptrb.logistic(loc, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def vonmises(self, mu, kappa, size=None) -> RNG_AND_DRAW:
        return ptrb.vonmises(mu, kappa, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def pareto(self, b, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.pareto(b, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def gumbel(self, loc=0.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.gumbel(loc, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def cauchy(self, loc=0.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.cauchy(loc, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def halfcauchy(self, loc=0.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.halfcauchy(loc, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def invgamma(self, shape, scale, size=None) -> RNG_AND_DRAW:
        return ptrb.invgamma(shape, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def wald(self, mean=1.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.wald(mean, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def truncexpon(self, b, loc=0.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.truncexpon(b, loc, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def laplace(self, loc=0.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.laplace(loc, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def t(self, df, loc=0.0, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.t(df, loc, scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def gengamma(self, alpha=1.0, p=1.0, lambd=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.gengamma(alpha, p, lambd, size=size, rng=self, return_next_rng=True)

    # Multivariate continuous distributions

    @warn_reuse
    def multivariate_normal(self, mean, cov, size=None, method=None) -> RNG_AND_DRAW:
        return ptrb.multivariate_normal(
            mean, cov, size=size, method=method, rng=self, return_next_rng=True
        )

    @warn_reuse
    def dirichlet(self, alphas, size=None) -> RNG_AND_DRAW:
        return ptrb.dirichlet(alphas, size=size, rng=self, return_next_rng=True)

    # Discrete distributions

    @warn_reuse
    def poisson(self, lam=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.poisson(lam, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def geometric(self, p, size=None) -> RNG_AND_DRAW:
        return ptrb.geometric(p, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def hypergeometric(self, ngood, nbad, nsample, size=None) -> RNG_AND_DRAW:
        return ptrb.hypergeometric(
            ngood, nbad, nsample, size=size, rng=self, return_next_rng=True
        )

    @warn_reuse
    def bernoulli(self, p, size=None) -> RNG_AND_DRAW:
        return ptrb.bernoulli(p, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def binomial(self, n, p, size=None) -> RNG_AND_DRAW:
        return ptrb.binomial(n, p, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def negative_binomial(self, n, p, size=None) -> RNG_AND_DRAW:
        return ptrb.negative_binomial(n, p, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def betabinom(self, n, a, b, size=None) -> RNG_AND_DRAW:
        return ptrb.betabinom(n, a, b, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def multinomial(self, n, p, size=None) -> RNG_AND_DRAW:
        return ptrb.multinomial(n, p, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def categorical(self, p, size=None) -> RNG_AND_DRAW:
        return ptrb.categorical(p, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def integers(self, low, high=None, size=None) -> RNG_AND_DRAW:
        return ptrb.integers(low, high, size=size, rng=self, return_next_rng=True)

    # Function-based distributions

    @warn_reuse
    def standard_normal(self, *, size=None) -> RNG_AND_DRAW:
        return ptrb.standard_normal(size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def chisquare(self, df, size=None) -> RNG_AND_DRAW:
        return ptrb.chisquare(df, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def rayleigh(self, scale=1.0, size=None) -> RNG_AND_DRAW:
        return ptrb.rayleigh(scale, size=size, rng=self, return_next_rng=True)

    @warn_reuse
    def choice(self, a, size=None, replace=True, p=None) -> RNG_AND_DRAW:
        return ptrb.choice(
            a, size=size, replace=replace, p=p, rng=self, return_next_rng=True
        )

    @warn_reuse
    def permutation(self, x) -> RNG_AND_DRAW:
        return ptrb.permutation(x, rng=self, return_next_rng=True)


class RandomGeneratorVariable(
    _random_generator_py_operators,
    Variable[RandomGeneratorType, OptionalApplyType],
):
    """The Variable type used for random number generator states."""


RandomGeneratorType.variable_type = RandomGeneratorVariable


def rng(name=None) -> RandomGeneratorVariable:
    """Create a new default random number generator variable.

    Returns
    -------
    RandomGeneratorVariable
        A new random number generator variable initialized with the default
        numpy random generator.
    """

    return random_generator_type(name=name)


class RandomGeneratorSharedVariable(SharedVariable, RandomGeneratorVariable):
    def __str__(self):
        return self.name or f"RNG({self.container!r})"


@shared_constructor.register(np.random.RandomState)
@shared_constructor.register(np.random.Generator)
def randomgen_constructor(
    value, name=None, strict=False, allow_downcast=None, borrow=False
):
    r"""`SharedVariable` constructor for NumPy's `Generator` and/or `RandomState`."""
    if isinstance(value, np.random.RandomState):
        raise TypeError(
            "`np.RandomState` is no longer supported in PyTensor. Use `np.random.Generator` instead."
        )

    rng_sv_type = RandomGeneratorSharedVariable
    rng_type = random_generator_type

    if not borrow:
        value = copy.deepcopy(value)

    return rng_sv_type(
        type=rng_type,
        value=value,
        strict=strict,
        allow_downcast=allow_downcast,
        name=name,
    )
