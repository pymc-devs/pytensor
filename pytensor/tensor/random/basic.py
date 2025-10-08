import abc
import warnings
from typing import Literal

import numpy as np
from numpy import broadcast_shapes as np_broadcast_shapes
from numpy import einsum as np_einsum
from numpy import sqrt as np_sqrt
from numpy.linalg import cholesky as np_cholesky
from numpy.linalg import eigh as np_eigh
from numpy.linalg import svd as np_svd

from pytensor.tensor import get_vector_length, specify_shape
from pytensor.tensor.basic import as_tensor_variable
from pytensor.tensor.math import sqrt
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import (
    broadcast_params,
    normalize_size_param,
)
from pytensor.tensor.utils import faster_broadcast_to, faster_ndindex


# Scipy.stats is considerably slow to import
# We import scipy.stats lazily inside `ScipyRandomVariable`
stats = None


try:
    broadcast_shapes = np.broadcast_shapes
except AttributeError:
    from numpy.lib.stride_tricks import _broadcast_shape

    def broadcast_shapes(*shapes):
        return _broadcast_shape(*[np.empty(x, dtype=[]) for x in shapes])


class ScipyRandomVariable(RandomVariable):
    r"""A class for straightforward `RandomVariable`\s that use SciPy-based samplers.

    By "straightforward" we mean `RandomVariable`\s for which the output shape
    is entirely determined by broadcasting the distribution parameters
    (e.g. basic scalar distributions).

    The more sophisticated shape logic performed by `RandomVariable` is avoided
    in order to reduce the amount of unnecessary steps taken to correct SciPy's
    shape-reducing defects.

    """

    @classmethod
    @abc.abstractmethod
    def rng_fn_scipy(cls, rng, *args, **kwargs):
        r"""

        `RandomVariable`\s implementations that want to use SciPy-based samplers
        need to implement this method instead of the base
        `RandomVariable.rng_fn`; otherwise their broadcast dimensions will be
        dropped by SciPy.

        """

    @classmethod
    def rng_fn(cls, *args, **kwargs):
        global stats
        if stats is None:
            import scipy.stats as stats
        size = args[-1]
        res = cls.rng_fn_scipy(*args, **kwargs)

        if np.ndim(res) == 0:
            # The sample is an `np.number`, and is not writeable, or non-NumPy
            # type, so we need to clone/create a usable NumPy result
            res = np.asarray(res)

        if size is None:
            # SciPy will sometimes drop broadcastable dimensions; we need to
            # check and, if necessary, add them back
            exp_shape = broadcast_shapes(*[np.shape(a) for a in args[1:-1]])
            if res.shape != exp_shape:
                return np.broadcast_to(res, exp_shape).copy()

        return res


class UniformRV(RandomVariable):
    r"""A uniform continuous random variable.

    The probability density function for `uniform` within the interval :math:`[l, h)` is:

    .. math::
        \begin{split}
            f(x; l, h) = \begin{cases}
                          \frac{1}{h-l}\quad \text{for $l \leq x \leq h$},\\
                           0\quad \text{otherwise}.
                       \end{cases}
        \end{split}

    """

    name = "uniform"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Uniform", "\\operatorname{Uniform}")

    def __call__(self, low=0.0, high=1.0, size=None, **kwargs):
        r"""Draw samples from a uniform distribution.

        The results are undefined when `high < low`.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        low
           Lower boundary :math:`l` of the output interval; all values generated
           will be greater than or equal to `low`.
        high
           Upper boundary :math:`h` of the output interval; all values generated
           will be less than or equal to `high`.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(low, high, size=size, **kwargs)


uniform = UniformRV()


class TriangularRV(RandomVariable):
    r"""A triangular continuous random variable.

    The probability density function for `triangular` within the interval :math:`[l, r)`
    and mode :math:`m` (where the peak of the distribution occurs) is:

    .. math::

        \begin{split}
            f(x; l, m, r) = \begin{cases}
                                \frac{2(x-l)}{(r-l)(m-l)}\quad \text{for $l \leq x \leq m$},\\
                                \frac{2(r-x)}{(r-l)(r-m)}\quad \text{for $m \leq x \leq r$},\\
                                0\quad \text{otherwise}.
                            \end{cases}
        \end{split}

    """

    name = "triangular"
    signature = "(),(),()->()"
    dtype = "floatX"
    _print_name = ("Triangular", "\\operatorname{Triangular}")

    def __call__(self, left, mode, right, size=None, **kwargs):
        r"""Draw samples from a triangular distribution.

        Signature
        ---------

        `(), (), () -> ()`

        Parameters
        ----------
        left
           Lower boundary :math:`l` of the output interval; all values generated
           will be greater than or equal to `left`.
        mode
           Mode :math:`m` of the distribution, where the peak occurs. Must be such
           that `left <= mode <= right`.
        right
           Upper boundary :math:`r` of the output interval; all values generated
           will be less than or equal to `right`. Must be larger than `left`.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(left, mode, right, size=size, **kwargs)


triangular = TriangularRV()


class BetaRV(RandomVariable):
    r"""A beta continuous random variable.

    The probability density function for `beta` in terms of its parameters :math:`\alpha`
    and :math:`\beta` is:

    .. math::

        f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha-1} (1-x)^{\beta-1}

    for :math:`0 \leq x \leq 1`. :math:`B` is the beta function defined as:

    .. math::

        B(\alpha, \beta) = \int_0^1 t^{\alpha-1} (1-t)^{\beta-1} \mathrm{d}t

    """

    name = "beta"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Beta", "\\operatorname{Beta}")

    def __call__(self, alpha, beta, size=None, **kwargs):
        r"""Draw samples from a beta distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        alpha
            Alpha parameter :math:`\alpha` of the distribution. Must be positive.
        beta
            Beta parameter :math:`\beta` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(alpha, beta, size=size, **kwargs)


beta = BetaRV()


class NormalRV(RandomVariable):
    r"""A normal continuous random variable.

    The probability density function for `normal` in terms of its location parameter (mean)
    :math:`\mu` and scale parameter (standard deviation) :math:`\sigma` is:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    for :math:`\sigma > 0`.

    """

    name = "normal"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Normal", "\\operatorname{Normal}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a normal distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        loc
            Mean :math:`\mu` of the normal distribution.
        scale
            Standard deviation :math:`\sigma` of the normal distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)


normal = NormalRV()


def standard_normal(*, size=None, rng=None, dtype=None):
    """Draw samples from a standard normal distribution.

    Signature
    ---------

    `nil -> ()`

    Parameters
    ----------
    size
        Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
        independent, identically distributed random variables are
        returned. Default is `None` in which case a single random variable
        is returned.

    """
    return normal(0.0, 1.0, size=size, rng=rng, dtype=dtype)


class HalfNormalRV(ScipyRandomVariable):
    r"""A half-normal continuous random variable.

    The probability density function for `halfnormal` in terms of its location parameter
    :math:`\mu` and scale parameter :math:`\sigma` is:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    for :math:`x \geq 0` and :math:`\sigma > 0`.

    """

    name = "halfnormal"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("HalfNormal", "\\operatorname{HalfNormal}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a half-normal distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        loc
            Location parameter :math:`\mu` of the distribution.
        scale
            Scale parameter :math:`\sigma` of the distribution.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, loc, scale, size):
        r"""Draw sample from a half-normal distribution using Scipy's generator.

        Parameters
        ----------
        loc
            Location parameter :math:`\mu` of the distribution.
        scale
            Scale parameter :math:`\sigma` of the distribution.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return stats.halfnorm.rvs(loc, scale, random_state=rng, size=size)


halfnormal = HalfNormalRV()


class LogNormalRV(RandomVariable):
    r"""A lognormal continuous random variable.

    The probability density function for `lognormal` in terms of the mean
    parameter :math:`\mu` and sigma parameter :math:`\sigma` is:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{x \sqrt{2 \pi \sigma^2}} e^{-\frac{(\ln(x)-\mu)^2}{2\sigma^2}}

    for :math:`x > 0` and :math:`\sigma > 0`.

    """

    name = "lognormal"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("LogNormal", "\\operatorname{LogNormal}")

    def __call__(self, mean=0.0, sigma=1.0, size=None, **kwargs):
        r"""Draw sample from a lognormal distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        mean
            Mean :math:`\mu` of the random variable's natural logarithm.
        sigma
            Standard deviation :math:`\sigma` of the random variable's natural logarithm.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(mean, sigma, size=size, **kwargs)


lognormal = LogNormalRV()


class GammaRV(RandomVariable):
    r"""A gamma continuous random variable.

    The probability density function for `gamma` in terms of the shape parameter
    :math:`\alpha` and rate parameter :math:`\beta` is:

    .. math::

        f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}

    for :math:`x \geq 0`, :math:`\alpha > 0` and :math:`\beta > 0`. :math:`\Gamma` is
    the gamma function:

    .. math::

        \Gamma(x) = \int_0^{\infty} t^{x-1} e^{-t} \mathrm{d}t

    """

    name = "gamma"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Gamma", "\\operatorname{Gamma}")

    def __call__(self, shape, scale, size=None, **kwargs):
        r"""Draw samples from a gamma distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        shape
            The shape :math:`\alpha` of the gamma distribution. Must be positive.
        scale
            The scale :math:`1/\beta` of the gamma distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(shape, scale, size=size, **kwargs)


_gamma = GammaRV()


def gamma(shape, rate=None, scale=None, **kwargs):
    # TODO: Remove helper when rate is deprecated
    if rate is not None and scale is not None:
        raise ValueError("Cannot specify both rate and scale")
    elif rate is None and scale is None:
        raise ValueError("Must specify scale")
    elif rate is not None:
        warnings.warn(
            "Gamma rate argument is deprecated and will stop working, use scale instead",
            FutureWarning,
        )
        scale = 1.0 / rate

    return _gamma(shape, scale, **kwargs)


def chisquare(df, size=None, **kwargs):
    r"""Draw samples from a chisquare distribution.

    The probability density function for `chisquare` in terms of the number of degrees of
    freedom :math:`k` is:

    .. math::
        f(x; k) = \frac{(1/2)^{k/2}}{\Gamma(k/2)} x^{k/2-1} e^{-x/2}
    for :math:`k > 2`. :math:`\Gamma` is the gamma function:

    .. math::
        \Gamma(x) = \int_0^{\infty} t^{x-1} e^{-t} \mathrm{d}t

    This variable is obtained by summing the squares :math:`k` independent, standard normally
    distributed random variables.

    Signature
    ---------
    `() -> ()`

    Parameters
    ----------
    df
        The number :math:`k` of degrees of freedom. Must be positive.
    size
        Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
        independent, identically distributed random variables are
        returned. Default is `None` in which case a single random variable
        is returned.
    """
    return gamma(shape=df / 2.0, scale=2.0, size=size, **kwargs)


def rayleigh(scale=1.0, *, size=None, **kwargs):
    r"""Draw samples from a Rayleigh distribution.

    The probability density function for `rayleigh` with parameter `scale` is given by:

    .. math::
        f(x; s) = \frac{x}{s^2} e^{-x^2/(2 s^2)}

    where :math:`s` is the scale parameter.

    This variable is obtained by taking the square root of the sum of the squares of
    two independent, standard normally distributed random variables.

    Signature
    ---------
    `() -> ()`

    Parameters
    ----------
    scale : float or array_like of floats, optional
        Scale parameter of the distribution (positive). Default is 1.0.
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., `(m, n, k)`, then `m * n * k` samples
        are drawn. Default is None, in which case the output shape is determined by the
        shape of `scale`.

    Notes
    -----
    `Rayleigh` is a special case of `chisquare` with ``df=2``.
    """

    scale = as_tensor_variable(scale)
    if size is None:
        size = scale.shape
    return sqrt(chisquare(df=2, size=size, **kwargs)) * scale


class ParetoRV(ScipyRandomVariable):
    r"""A pareto continuous random variable.

    The probability density function for `pareto` in terms of its shape parameter :math:`b` and
    scale parameter :math:`x_m` is:

    .. math::

        f(x; b, x_m) = \frac{b x_m^b}{x^{b+1}}

    and is defined for :math:`x \geq x_m`.

    """

    name = "pareto"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Pareto", "\\operatorname{Pareto}")

    def __call__(self, b, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a pareto distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        b
            The shape :math:`b` (or exponent) of the pareto distribution. Must be positive.
        scale
            The scale :math:`x_m` of the pareto distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(b, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, b, scale, size):
        return stats.pareto.rvs(b, scale=scale, size=size, random_state=rng)


pareto = ParetoRV()


class GumbelRV(ScipyRandomVariable):
    r"""A gumbel continuous random variable.

    The probability density function for `gumbel` in terms of its location parameter :math:`\mu` and
    scale parameter :math:`\beta` is:

    .. math::

        f(x; \mu, \beta) = \frac{\exp(-(x + e^{(x-\mu)/\beta})}{\beta}

    for :math:`\beta > 0`.

    """

    name = "gumbel"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Gumbel", "\\operatorname{Gumbel}")

    def __call__(
        self,
        loc: np.ndarray | float,
        scale: np.ndarray | float = 1.0,
        size: list[int] | int | None = None,
        **kwargs,
    ) -> RandomVariable:
        r"""Draw samples from a gumbel distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        loc
            The location parameter :math:`\mu` of the distribution.
        scale
            The scale :math:`\beta` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(
        cls,
        rng: np.random.Generator,
        loc: np.ndarray | float,
        scale: np.ndarray | float,
        size: list[int] | int | None,
    ) -> np.ndarray:
        return stats.gumbel_r.rvs(loc=loc, scale=scale, size=size, random_state=rng)


gumbel = GumbelRV()


class ExponentialRV(RandomVariable):
    r"""An exponential continuous random variable.

    The probability density function for `exponential` in terms of its scale parameter :math:`\beta` is:

    .. math::

        f(x; \beta) = \frac{\exp(-x / \beta)}{\beta}

    for :math:`x \geq 0` and :math:`\beta > 0`.

    """

    name = "exponential"
    signature = "()->()"
    dtype = "floatX"
    _print_name = ("Exponential", "\\operatorname{Exponential}")

    def __call__(self, scale=1.0, size=None, **kwargs):
        r"""Draw samples from an exponential distribution.

        Signature
        ---------

        `() -> ()`

        Parameters
        ----------
        scale
            The scale :math:`\beta` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(scale, size=size, **kwargs)


exponential = ExponentialRV()


class WeibullRV(RandomVariable):
    r"""A weibull continuous random variable.

    The probability density function for `weibull` in terms of its shape parameter :math:`k` is :

    .. math::

        f(x; k) = k x^{k-1} e^{-x^k}

    for :math:`x \geq 0` and :math:`k > 0`.

    """

    name = "weibull"
    signature = "()->()"
    dtype = "floatX"
    _print_name = ("Weibull", "\\operatorname{Weibull}")

    def __call__(self, shape, size=None, **kwargs):
        r"""Draw samples from a weibull distribution.

        Signature
        ---------

        `() -> ()`

        Parameters
        ----------
        shape
            The shape :math:`k` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(shape, size=size, **kwargs)


weibull = WeibullRV()


class LogisticRV(RandomVariable):
    r"""A logistic continuous random variable.

    The probability density function for `logistic` in terms of its location parameter :math:`\mu` and
    scale parameter :math:`s` is :

    .. math::

        f(x; \mu, s) = \frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^2}

    for :math:`s > 0`.

    """

    name = "logistic"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Logistic", "\\operatorname{Logistic}")

    def __call__(self, loc=0, scale=1, size=None, **kwargs):
        r"""Draw samples from a logistic distribution.

        Signature
        ---------

        `(), () -> ()`


        Parameters
        ----------
        loc
            The location parameter :math:`\mu` of the distribution.
        scale
            The scale :math:`s` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)


logistic = LogisticRV()


class VonMisesRV(RandomVariable):
    r"""A von Misses continuous random variable.

    The probability density function for `vonmisses` in terms of its mode :math:`\mu` and
    dispersion parameter :math:`\kappa` is :

    .. math::

        f(x; \mu, \kappa) = \frac{e^{\kappa \cos(x-\mu)}}{2 \pi I_0(\kappa)}

    for :math:`x \in [-\pi, \pi]` and :math:`\kappa > 0`. :math:`I_0` is the modified Bessel
    function of order 0.

    """

    name = "vonmises"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("VonMises", "\\operatorname{VonMises}")

    def __call__(self, mu, kappa, size=None, **kwargs):
        r"""Draw samples from a von Mises distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        mu
            The mode :math:`\mu` of the distribution.
        kappa
            The dispersion parameter :math:`\kappa` of the distribution. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(mu, kappa, size=size, **kwargs)


vonmises = VonMisesRV()


class MvNormalRV(RandomVariable):
    r"""A multivariate normal random variable.

    The probability density function for `multivariate_normal` in term of its location parameter
    :math:`\boldsymbol{\mu}` and covariance matrix :math:`\Sigma` is

    .. math::

        f(\boldsymbol{x}; \boldsymbol{\mu}, \Sigma) = \det(2 \pi \Sigma)^{-1/2}  \exp\left(-\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^T \Sigma (\boldsymbol{x} - \boldsymbol{\mu})\right)

    where :math:`\Sigma` is a positive semi-definite matrix.

    """

    name = "multivariate_normal"
    signature = "(n),(n,n)->(n)"
    dtype = "floatX"
    _print_name = ("MultivariateNormal", "\\operatorname{MultivariateNormal}")
    __props__ = ("name", "signature", "dtype", "inplace", "method")

    def __init__(self, *args, method: Literal["cholesky", "svd", "eigh"], **kwargs):
        super().__init__(*args, **kwargs)
        if method not in ("cholesky", "svd", "eigh"):
            raise ValueError(
                f"Unknown method {method}. The method must be one of 'cholesky', 'svd', or 'eigh'."
            )
        self.method = method

    def __call__(self, mean, cov, size=None, method=None, **kwargs):
        r""" "Draw samples from a multivariate normal distribution.

        Signature
        ---------

        `(n), (n,n) -> (n)`

        Parameters
        ----------
        mean
            Location parameter (mean) :math:`\boldsymbol{\mu}` of the distribution. Vector
            of length `N`.
        cov
            Covariance matrix :math:`\Sigma` of the distribution. Must be a symmetric
            and positive-semidefinite `NxN` matrix.
        size
            Given a size of, for example, `(m, n, k)`, `m * n * k` independent,
            identically distributed samples are generated. Because each sample
            is `N`-dimensional, the output shape is `(m, n, k, N)`. If no shape
            is specified, a single `N`-dimensional sample is returned.

        """
        if method is not None and method != self.method:
            # Recreate Op with the new method
            props = self._props_dict()
            props["method"] = method
            new_op = type(self)(**props)
            return new_op.__call__(mean, cov, size=size, method=method, **kwargs)
        return super().__call__(mean, cov, size=size, **kwargs)

    def rng_fn(self, rng, mean, cov, size):
        if size is None:
            size = np_broadcast_shapes(mean.shape[:-1], cov.shape[:-2])

        if self.method == "cholesky":
            A = np_cholesky(cov)
        elif self.method == "svd":
            A, s, _ = np_svd(cov)
            A *= np_sqrt(s, out=s)[..., None, :]
        else:
            w, A = np_eigh(cov)
            A *= np_sqrt(w, out=w)[..., None, :]

        out = rng.normal(size=(*size, mean.shape[-1]))
        np_einsum(
            "...ij,...j->...i",  # numpy doesn't have a batch matrix-vector product
            A,
            out,
            optimize=False,  # Nothing to optimize with two operands, skip costly setup
            out=out,
        )
        out += mean
        return out


multivariate_normal = MvNormalRV(method="cholesky")


class DirichletRV(RandomVariable):
    r"""A Dirichlet continuous random variable.

    The probability density function for `dirichlet` in terms of the vector of
    concentration parameters :math:`\boldsymbol{\alpha}` is:

    .. math::

        f(x; \boldsymbol{\alpha}) = \prod_{i=1}^k x_i^{\alpha_i-1}

    where :math:`x` is a vector, such that :math:`x_i > 0\;\forall i` and
    :math:`\sum_{i=1}^k x_i = 1`.

    """

    name = "dirichlet"
    signature = "(a)->(a)"
    dtype = "floatX"
    _print_name = ("Dirichlet", "\\operatorname{Dirichlet}")

    def __call__(self, alphas, size=None, **kwargs):
        r"""Draw samples from a dirichlet distribution.

        Signature
        ---------

        `(k) -> (k)`

        Parameters
        ----------
        alphas
            A sequence of concentration parameters :math:`\boldsymbol{\alpha}` of the
            distribution. A sequence of length `k` will produce samples of length `k`.
        size
            Given a size of, for example, `(r, s, t)`, `r * s * t` independent,
            identically distributed samples are generated. Because each sample
            is `k`-dimensional, the output shape is `(r, s, t, k)`. If no shape
            is specified, a single `k`-dimensional sample is returned.

        """
        return super().__call__(alphas, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, alphas, size):
        if alphas.ndim > 1:
            if size is not None:
                alphas = faster_broadcast_to(alphas, size + alphas.shape[-1:])

            samples_shape = alphas.shape
            samples = np.empty(samples_shape)
            for index in faster_ndindex(samples_shape[:-1]):
                samples[index] = rng.dirichlet(alphas[index])
            return samples
        else:
            return rng.dirichlet(alphas, size=size)


dirichlet = DirichletRV()


class PoissonRV(RandomVariable):
    r"""A poisson discrete random variable.

    The probability mass function for `poisson` in terms of the expected number
    of events :math:`\lambda` is:

    .. math::

        f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

    for :math:`\lambda > 0`.

    """

    name = "poisson"
    signature = "()->()"
    dtype = "int64"
    _print_name = ("Poisson", "\\operatorname{Poisson}")

    def __call__(self, lam=1.0, size=None, **kwargs):
        r"""Draw samples from a poisson distribution.

        Signature
        ---------

        `() -> ()`

        Parameters
        ----------
        lam
            Expected number of events :math:`\lambda`. Must be positive.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
            independent, identically distributed random variables are
            returned. Default is `None` in which case a single random variable
            is returned.

        """
        return super().__call__(lam, size=size, **kwargs)


poisson = PoissonRV()


class GeometricRV(RandomVariable):
    r"""A geometric discrete random variable.

    The probability mass function for `geometric` for the number of successes :math:`k`
    before the first failure in terms of the probability of success :math:`p` of a single
    trial is:

    .. math::

        f(k; p) = p^{k-1}(1-p)

    for :math:`0 \geq p \geq 1`.

    """

    name = "geometric"
    signature = "()->()"
    dtype = "int64"
    _print_name = ("Geometric", "\\operatorname{Geometric}")

    def __call__(self, p, size=None, **kwargs):
        r"""Draw samples from a geometric distribution.

        Signature
        ---------

        `() -> ()`

        Parameters
        ----------
        p
            Probability of success :math:`p` of an individual trial.
        size
            Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n *
            k` independent, identically distributed samples are returned.
            Default is `None` in which case a single sample is returned.

        """
        return super().__call__(p, size=size, **kwargs)


geometric = GeometricRV()


class HyperGeometricRV(RandomVariable):
    r"""A hypergeometric discrete random variable.

    The probability mass function for `hypergeometric` for the number of
    successes :math:`k` in :math:`n` draws without replacement, from a
    finite population of size :math:`N` with :math:`K` desired items is:

    .. math::

        f(k; n, N, K) = \frac{{K \choose k} {N-K \choose n-k}}{{N \choose n}}

    """

    name = "hypergeometric"
    signature = "(),(),()->()"
    dtype = "int64"
    _print_name = ("HyperGeometric", "\\operatorname{HyperGeometric}")

    def __call__(self, ngood, nbad, nsample, size=None, **kwargs):
        r"""Draw samples from a geometric distribution.

        Signature
        ---------

        `(), (), () -> ()`

        Parameters
        ----------
        ngood
            Number :math:`K` of desirable items in the population. Positive integer.
        nbad
            Number :math:`N-K` of undesirable items in the population. Positive integer.
        nsample
            Number :math:`n` of items sampled. Must be less than :math:`N`,
            i.e. `ngood + nbad`.` Positive integer.
        size
           Sample shape. If the given size is, e.g. `(m, n, k)` then `m * n * k`
           independent, identically distributed samples are returned. Default is
           `None` in which case a single sample is returned.

        """
        return super().__call__(ngood, nbad, nsample, size=size, **kwargs)


hypergeometric = HyperGeometricRV()


class CauchyRV(ScipyRandomVariable):
    r"""A Cauchy continuous random variable.

    The probability density function for `cauchy` in terms of its location
    parameter :math:`x_0` and scale parameter :math:`\gamma` is:

    .. math::

        f(x; x_0, \gamma) = \frac{1}{\pi \gamma \left(1 + (\frac{x-x_0}{\gamma})^2\right)}

    where :math:`\gamma > 0`.

    """

    name = "cauchy"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Cauchy", "\\operatorname{Cauchy}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a Cauchy distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        loc
            Location parameter :math:`x_0` of the distribution.
        scale
            Scale parameter :math:`\gamma` of the distribution. Must be
            positive.
        size
           Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
           independent, identically distributed samples are returned. Default is
           `None` in which case a single sample is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, loc, scale, size):
        return stats.cauchy.rvs(loc=loc, scale=scale, random_state=rng, size=size)


cauchy = CauchyRV()


class HalfCauchyRV(ScipyRandomVariable):
    r"""A half-Cauchy continuous random variable.

    The probability density function for `halfcauchy` in terms of its location
    parameter :math:`x_0` and scale parameter :math:`\gamma` is:

    .. math::

        f(x; x_0, \gamma) = \frac{1}{\pi \gamma \left(1 + (\frac{x-x_0}{\gamma})^2\right)}

    for :math:`x \geq 0` where :math:`\gamma > 0`.

    """

    name = "halfcauchy"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("HalfCauchy", "\\operatorname{HalfCauchy}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a half-Cauchy distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        loc
            Location parameter :math:`x_0` of the distribution.
        scale
            Scale parameter :math:`\gamma` of the distribution. Must be
            positive.
        size
           Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
           independent, identically distributed samples are returned. Default is
           `None`, in which case a single sample is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, loc, scale, size):
        return stats.halfcauchy.rvs(loc=loc, scale=scale, random_state=rng, size=size)


halfcauchy = HalfCauchyRV()


class InvGammaRV(RandomVariable):
    r"""An inverse-gamma continuous random variable.

    The probability density function for `invgamma` in terms of its shape
    parameter :math:`\alpha` and scale parameter :math:`\beta` is:

    .. math::

        f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-(\alpha+1)} \exp\left(-\frac{\beta}{x}\right)

    for :math:`x > 0`, where :math:`\alpha > 0` and :math:`\beta > 0`. :math:`Gamma` is the gamma function :

    .. math::

        \Gamma(x) = \int_0^{\infty} t^{x-1} e^{-t} \mathrm{d}t

    """

    name = "invgamma"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("InverseGamma", "\\operatorname{InverseGamma}")

    def __call__(self, shape, scale, size=None, **kwargs):
        r"""Draw samples from an inverse-gamma distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        shape
            Shape parameter :math:`\alpha` of the distribution. Must be positive.
        scale
            Scale parameter :math:`\beta` of the distribution. Must be
            positive.
        size
           Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
           independent, identically distributed sample are returned. Default is
           `None`, in which case a single sample is returned.

        """
        return super().__call__(shape, scale, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, shape, scale, size):
        return 1 / rng.gamma(shape, 1 / scale, size)


invgamma = InvGammaRV()


class WaldRV(RandomVariable):
    r"""A Wald (or inverse Gaussian) continuous random variable.

    The probability density function for `wald` in terms of its mean
    parameter :math:`\mu` and shape parameter :math:`\lambda` is:

    .. math::

        f(x; \mu, \lambda) = \sqrt{\frac{\lambda}{2 \pi x^3}} \exp\left(-\frac{\lambda (x-\mu)^2}{2 \mu^2 x}\right)

    for :math:`x > 0`, where :math:`\mu > 0` and :math:`\lambda > 0`.

    """

    name = "wald"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name_ = ("Wald", "\\operatorname{Wald}")

    def __call__(self, mean=1.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a Wald distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        mean
            Mean parameter :math:`\mu` of the distribution. Must be positive.
        shape
            Shape parameter :math:`\lambda` of the distribution. Must be
            positive.
        size
           Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
           independent, identically distributed samples are returned. Default is
           `None`, in which case a single sample is returned.

        """
        return super().__call__(mean, scale, size=size, **kwargs)


wald = WaldRV()


class TruncExponentialRV(ScipyRandomVariable):
    r"""A truncated exponential continuous random variable.

    The probability density function for `truncexp` in terms of its shape
    parameter :math:`b`, location parameter :math:`\alpha` and scale
    parameter :math:`\beta` is:

    .. math::

        f(x; b, \alpha, \beta) = \frac{\exp(-(x-\alpha)/\beta)}{\beta (1-\exp(-b))}

    for :math:`0 \leq x \leq b` and :math:`\beta > 0`.

    """

    name = "truncexpon"
    signature = "(),(),()->()"
    dtype = "floatX"
    _print_name = ("TruncatedExponential", "\\operatorname{TruncatedExponential}")

    def __call__(self, b, loc=0.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a truncated exponential distribution.

        Signature
        ---------

        `(), (), () -> ()`

        Parameters
        ----------
        b
            Shape parameter :math:`b` of the distribution. Must be positive.
        loc
            Location parameter :math:`\alpha` of the distribution.
        scale
            Scale parameter :math:`\beta` of the distribution. Must be
            positive.
        size
           Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
           independent, identically distributed samples are returned. Default is
           `None` in which case a single sample is returned.

        """
        return super().__call__(b, loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, b, loc, scale, size):
        return stats.truncexpon.rvs(
            b, loc=loc, scale=scale, size=size, random_state=rng
        )


truncexpon = TruncExponentialRV()


class StudentTRV(ScipyRandomVariable):
    r"""A Student's t continuous random variable.

    The probability density function for `t` in terms of its degrees of freedom
    parameter :math:`\nu`, location parameter :math:`\mu` and scale
    parameter :math:`\sigma` is:

    .. math::

        f(x; \nu, \alpha, \beta) = \frac{\Gamma(\frac{\nu + 1}{2})}{\Gamma(\frac{\nu}{2})} \left(\frac{1}{\pi\nu\sigma}\right)^{\frac{1}{2}} \left[1+\frac{(x-\mu)^2}{\nu\sigma}\right]^{-\frac{\nu+1}{2}}

    for :math:`\nu > 0`, :math:`\sigma > 0`.

    """

    name = "t"
    signature = "(),(),()->()"
    dtype = "floatX"
    _print_name = ("StudentT", "\\operatorname{StudentT}")

    def __call__(self, df, loc=0.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a Student's t distribution.

        Signature
        ---------

        `(), (), () -> ()`

        Parameters
        ----------
        df
            Degrees of freedom parameter :math:`\nu` of the distribution. Must be
            positive.
        loc
            Location parameter :math:`\mu` of the distribution.
        scale
            Scale parameter :math:`\sigma` of the distribution. Must be
            positive.
        size
           Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
           independent, identically distributed samples are returned. Default is
           `None` in which case a single sample is returned.

        """
        return super().__call__(df, loc, scale, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, df, loc, scale, size):
        return stats.t.rvs(df, loc=loc, scale=scale, size=size, random_state=rng)


t = StudentTRV()


class BernoulliRV(ScipyRandomVariable):
    r"""A Bernoulli discrete random variable.

    The probability mass function for `bernoulli` in terms of the probability
    of success :math:`p` of a single trial is:


    .. math::

        \begin{split}
            f(k; p) = \begin{cases}
                                (1-p)\quad \text{if $k = 0$},\\
                                p\quad \text{if $k=1$}\\
                        \end{cases}
        \end{split}

    where :math:`0 \leq p \leq 1`.

    """

    name = "bernoulli"
    signature = "()->()"
    dtype = "int64"
    _print_name = ("Bernoulli", "\\operatorname{Bernoulli}")

    def __call__(self, p, size=None, **kwargs):
        r"""Draw samples from a Bernoulli distribution.

        Signature
        ---------

        `() -> ()`

        Parameters
        ----------
        p
            Probability of success :math:`p` of a single trial.
        size
            Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
            independent, identically distributed samples are returned. Default
            is `None` in which case a single sample is returned.

        """
        return super().__call__(p, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, p, size):
        return stats.bernoulli.rvs(p, size=size, random_state=rng)


bernoulli = BernoulliRV()


class LaplaceRV(RandomVariable):
    r"""A Laplace continuous random variable.

    The probability density function for `laplace` in terms of its location
    parameter :math:`\mu` and scale parameter :math:`\lambda` is:

    .. math::

        f(x; \mu, \lambda) = \frac{1}{2 \lambda} \exp\left(-\frac{|x-\mu|}{\lambda}\right)

    with :math:`\lambda > 0`.

    """

    name = "laplace"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Laplace", "\\operatorname{Laplace}")

    def __call__(self, loc=0.0, scale=1.0, size=None, **kwargs):
        r"""Draw samples from a Laplace distribution.

        Signature
        ---------

        `(), () -> ()`


        Parameters
        ----------
        loc
            Location parameter :math:`\mu` of the distribution.
        scale
            Scale parameter :math:`\lambda` of the distribution. Must be
            positive.
        size
            Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
            independent, identically distributed samples are returned. Default
            is `None` in which case a single sample is returned.

        """
        return super().__call__(loc, scale, size=size, **kwargs)


laplace = LaplaceRV()


class BinomialRV(RandomVariable):
    r"""A binomial discrete random variable.

    The probability mass function for `binomial` for the number :math:`k` of successes
    in terms of the probability of success :math:`p` of a single trial and the number
    :math:`n` of trials is:

    .. math::

            f(k; p, n) = {n \choose k} p^k (1-p)^{n-k}

    """

    name = "binomial"
    signature = "(),()->()"
    dtype = "int64"
    _print_name = ("Binomial", "\\operatorname{Binomial}")

    def __call__(self, n, p, size=None, **kwargs):
        r"""Draw samples from a binomial distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        n
            Number of trials :math:`n`. Must be a positive integer.
        p
            Probability of success :math:`p` of a single trial.
        size
            Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
            independent, identically distributed samples are returned. Default
            is `None` in which case a single sample is returned.

        """
        return super().__call__(n, p, size=size, **kwargs)


binomial = BinomialRV()


class NegBinomialRV(ScipyRandomVariable):
    r"""A negative binomial discrete random variable.

    The probability mass function for `nbinom` for the number :math:`k` of draws
    before observing the :math:`n`\th success in terms of the probability of
    success :math:`p` of a single trial is:

    .. math::

            f(k; p, n) = {k+n-1 \choose n-1} p^n (1-p)^{k}

    """

    name = "negative_binomial"
    signature = "(),()->()"
    dtype = "int64"
    _print_name = ("NegativeBinomial", "\\operatorname{NegativeBinomial}")

    def __call__(self, n, p, size=None, **kwargs):
        r"""Draw samples from a negative binomial distribution.

        Signature
        ---------

        `(), () -> ()`

        Parameters
        ----------
        n
            Number of successes :math:`n`. Must be a positive integer.
        p
            Probability of success :math:`p` of a single trial.
        size
            Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
            independent, identically distributed samples are returned. Default
            is `None` in which case a single sample is returned.

        """
        return super().__call__(n, p, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, n, p, size):
        return stats.nbinom.rvs(n, p, size=size, random_state=rng)


nbinom = negative_binomial = NegBinomialRV()


class BetaBinomialRV(ScipyRandomVariable):
    r"""A beta-binomial discrete random variable.

    The probability mass function for `betabinom` in terms of its shape
    parameters :math:`n \geq 0`, :math:`a > 0`, :math:`b > 0` and the probability
    :math:`p` is:

    .. math::

            f(k; p, n, a, b) = {n \choose k} \frac{\operatorname{B}(k+a, n-k+b)}{\operatorname{B}(a,b)}

    where :math:`\operatorname{B}` is the beta function:

    .. math::

        \operatorname{B}(a, b) = \int_0^1 t^{a-1} (1-t)^{b-1} \mathrm{d}t

    """

    name = "beta_binomial"
    signature = "(),(),()->()"
    dtype = "int64"
    _print_name = ("BetaBinomial", "\\operatorname{BetaBinomial}")

    def __call__(self, n, a, b, size=None, **kwargs):
        r"""Draw samples from a beta-binomial distribution.

        Signature
        ---------

        `(), (), () -> ()`

        Parameters
        ----------
        n
            Shape parameter :math:`n`. Must be a positive integer.
        a
            Shape parameter :math:`a`. Must be positive.
        b
            Shape parameter :math:`b`. Must be positive.
        size
            Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
            independent, identically distributed samples are returned. Default
            is `None` in which case a single sample is returned.

        """
        return super().__call__(n, a, b, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, n, a, b, size):
        return stats.betabinom.rvs(n, a, b, size=size, random_state=rng)


betabinom = BetaBinomialRV()


class GenGammaRV(ScipyRandomVariable):
    r"""A generalized gamma continuous random variable.

    The probability density function of `gengamma` in terms of its scale parameter
    :math:`\alpha` and other parameters :math:`p` and :math:`\lambda` is:

    .. math::

            f(x; \alpha, \lambda, p) = \frac{p/\lambda^\alpha}{\Gamma(\alpha/p)} x^{\alpha-1} e^{-(x/\lambda)^p}

    for :math:`x > 0`, where :math:`\alpha, \lambda, p > 0`.

    """

    name = "gengamma"
    signature = "(),(),()->()"
    dtype = "floatX"
    _print_name = ("GeneralizedGamma", "\\operatorname{GeneralizedGamma}")

    def __call__(self, alpha=1.0, p=1.0, lambd=1.0, size=None, **kwargs):
        r"""Draw samples from a generalized gamma distribution.

        Signature
        ---------

        `(), (), () -> ()`

        Parameters
        ----------
        alpha
            Parameter :math:`\alpha`. Must be positive.
        p
            Parameter :math:`p`. Must be positive.
        lambd
            Scale parameter :math:`\lambda`. Must be positive.
        size
            Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
            independent, identically distributed samples are
            returned. Default is `None` in which case a single sample
            is returned.

        """
        return super().__call__(alpha, p, lambd, size=size, **kwargs)

    @classmethod
    def rng_fn_scipy(cls, rng, alpha, p, lambd, size):
        return stats.gengamma.rvs(
            alpha / p, p, scale=lambd, size=size, random_state=rng
        )


gengamma = GenGammaRV()


class MultinomialRV(RandomVariable):
    r"""A multinomial discrete random variable.

    The probability mass function of `multinomial` in terms of the number
    of experiments :math:`n` and the probabilities :math:`p_1, \dots, p_k`
    of the :math:`k` different possible outcomes is:


    .. math::

        f(x_1,\dots,x_k; n, p_1, \dots, p_k) = \frac{n!}{x_1! \dots x_k!} \prod_{i=1}^k x_i^{p_i}


    where :math:`n>0` and :math:`\sum_{i=1}^k p_i = 1`.

    Notes
    -----
    The length of the support dimension is determined by the last
    dimension in the *second* parameter (i.e.  the probabilities vector).

    """

    name = "multinomial"
    signature = "(),(p)->(p)"
    dtype = "int64"
    _print_name = ("Multinomial", "\\operatorname{Multinomial}")

    def __call__(self, n, p, size=None, **kwargs):
        r"""Draw samples from a discrete multinomial distribution.

        Signature
        ---------

        `(), (n) -> (n)`

        Parameters
        ----------
        n
            Number of experiments :math:`n`. Must be a positive integer.
        p
            Probabilities of each of the :math:`k` different outcomes.
        size
            Given a size of, for example, `(r, s, t)`, `r * s * t` independent,
            identically distributed samples are generated. Because each sample
            is `k`-dimensional, the output shape is `(r, s, t, k)`. If no shape
            is specified, a single `k`-dimensional sample is returned.

        """
        return super().__call__(n, p, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, n, p, size):
        if n.ndim > 0 or p.ndim > 1:
            if size is None:
                n, p = broadcast_params([n, p], [0, 1])
            else:
                n = faster_broadcast_to(n, size)
                p = faster_broadcast_to(p, size + p.shape[-1:])

            res = np.empty(p.shape, dtype=cls.dtype)
            for idx in faster_ndindex(p.shape[:-1]):
                res[idx] = rng.multinomial(n[idx], p[idx])
            return res
        else:
            return rng.multinomial(n, p, size=size)


multinomial = MultinomialRV()


vsearchsorted = np.vectorize(np.searchsorted, otypes=[int], signature="(n),()->()")


class CategoricalRV(RandomVariable):
    r"""A categorical discrete random variable.

    The probability mass function of `categorical` in terms of its :math:`N` event
    probabilities :math:`p_1, \dots, p_N` is:

    .. math::

        P(k=i) = p_k

    where :math:`\sum_i p_i = 1`.

    """

    name = "categorical"
    signature = "(p)->()"
    dtype = "int64"
    _print_name = ("Categorical", "\\operatorname{Categorical}")

    def __call__(self, p, size=None, **kwargs):
        r"""Draw samples from a discrete categorical distribution.

        Signature
        ---------

        `(j) -> ()`

        Parameters
        ----------
        p
            An array that contains the :math:`N` event probabilities.
        size
            Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
            independent, identically distributed random samples are
            returned. Default is `None`, in which case a single sample
            is returned.

        """
        return super().__call__(p, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, p, size):
        if size is None:
            size = p.shape[:-1]
        else:
            # Check that `size` does not define a shape that would be broadcasted
            # to `p.shape[:-1]` in the call to `vsearchsorted` below.
            if len(size) < (p.ndim - 1):
                raise ValueError("`size` is incompatible with the shape of `p`")
            # zip strict not specified because we are in a hot loop
            for s, ps in zip(reversed(size), reversed(p.shape[:-1])):
                if s == 1 and ps != 1:
                    raise ValueError("`size` is incompatible with the shape of `p`")

        unif_samples = rng.uniform(size=size)
        samples = vsearchsorted(p.cumsum(axis=-1), unif_samples)

        return samples


categorical = CategoricalRV()


class IntegersRV(RandomVariable):
    r"""A discrete uniform random variable.

    Only available for `RandomGeneratorType`. Use `randint` with `RandomStateType`\s.

    """

    name = "integers"
    signature = "(),()->()"
    dtype = "int64"
    _print_name = ("integers", "\\operatorname{integers}")

    def __call__(self, low, high=None, size=None, **kwargs):
        r"""Draw samples from a discrete uniform distribution.

        Signature
        ---------

        `() -> ()`

        Parameters
        ----------
        low
            Lower boundary of the output interval.  All values generated
            will be greater than or equal to `low` (inclusive).
        high
            Upper boundary of the output interval.  All values generated
            will be smaller than `high` (exclusive).
        size
            Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
            independent, identically distributed samples are
            returned. Default is `None`, in which case a single sample
            is returned.

        """
        if high is None:
            low, high = 0, low
        return super().__call__(low, high, size=size, **kwargs)


integers = IntegersRV()


class ChoiceWithoutReplacement(RandomVariable):
    """Randomly choose an element in a sequence."""

    name = "choice_without_replacement"
    dtype = None
    _print_name = (
        "choice_without_replacement",
        "\\operatorname{choice_without_replacement}",
    )

    @property
    def has_p_param(self) -> bool:
        return len(self.ndims_params) == 3

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        a, *_, core_shape = dist_params
        a_shape = tuple(a.shape) if param_shapes is None else tuple(param_shapes[0])
        a_batch_ndim = len(a_shape) - self.ndims_params[0]
        a_core_shape = a_shape[a_batch_ndim:]
        core_shape_ndim = core_shape.type.ndim
        if core_shape_ndim > 1:
            # Batch core shapes are only valid if homogeneous or broadcasted,
            # as otherwise they would imply ragged choice arrays
            core_shape = core_shape[(0,) * (core_shape_ndim - 1)]
        return tuple(core_shape) + a_core_shape[1:]

    def rng_fn(self, *params):
        if self.has_p_param:
            rng, a, p, core_shape, size = params
        else:
            rng, a, core_shape, size = params
            p = None

        if core_shape.ndim > 1:
            core_shape = core_shape[(0,) * (core_shape.ndim - 1)]
        core_shape = tuple(core_shape)

        batch_ndim = a.ndim - self.ndims_params[0]

        if batch_ndim == 0:
            # Numpy choice fails with size=() if a.ndim > 1 is batched
            # https://github.com/numpy/numpy/issues/26518
            if core_shape == ():
                core_shape = None
            return rng.choice(a, p=p, size=core_shape, replace=False)

        # Numpy choice doesn't have a concept of batch dims
        if size is None:
            if p is None:
                size = a.shape[:batch_ndim]
            else:
                size = np.broadcast_shapes(
                    a.shape[:batch_ndim],
                    p.shape[:batch_ndim],
                )

        a = faster_broadcast_to(a, size + a.shape[batch_ndim:])
        if p is not None:
            p = faster_broadcast_to(p, size + p.shape[batch_ndim:])

        a_indexed_shape = a.shape[len(size) + 1 :]
        out = np.empty(size + core_shape + a_indexed_shape, dtype=a.dtype)
        for idx in faster_ndindex(size):
            out[idx] = rng.choice(
                a[idx], p=None if p is None else p[idx], size=core_shape, replace=False
            )
        return out


def choice(a, size=None, replace=True, p=None, rng=None):
    r"""Generate a random sample from an array.


    Parameters
    ----------
    a
        The array from which to randomly sample an element. If an int,
        a sample is generated from `pytensor.tensor.arange(a)`.
    p
        The probabilities associated with each entry in `a`. If not
        given, all elements have equal probability.
    replace
        When `True`, sampling is performed with replacement.
    size
        Sample shape. If the given size is `(m, n, k)`, then `m * n *
        k` independent samples are returned. Default is `None`, in
        which case a single sample is returned.
    """
    a = as_tensor_variable(a)
    a_size = a if (a.type.ndim == 0) else a.shape[0]

    if p is not None:
        p = specify_shape(p, (a_size,))

    if replace or size is None:
        # In this case we build an expression out of simpler RVs
        # This is equivalent to the numpy implementation:
        # https://github.com/numpy/numpy/blob/2a9b9134270371b43223fc848b753fceab96b4a5/numpy/random/_generator.pyx#L905-L914
        if p is None:
            idxs = integers(0, a_size, size=size, rng=rng)
        else:
            idxs = categorical(p, size=size, rng=rng)

        if a.type.ndim == 0:
            # A was an implicit arange, we don't need to do any indexing
            # TODO: Add rewrite for this optimization if users passed arange
            return idxs

        # TODO: Can use take(a, idxs, axis) to support numpy axis argument to choice
        return a[idxs]

    # Sampling with p is not as trivial
    # It involves some form of rejection sampling or iterative shuffling under the hood.
    # We use a specialized RandomVariable Op for these case.

    # Because choice happens on a single axis, the core case includes a.ndim-1 dimensions
    # Furthermore, due to replace=False the draws are not independent, and the core case of
    # this RV includes the user provided size.
    # If we have a tensor3 and a size=(2, 3) the signature of the underlying RV is
    # "(a0, a1, a2), (2) -> (s0, s1, a1, a2)" if p is None and
    # "(a0, a1, a2), (a0), (2) -> (s0, s1, a1, a2)" otherwise

    core_shape = normalize_size_param(size)
    core_shape_length = get_vector_length(core_shape)
    a_ndim = a.type.ndim
    dtype = a.type.dtype

    a_dims = [f"a{i}" for i in range(a_ndim)]
    a_sig = ",".join(a_dims)
    idx_dims = [f"s{i}" for i in range(core_shape_length)]
    if a_ndim == 0:
        p_sig = "a"
        out_dims = idx_dims
    else:
        p_sig = a_dims[0]
        out_dims = idx_dims + a_dims[1:]
    out_sig = ",".join(out_dims)

    if p is None:
        signature = f"({a_sig}),({core_shape_length})->({out_sig})"
    else:
        signature = f"({a_sig}),({p_sig}),({core_shape_length})->({out_sig})"

    op = ChoiceWithoutReplacement(signature=signature, dtype=dtype)

    params = (a, core_shape) if p is None else (a, p, core_shape)
    return op(*params, size=None, rng=rng)


class PermutationRV(RandomVariable):
    """Randomly shuffle a sequence."""

    name = "permutation"
    _print_name = ("permutation", "\\operatorname{permutation}")

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        [x] = dist_params
        x_shape = tuple(x.shape if param_shapes is None else param_shapes[0])
        if self.ndims_params[0] == 0:
            # Implicit arange, this is only valid for homogeneous arrays
            # Otherwise it would imply a ragged permutation array.
            return (x.ravel()[0],)
        else:
            batch_x_ndim = x.type.ndim - self.ndims_params[0]
            return x_shape[batch_x_ndim:]

    def rng_fn(self, rng, x, size):
        # We don't have access to the node in rng_fn :(
        batch_ndim = x.ndim - self.ndims_params[0]

        if batch_ndim:
            # rng.permutation has no concept of batch dims
            if size is None:
                size = x.shape[:batch_ndim]
            else:
                x = faster_broadcast_to(x, size + x.shape[batch_ndim:])

            out = np.empty(size + x.shape[batch_ndim:], dtype=x.dtype)
            for idx in faster_ndindex(size):
                out[idx] = rng.permutation(x[idx])
            return out

        else:
            return rng.permutation(x.item() if self.ndims_params[0] == 0 else x)


def permutation(x, **kwargs):
    r"""Randomly permute a sequence or a range of values.

    Signature
    ---------

    `() -> (x)` if x is a scalar, `(*x) -> (*x)` otherwise

    Parameters
    ----------
    x
        If `x` is an integer, randomly permute `np.arange(x)`. If `x` is a sequence,
        shuffle its elements randomly.

    """
    x = as_tensor_variable(x)
    x_ndim = x.type.ndim
    x_dtype = x.type.dtype
    # PermutationRV has a signature () -> (x) if x is a scalar
    # and (*x) -> (*x) otherwise, with has many entries as the dimensionsality of x
    if x_ndim == 0:
        signature = "()->(x)"
    else:
        arg_sig = ",".join(f"x{i}" for i in range(x_ndim))
        signature = f"({arg_sig})->({arg_sig})"
    return PermutationRV(signature=signature, dtype=x_dtype)(x, **kwargs)


__all__ = [
    "bernoulli",
    "beta",
    "betabinom",
    "binomial",
    "categorical",
    "cauchy",
    "chisquare",
    "choice",
    "dirichlet",
    "exponential",
    "gamma",
    "gengamma",
    "geometric",
    "gumbel",
    "halfcauchy",
    "halfnormal",
    "hypergeometric",
    "integers",
    "invgamma",
    "laplace",
    "logistic",
    "lognormal",
    "multinomial",
    "multivariate_normal",
    "nbinom",
    "negative_binomial",
    "normal",
    "pareto",
    "permutation",
    "poisson",
    "standard_normal",
    "t",
    "triangular",
    "truncexpon",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
]
