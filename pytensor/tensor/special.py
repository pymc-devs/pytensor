import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from pytensor.configdefaults import config
from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph.replace import _vectorize_node
from pytensor.tensor import as_tensor_variable
from pytensor.tensor.elemwise import get_normalized_batch_axes
from pytensor.tensor.math import (
    eq,
    erfc,
    erfcinv,
    erfcx,
    exp,
    floor,
    gamma,
    gammainc,
    gammaincc,
    gammainccinv,
    gammaln,
    le,
    log,
    log1p,
    lt,
    mul,
    softplus,
    sqr,
    sum,
    switch,
)
from pytensor.tensor.symbolic import TensorSymbolicOp


class Softmax(TensorSymbolicOp):
    r"""Softmax activation function.

    :math:`\sigma(\mathbf{x})_j = \frac{e^{x_j}}{\sum_k e^{x_k}}`

    Includes the numerical stabilization trick (subtracting the maximum).
    """

    __props__ = ("axis",)

    def __init__(self, *, axis, **kwargs):
        if isinstance(axis, (list, tuple)):
            axis = tuple(axis)
        elif isinstance(axis, int):
            axis = (axis,)
        self.axis = axis
        super().__init__(**kwargs)

    def build_inner_graph(self, x):
        x_stable = x - x.max(axis=self.axis, keepdims=True)
        e_x = exp(x_stable)
        return [e_x / e_x.sum(axis=self.axis, keepdims=True)]

    def pullback(self, inputs, outputs, output_grads):
        (sm,) = outputs
        (gz,) = output_grads
        d = gz * sm
        return [d - sm * sum(d, axis=self.axis, keepdims=True)]

    def pushforward(self, inputs, outputs, eval_points):
        if any(isinstance(t.type, DisconnectedType) for t in eval_points):
            return [disconnected_type()]
        return self.pullback(inputs, outputs, eval_points)


def softmax(c, axis=None):
    c = as_tensor_variable(c)
    if axis is not None:
        axis = normalize_axis_tuple(axis, c.type.ndim)
    return Softmax(axis=axis)(c)


class LogSoftmax(TensorSymbolicOp):
    r"""Log-softmax activation function.

    :math:`\log \sigma(\mathbf{x})_j = x_j - \log \sum_k e^{x_k}`

    Includes the numerical stabilization trick (subtracting the maximum).
    """

    __props__ = ("axis",)

    def __init__(self, *, axis, **kwargs):
        if isinstance(axis, (list, tuple)):
            axis = tuple(axis)
        elif isinstance(axis, int):
            axis = (axis,)
        self.axis = axis
        super().__init__(**kwargs)

    def build_inner_graph(self, x):
        x_stable = x - x.max(axis=self.axis, keepdims=True)
        return [x_stable - log(exp(x_stable).sum(axis=self.axis, keepdims=True))]

    def pullback(self, inputs, outputs, output_grads):
        (x,) = inputs
        (gz,) = output_grads
        sm = softmax(x, axis=self.axis)
        return [gz - sum(gz, axis=self.axis, keepdims=True) * sm]


def log_softmax(c, axis=None):
    c = as_tensor_variable(c)
    if axis is not None:
        axis = normalize_axis_tuple(axis, c.type.ndim)
    return LogSoftmax(axis=axis)(c)


@_vectorize_node.register(Softmax)
@_vectorize_node.register(LogSoftmax)
def vectorize_softmax_node(op, node, batched_x):
    core_ndim = node.inputs[0].type.ndim
    batch_ndim = batched_x.type.ndim - core_ndim

    if not batch_ndim:
        return [op(batched_x)]

    batch_axes = get_normalized_batch_axes(op.axis, core_ndim, batch_ndim)
    return [type(op)(axis=batch_axes)(batched_x)]


def poch(z, m):
    """
    Pochhammer symbol (rising factorial) function.

    """
    return gamma(z + m) / gamma(z)


def factorial(n):
    """
    Factorial function of a scalar or array of numbers.

    """
    return gamma(n + 1)


def logit(x):
    """
    Logit function.

    """
    return log(x / (1 - x))


def log_expit(x):
    """Logarithm of the logistic sigmoid.

    Matches :func:`scipy.special.log_expit`. Unlike ``log(expit(x))``, this does not
    underflow to ``-inf`` for large negative x.

    """
    return -softplus(-x)


def rgamma(x):
    """Reciprocal of the gamma function.

    Matches :func:`scipy.special.rgamma`, which is 0 at the poles of gamma.

    """
    x = as_tensor_variable(x)
    # gamma returns nan rather than inf at the non-positive integers, so 1 / gamma(x)
    # cannot pick up the 0 that the poles call for on its own.
    is_pole = le(x, 0) & eq(x, floor(x))
    return switch(is_pole, 0, 1 / gamma(x))


def chdtr(v, x):
    """Chi-square cumulative distribution function.

    Matches :func:`scipy.special.chdtr`.

    """
    return gammainc(v / 2, x / 2)


def chdtrc(v, x):
    """Chi-square survival function.

    Matches :func:`scipy.special.chdtrc`.

    """
    return gammaincc(v / 2, x / 2)


def chdtri(v, p):
    """Inverse of the chi-square survival function `chdtrc`.

    Matches :func:`scipy.special.chdtri`.

    """
    return 2 * gammainccinv(v / 2, p)


def gdtr(a, b, x):
    """Gamma cumulative distribution function.

    Matches :func:`scipy.special.gdtr`.

    """
    return gammainc(b, a * x)


def gdtrc(a, b, x):
    """Gamma survival function.

    Matches :func:`scipy.special.gdtrc`.

    """
    return gammaincc(b, a * x)


def pdtr(k, m):
    """Poisson cumulative distribution function.

    Matches :func:`scipy.special.pdtr`, including its truncation of ``k``.

    """
    return gammaincc(floor(k) + 1, m)


def pdtrc(k, m):
    """Poisson survival function.

    Matches :func:`scipy.special.pdtrc`, including its truncation of ``k``.

    """
    return gammainc(floor(k) + 1, m)


def _sqrt_2():
    return as_tensor_variable(np.sqrt(2.0), dtype=config.floatX)


class Ndtr(TensorSymbolicOp):
    """Op for `ndtr`."""

    def build_inner_graph(self, x):
        return [0.5 * erfc(-x / _sqrt_2())]


def ndtr(x):
    """Cumulative distribution function of the standard normal distribution.

    Computes the element-wise area under the standard normal density from -inf to x.

    Parameters
    ----------
    x : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the normal CDF evaluated at each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.special.ndtr(x))
    >>> f([-1, 0, 1])
    array([0.15865525, 0.5       , 0.84134475])

    Notes
    -----
    This function corresponds to SciPy's `scipy.special.ndtr` function. Prefer
    `log_ndtr` over ``log(ndtr(x))``, which underflows to ``-inf`` for x below about
    -37 and saturates to ``0`` for x above about 8.

    """
    return Ndtr()(x)


class LogNdtr(TensorSymbolicOp):
    """Op for `log_ndtr`."""

    def build_inner_graph(self, x):
        sqrt_2 = _sqrt_2()
        # For x < -1 the naive log(ndtr(x)) underflows to -inf once ndtr(x) drops below
        # the smallest float, so use erfc(z) = exp(-z**2) * erfcx(z), which keeps the
        # tail in log space. Above it log1p avoids the cancellation as ndtr(x) -> 1.
        return [
            switch(
                lt(x, -1.0),
                log(erfcx(-x / sqrt_2) / 2.0) - sqr(x) / 2.0,
                log1p(-erfc(x / sqrt_2) / 2.0),
            )
        ]


def log_ndtr(x):
    """Logarithm of the cumulative distribution function of the standard normal.

    Stays accurate in both tails, where the naive ``log(ndtr(x))`` does not.

    Parameters
    ----------
    x : TensorLike
        Input tensor

    Returns
    -------
    TensorVariable
        Output tensor with the log normal CDF evaluated at each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> x = pt.vector("x")
    >>> f = pytensor.function([x], pt.special.log_ndtr(x))
    >>> f([-1, 0, 1])
    array([-1.84102165, -0.69314718, -0.17275378])

    Where the naive form has lost the value entirely, this one has not:

    >>> f([-300.0, 10.0])
    array([-4.50066227e+04, -7.61985302e-24])

    Notes
    -----
    This function corresponds to SciPy's `scipy.special.log_ndtr` function.

    """
    return LogNdtr()(x)


def ndtri(p):
    """Quantile function of the standard normal distribution.

    Inverse of :func:`ndtr`: returns the x for which ``ndtr(x) == p``.

    Parameters
    ----------
    p : TensorLike
        Input tensor, with values in [0, 1]

    Returns
    -------
    TensorVariable
        Output tensor with the normal quantile evaluated at each element

    Examples
    --------
    >>> import pytensor
    >>> import pytensor.tensor as pt
    >>> p = pt.vector("p")
    >>> f = pytensor.function([p], pt.special.ndtri(p))
    >>> f([0.1, 0.5, 0.9])
    array([-1.28155157,  0.        ,  1.28155157])

    Notes
    -----
    This function corresponds to SciPy's `scipy.special.ndtri` function.
    """
    p = as_tensor_variable(p)
    return -_sqrt_2() * erfcinv(2 * p)


def beta(a, b):
    """
    Beta function.

    """
    return (gamma(a) * gamma(b)) / gamma(a + b)


def betaln(a, b):
    """
    Log beta function.

    """
    return gammaln(a) + gammaln(b) - gammaln(a + b)


class XLogY(TensorSymbolicOp):
    """Compute x * log(y), returning 0 when x = 0.

    Matches :func:`scipy.special.xlogy`. The gradient is not masked at x=0,
    matching the mathematically correct result (``-inf`` when y=0).
    """

    # Inlined late (at specialize) by `late_inline_OpFromGraph` so the inner
    # `x * log(y)` is hidden from canonicalize/stabilize rewrites that are
    # unsafe at infinity (e.g. `local_greedy_distributor` turns
    # `(a-1)*log(y)` into `a*log(y) - log(y)`, which yields nan when log(y) is
    # -inf at the boundary). After stabilize the body is exposed for fusion.
    inline = False

    def build_inner_graph(self, x, y):
        return [switch(eq(x, 0), 0, mul(x, log(y)))]

    def pullback(self, inputs, outputs, output_grads):
        x, y = inputs
        (gz,) = output_grads
        return [gz * log(y), gz * x / y]


_xlogy = XLogY()


def xlogy(x, y):
    """Compute x * log(y), returning 0 when x = 0.

    Matches :func:`scipy.special.xlogy`.

    Parameters
    ----------
    x : array_like
    y : array_like

    """
    return _xlogy(x, y)


class XLog1PY(TensorSymbolicOp):
    """Compute x * log(1 + y), returning 0 when x = 0.

    Matches :func:`scipy.special.xlog1py`. The gradient is not masked at x=0,
    matching the mathematically correct result.
    """

    # See note on `XLogY.inline`. Same hazard at y = -1 where log1p(y) = -inf.
    inline = False

    def build_inner_graph(self, x, y):
        return [switch(eq(x, 0), 0, mul(x, log1p(y)))]

    def pullback(self, inputs, outputs, output_grads):
        x, y = inputs
        (gz,) = output_grads
        return [gz * log1p(y), gz * x / (1 + y)]


_xlog1py = XLog1PY()


def xlog1py(x, y):
    """Compute x * log(1 + y), returning 0 when x = 0.

    Matches :func:`scipy.special.xlog1py`.

    Parameters
    ----------
    x : array_like
    y : array_like

    """
    return _xlog1py(x, y)


__all__ = [
    "beta",
    "betaln",
    "chdtr",
    "chdtrc",
    "chdtri",
    "factorial",
    "gdtr",
    "gdtrc",
    "log_expit",
    "log_ndtr",
    "log_softmax",
    "logit",
    "ndtr",
    "ndtri",
    "pdtr",
    "pdtrc",
    "poch",
    "rgamma",
    "softmax",
    "xlog1py",
    "xlogy",
]
