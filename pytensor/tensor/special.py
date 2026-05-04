from numpy.lib.array_utils import normalize_axis_tuple

from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph.replace import _vectorize_node
from pytensor.tensor.elemwise import get_normalized_batch_axes
from pytensor.tensor.math import (
    eq,
    exp,
    gamma,
    gammaln,
    log,
    log1p,
    mul,
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

    inline = True

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

    inline = True

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
    "factorial",
    "log_softmax",
    "logit",
    "poch",
    "softmax",
    "xlog1py",
    "xlogy",
]
