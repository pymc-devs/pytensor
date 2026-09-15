from functools import reduce

from pytensor.gradient import DisconnectedType, disconnected_type
from pytensor.graph.replace import _vectorize_node
from pytensor.tensor import as_tensor_variable
from pytensor.tensor.basic import expand_dims
from pytensor.tensor.elemwise import get_normalized_batch_axes
from pytensor.tensor.math import (
    add,
    eq,
    exp,
    gamma,
    gammaln,
    isinf,
    log,
    log1p,
    maximum,
    mul,
    sum,
    switch,
)
from pytensor.tensor.symbolic import TensorSymbolicOp
from pytensor.tensor.utils import normalize_reduce_axis


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
    axis = normalize_reduce_axis(axis, c.type.ndim, normalize_none=True)
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
    axis = normalize_reduce_axis(axis, c.type.ndim, normalize_none=True)
    return LogSoftmax(axis=axis)(c)


class LogSumExp(TensorSymbolicOp):
    r"""Log of the sum of exponentials.

    :math:`\log \sum_k e^{x_k}`

    Includes the numerical stabilization trick (subtracting the maximum), so unlike a
    bare ``log(sum(exp(x)))`` the gradient taken from it is stable too.
    """

    __props__ = ("axis",)

    # See the note on `XLogY.inline`.
    inline = False

    def __init__(self, *, axis, **kwargs):
        self.axis = tuple(axis)
        super().__init__(**kwargs)

    def build_inner_graph(self, x):
        x_max = x.max(axis=self.axis, keepdims=True)
        # Do not offset when x_max = -inf, to avoid nan in the output
        x_max = switch(isinf(x_max), 0.0, x_max)
        out = log(exp(x - x_max).sum(axis=self.axis, keepdims=True)) + x_max
        return [out.squeeze(axis=self.axis)]


def logsumexp(x, axis=None, keepdims=False):
    """Compute the log of the sum of exponentials of input elements.

    See ``scipy.special.logsumexp``.

    Parameters
    ----------
    x : symbolic tensor
        Input

    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default axis is None,
        and all elements are summed.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the original array.

    Returns
    -------
    TensorVariable

    """
    x = as_tensor_variable(x)
    axis = normalize_reduce_axis(axis, x.type.ndim, normalize_none=True)
    out = LogSumExp(axis=axis)(x)
    return expand_dims(out, axis) if keepdims else out


class LogAddExp(TensorSymbolicOp):
    r"""Log of the sum of exponentials of separate (broadcastable) inputs.

    :math:`\log \sum_k e^{x_k}`, where the :math:`x_k` are the variadic inputs.

    Includes the numerical stabilization trick (subtracting the maximum), so unlike a
    bare ``log(exp(x) + exp(y))`` the gradient taken from it is stable too.
    """

    # See the note on `XLogY.inline`.
    inline = False

    def build_inner_graph(self, *xs):
        x_max = reduce(maximum, xs)
        # Do not offset when x_max = -inf, to avoid nan in the output
        x_max = switch(isinf(x_max), 0.0, x_max)
        return [log(add(*(exp(x - x_max) for x in xs))) + x_max]


def logaddexp(*xs):
    """Logarithm of the sum of exponentiations of the inputs.

    See ``numpy.logaddexp``.

    Parameters
    ----------
    xs : symbolic tensors
        Input

    Returns
    -------
    TensorVariable

    """
    return LogAddExp()(*xs)


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
    "factorial",
    "log_softmax",
    "logaddexp",
    "logit",
    "logsumexp",
    "poch",
    "softmax",
    "xlog1py",
    "xlogy",
]
