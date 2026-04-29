import scipy.signal as scipy_signal

from pytensor.graph.basic import Apply
from pytensor.tensor import Op, as_tensor_variable
from pytensor.tensor.type import TensorType


class GaussSpline(Op):
    __props__ = ("n",)

    def __init__(self, n: int):
        self.n = n

    def make_node(self, knots):
        knots = as_tensor_variable(knots)
        if not isinstance(knots.type, TensorType):
            raise TypeError("Input must be a TensorType")

        if not isinstance(self.n, int) or self.n is None or self.n < 0:
            raise ValueError("n must be a non-negative integer")

        if knots.ndim < 1:
            raise TypeError("Input must be at least 1-dimensional")

        out = knots.type()
        return Apply(self, [knots], [out])

    def perform(self, node, inputs, output_storage):
        [x] = inputs
        [out] = output_storage
        out[0] = scipy_signal.gauss_spline(x, self.n)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


def gauss_spline(x, n):
    return GaussSpline(n)(x)
