import numpy as np

from pytensor import scalar as ps
from pytensor.tensor.elemwise import Elemwise


class XlogX(ps.UnaryScalarOp):
    """
    Compute X * log(X), with special case 0 log(0) = 0.

    """

    def impl(self, x):
        if x == 0.0:
            return 0.0
        return x * np.log(x)

    def grad(self, inputs, grads):
        (x,) = inputs
        (gz,) = grads
        return [gz * (1 + ps.log(x))]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in [ps.float32, ps.float64]:
            return f"""{z} =
                {x} == 0.0
                ? 0.0
                : {x} * log({x});"""
        raise NotImplementedError("only floatingpoint is implemented")


scalar_xlogx = XlogX(ps.upgrade_to_float)
xlogx = Elemwise(scalar_xlogx, name="xlogx")


class XlogY0(ps.BinaryScalarOp):
    """
    Compute X * log(Y), with special case 0 log(0) = 0.

    """

    def impl(self, x, y):
        if x == 0.0:
            return 0.0
        return x * np.log(y)

    def grad(self, inputs, grads):
        x, y = inputs
        (gz,) = grads
        return [gz * ps.log(y), gz * x / y]

    def c_code(self, node, name, inputs, outputs, sub):
        x, y = inputs
        (z,) = outputs
        if node.inputs[0].type in [ps.float32, ps.float64]:
            return f"""{z} =
                {x} == 0.0
                ? 0.0
                : {x} * log({y});"""
        raise NotImplementedError("only floatingpoint is implemented")


scalar_xlogy0 = XlogY0(ps.upgrade_to_float)
xlogy0 = Elemwise(scalar_xlogy0, name="xlogy0")
