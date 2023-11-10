import torch

from pytensor.graph import Constant
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape, Unbroadcast
from pytensor.tensor.type import TensorType


class PyTorchShapeTuple(Op):
    """Dummy Op that represents a `size` specified as a tuple."""

    def make_node(self, *inputs):
        dtype = inputs[0].type.dtype
        otype = TensorType(dtype, shape=(len(inputs),))
        return Apply(self, inputs, [otype()])

    def perform(self, *inputs):
        return tuple(inputs)


@pytorch_funcify.register(PyTorchShapeTuple)
def pytorch_funcify_PyTorchShapeTuple(op, **kwargs):
    def shape_tuple_fn(*x):
        return tuple(x)

    return shape_tuple_fn


SHAPE_NOT_COMPATIBLE = """PyTorch requires concrete values for the `shape` parameter of `torch.reshape`.
Concrete values are either constants:

>>> import pytensor.tensor as at
>>> x = at.ones(6)
>>> y = x.reshape((2, 3))

Or the shape of an array:

>>> mat = at.matrix('mat')
>>> y = x.reshape(mat.shape)
"""


def assert_shape_argument_pytorch_compatible(shape):
    """Assert whether the current node can be JIT-compiled by PyTorch.

    PyTorch can JIT-compile functions with a `shape` or `size` argument if it is
    given a concrete value, i.e. either a constant or the shape of any traced
    value.

    """
    shape_op = shape.owner.op
    if not isinstance(shape_op, (Shape, Shape_i, PyTorchShapeTuple)):
        raise NotImplementedError(SHAPE_NOT_COMPATIBLE)


@pytorch_funcify.register(Reshape)
def pytorch_funcify_Reshape(op, node, **kwargs):
    shape = node.inputs[1]

    if isinstance(shape, Constant):
        constant_shape = shape.data

        def reshape(x, shape):
            return torch.reshape(x, constant_shape)

    else:
        assert_shape_argument_pytorch_compatible(shape)

        def reshape(x, shape):
            return torch.reshape(x, shape)

    return reshape


@pytorch_funcify.register(Shape)
def pytorch_funcify_Shape(op, **kwargs):
    def shape(x):
        return torch.shape(x)

    return shape


@pytorch_funcify.register(Shape_i)
def pytorch_funcify_Shape_i(op, **kwargs):
    i = op.i

    def shape_i(x):
        return torch.shape(x)[i]

    return shape_i


@pytorch_funcify.register(SpecifyShape)
def pytorch_funcify_SpecifyShape(op, node, **kwargs):
    def specifyshape(x, *shape):
        assert x.ndim == len(shape)
        for actual, expected in zip(x.shape, shape):
            if expected is None:
                continue
            if actual != expected:
                raise ValueError(f"Invalid shape: Expected {shape} but got {x.shape}")
        return x

    return specifyshape


@pytorch_funcify.register(Unbroadcast)
def pytorch_funcify_Unbroadcast(op, **kwargs):
    def unbroadcast(x):
        return x

    return unbroadcast