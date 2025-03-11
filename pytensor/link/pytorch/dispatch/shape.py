import torch

from pytensor.graph.basic import Constant
from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape


@pytorch_funcify.register(Reshape)
def pytorch_funcify_Reshape(op, node, **kwargs):
    _, shape = node.inputs

    if isinstance(shape, Constant):
        constant_shape = tuple(int(dim) for dim in shape.data)

        def reshape_constant_shape(x, *_):
            return torch.reshape(x, constant_shape)

        return reshape_constant_shape

    else:

        def reshape(x, shape):
            return torch.reshape(x, tuple(shape))

        return reshape


@pytorch_funcify.register(Shape)
def pytorch_funcify_Shape(op, **kwargs):
    def shape(x):
        return torch.tensor(x.shape)

    return shape


@pytorch_funcify.register(Shape_i)
def pytorch_funcify_Shape_i(op, **kwargs):
    i = op.i

    def shape_i(x):
        return torch.tensor(x.shape[i])

    return shape_i


@pytorch_funcify.register(SpecifyShape)
def pytorch_funcify_SpecifyShape(op, node, **kwargs):
    def specifyshape(x, *shape):
        assert x.ndim == len(shape)
        # strict=False because asserted above
        for actual, expected in zip(x.shape, shape, strict=False):
            if expected is None:
                continue
            if actual != expected:
                raise ValueError(f"Invalid shape: Expected {shape} but got {x.shape}")
        return x

    return specifyshape
