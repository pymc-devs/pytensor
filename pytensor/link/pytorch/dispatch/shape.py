import torch

from pytensor.link.pytorch.dispatch.basic import pytorch_funcify
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape, Unbroadcast


@pytorch_funcify.register(Reshape)
def pytorch_funcify_Reshape(op, node, **kwargs):
    def reshape(x, shape):
        return torch.reshape(x, tuple(shape))

    return reshape


@pytorch_funcify.register(Shape)
def pytorch_funcify_Shape(op, **kwargs):
    def shape(x):
        return x.shape

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
        for actual, expected in zip(x.shape, shape, strict=True):
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
