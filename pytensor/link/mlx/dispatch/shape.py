import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape


@mlx_funcify.register(Shape)
def mlx_funcify_Shape(op, **kwargs):
    def shape(x):
        return mx.array(x.shape, dtype=mx.int64)

    return shape


@mlx_funcify.register(SpecifyShape)
def mlx_funcify_SpecifyShape(op, node, **kwargs):
    def specifyshape(x, *shape):
        assert x.ndim == len(shape)
        for actual, expected in zip(x.shape, shape, strict=True):
            if expected is None:
                continue
            if actual != expected:
                raise ValueError(f"Invalid shape: Expected {shape} but got {x.shape}")
        return x

    return specifyshape


@mlx_funcify.register(Shape_i)
def mlx_funcify_Shape_i(op, node, **kwargs):
    def shape_i(x):
        return x.shape[op.i]

    return shape_i


@mlx_funcify.register(Reshape)
def mlx_funcify_Reshape(op, **kwargs):
    def reshape(x, shp):
        return mx.reshape(x, shp)

    return reshape
