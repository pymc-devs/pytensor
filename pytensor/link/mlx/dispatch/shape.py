from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.shape import Shape_i, SpecifyShape


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
