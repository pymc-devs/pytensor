import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.reshape import JoinDims, SplitDims
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
        # Wrap in an MLX array, like Shape, so downstream ops (e.g. Cast) get
        # an array rather than a bare Python int (#2096).
        return mx.array(x.shape[op.i], dtype=mx.int64)

    return shape_i


@mlx_funcify.register(Reshape)
def mlx_funcify_Reshape(op, **kwargs):
    def reshape(x, shp):
        return mx.reshape(x, shp)

    return reshape


@mlx_funcify.register(JoinDims)
def mlx_funcify_JoinDims(op, **kwargs):
    start = op.start_axis
    n = op.n_axes

    def join_dims(x):
        shape = x.shape
        return mx.reshape(x, (*shape[:start], -1, *shape[start + n :]))

    return join_dims


@mlx_funcify.register(SplitDims)
def mlx_funcify_SplitDims(op, **kwargs):
    axis = op.axis

    def split_dims(x, shape):
        split_sizes = tuple(int(s) for s in shape)
        return mx.reshape(x, (*x.shape[:axis], *split_sizes, *x.shape[axis + 1 :]))

    return split_dims
