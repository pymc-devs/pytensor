import mlx.core as mx

from pytensor.graph.basic import Constant
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
        # Wrap in an MLX array, like Shape, so downstream ops (e.g. Cast) get
        # an array rather than a bare Python int (#2096).
        return mx.array(x.shape[op.i], dtype=mx.int64)

    return shape_i


SHAPE_NOT_COMPATIBLE = """MLX requires a concrete value for the `shape` argument of `mx.reshape`.

The linker typifies every input to `mx.array`, and `mx.compile` traces the graph
with static shapes, so a shape whose *values* are only known at runtime cannot be
read back. Use a constant shape, or one that PyTensor's shape inference can
resolve statically:

>>> import pytensor.tensor as pt
>>> x = pt.ones((6, 4))
>>> y = x.reshape((24,))          # constant
>>> mat = pt.matrix("mat", shape=(6, 4))
>>> y = mat.reshape(mat.shape)    # statically resolvable
"""


@mlx_funcify.register(Reshape)
def mlx_funcify_Reshape(op, node, **kwargs):
    # `mx.reshape` wants a Python sequence of ints, but the linker typifies the
    # shape input to `mx.array` and `mx.compile` forbids reading a traced array,
    # so the shape has to be resolved at funcify time (#2386).
    static_shape = node.outputs[0].type.shape
    shape_input = node.inputs[1]

    if not any(dim is None for dim in static_shape):
        # Shape inference already resolved every dimension, including any -1.
        target = tuple(static_shape)
    elif isinstance(shape_input, Constant):
        target = tuple(int(dim) for dim in shape_input.data)
    else:
        target = None

    if target is not None:

        def reshape(x, shp):
            return mx.reshape(x, target)

    else:

        def reshape(x, shp):
            if isinstance(shp, mx.array):
                try:
                    shp = shp.tolist()
                except ValueError as exc:
                    raise NotImplementedError(SHAPE_NOT_COMPATIBLE) from exc
            return mx.reshape(x, tuple(shp))

    return reshape
