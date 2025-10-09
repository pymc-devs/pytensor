import mlx.core as mx

from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.tensor.math import Argmax, Dot, Max


@mlx_funcify.register(Dot)
def mlx_funcify_Dot(op, node=None, **kwargs):
    def dot(x, y):
        return mx.matmul(x, y)

    return dot


@mlx_funcify.register(Max)
def mlx_funcify_Max(op, node=None, **kwargs):
    def max_fn(x):
        axes = op.axis
        if axes is None:
            reduce_axes = None
        else:
            reduce_axes = tuple(int(ax) for ax in axes)

        keepdims = getattr(op, "keepdims", False)

        return mx.max(x, axis=reduce_axes, keepdims=keepdims)

    return max_fn


@mlx_funcify.register(Argmax)
def mlx_funcify_Argmax(op, node=None, **kwargs):
    axis = op.axis

    def argmax_fn(x):
        if axis is None:
            axes = tuple(range(x.ndim))
        else:
            axes = tuple(int(ax) for ax in axis)

        keep_axes = [i for i in range(x.ndim) if i not in axes]
        transposed_x = mx.transpose(x, tuple(keep_axes + list(axes)))

        kept_shape = transposed_x.shape[: len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes) :]

        flat_size = 1
        for dim in reduced_shape:
            flat_size *= int(dim)
        reshaped_x = transposed_x.reshape((*kept_shape, flat_size))

        max_idx = mx.argmax(reshaped_x, axis=-1)

        result = max_idx.astype(mx.int64)

        if getattr(op, "keepdims", False):
            reshape_shape = []
            keep_iter = iter(kept_shape)
            axis_iter = iter(sorted(axes))
            next_axis = next(axis_iter, None)
            for dim_idx in range(x.ndim):
                if next_axis is not None and dim_idx == next_axis:
                    reshape_shape.append(1)
                    next_axis = next(axis_iter, None)
                else:
                    reshape_shape.append(int(next(keep_iter)))

            return result.reshape(tuple(reshape_shape))

        return result

    return argmax_fn
