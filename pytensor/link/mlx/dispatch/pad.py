import mlx.core as mx
import numpy as np

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.link.mlx.dispatch.tensor_basic import mlx_to_list_shape
from pytensor.tensor.pad import Pad


def _runtime_pad_width_pairs(pad_width, ndim):
    widths = np.array(mlx_to_list_shape(pad_width.flatten())).reshape(pad_width.shape)
    widths = np.broadcast_to(widths, (ndim, 2))
    return [tuple(pair) for pair in widths.tolist()]


@mlx_funcify.register(Pad)
def mlx_funcify_pad(op, node, **kwargs):
    pad_mode = op.pad_mode
    ndim = node.inputs[0].type.ndim
    static_pairs = None if op.static_pad_width is None else list(op.static_pad_width)

    def pad_width_pairs(pad_width):
        if static_pairs is not None:
            return static_pairs
        return _runtime_pad_width_pairs(pad_width, ndim)

    if pad_mode == "constant":
        _, _, constant_values = node.inputs
        if constant_values.ndim != 0:
            raise NotImplementedError(
                "MLX's 'constant' mode only accepts a scalar constant_values, "
                "not per-side tuples like NumPy/JAX."
            )

        def constant_pad_fn(x, pad_width, constant_values):
            return mx.pad(
                x,
                pad_width_pairs(pad_width),
                mode="constant",
                constant_values=constant_values,
            )

        return constant_pad_fn

    elif pad_mode == "edge":

        def edge_pad_fn(x, pad_width):
            return mx.pad(x, pad_width_pairs(pad_width), mode="edge")

        return edge_pad_fn

    else:
        raise NotImplementedError(
            f"MLX does not support pad mode '{pad_mode}'. "
            f"Supported modes are 'constant' and 'edge'."
        )
