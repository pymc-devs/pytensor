import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.pad import Pad


@mlx_funcify.register(Pad)
def mlx_funcify_pad(op, node, **kwargs):
    pad_mode = op.pad_mode
    match pad_mode:
        case "constant":
            _, _, constant_values = node.inputs
            if constant_values.ndim != 0:
                raise NotImplementedError(
                    "MLX's 'constant' mode only accepts a scalar constant_values, "
                    "not per-side tuples like NumPy/JAX."
                )

            def pad_fn(x, pad_width, constant_values):
                return mx.pad(
                    x, pad_width, mode="constant", constant_values=constant_values
                )

        case "edge":

            def pad_fn(x, pad_width):
                return mx.pad(x, pad_width, mode="edge")

        case _:
            raise NotImplementedError(
                f"MLX does not support pad mode '{pad_mode}'. "
                f"Supported modes are 'constant' and 'edge'."
            )

    return pad_fn
