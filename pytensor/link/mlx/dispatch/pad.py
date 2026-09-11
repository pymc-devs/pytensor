import mlx.core as mx

from pytensor.graph.basic import Constant
from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.pad import Pad


PAD_WIDTH_NOT_COMPATIBLE = """MLX requires a concrete value for `pad_width`.

The linker typifies every input to `mx.array`, but `mx.pad` takes an int or a
list of (before, after) int pairs, and `mx.compile` forbids reading a traced
array. Use a constant `pad_width`.
"""


def _resolve_pad_width(pad_width_input):
    """Return `pad_width` as Python ints at funcify time, or None if not static."""
    if isinstance(pad_width_input, Constant):
        value = pad_width_input.data.tolist()
        # A 2-d (n, 2) spec has to be a list of pairs; `mx.pad` rejects a
        # nested list of lists in some versions, and an int stays an int.
        if isinstance(value, list) and value and isinstance(value[0], list):
            return [tuple(pair) for pair in value]
        return value
    return None


def _pad_width_at_runtime(pad_width):
    if isinstance(pad_width, mx.array):
        try:
            pad_width = pad_width.tolist()
        except ValueError as exc:
            raise NotImplementedError(PAD_WIDTH_NOT_COMPATIBLE) from exc
    if isinstance(pad_width, list) and pad_width and isinstance(pad_width[0], list):
        return [tuple(pair) for pair in pad_width]
    return pad_width


@mlx_funcify.register(Pad)
def mlx_funcify_pad(op, node, **kwargs):
    pad_mode = op.pad_mode

    if pad_mode == "constant":
        _, _, constant_values = node.inputs
        if constant_values.ndim != 0:
            raise NotImplementedError(
                "MLX's 'constant' mode only accepts a scalar constant_values, "
                "not per-side tuples like NumPy/JAX."
            )

        static_pad_width = _resolve_pad_width(node.inputs[1])

        def constant_pad_fn(x, pad_width, constant_values):
            # `mx.pad` needs Python ints; the linker hands us an `mx.array`.
            width = (
                static_pad_width
                if static_pad_width is not None
                else _pad_width_at_runtime(pad_width)
            )
            return mx.pad(x, width, mode="constant", constant_values=constant_values)

        return constant_pad_fn

    elif pad_mode == "edge":
        static_pad_width = _resolve_pad_width(node.inputs[1])

        def edge_pad_fn(x, pad_width):
            width = (
                static_pad_width
                if static_pad_width is not None
                else _pad_width_at_runtime(pad_width)
            )
            return mx.pad(x, width, mode="edge")

        return edge_pad_fn

    else:
        raise NotImplementedError(
            f"MLX does not support pad mode '{pad_mode}'. "
            f"Supported modes are 'constant' and 'edge'."
        )
