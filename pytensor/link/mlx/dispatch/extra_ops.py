import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.extra_ops import CumOp, Repeat


@mlx_funcify.register(CumOp)
def mlx_funcify_CumOp(op, **kwargs):
    axis = op.axis
    mode = op.mode

    def cumop(x, axis=axis, mode=mode):
        match mode:
            case "add":
                return mx.cumsum(x, axis=axis)
            case "mul":
                return mx.cumprod(x, axis=axis)
            case _:
                raise NotImplementedError(f"CumOp mode {mode} not implemented in MLX")

    return cumop


@mlx_funcify.register(Repeat)
def jax_funcify_Repeat(op, **kwargs):
    axis = op.axis

    def repeat(x, repeats, axis=axis):
        if not isinstance(repeats, int):
            raise NotImplementedError(
                "MLX repeat does not support sequence-valued repeat argument."
            )
        return mx.repeat(x, repeats, axis=axis)

    return repeat
