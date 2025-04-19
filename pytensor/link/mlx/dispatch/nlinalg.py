import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.nlinalg import (
    SVD,
)


@mlx_funcify.register(SVD)
def mlx_funcify_SVD(op, node, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv
    otype = (
        getattr(mx, node.outputs[0].dtype)
        if not compute_uv
        else [getattr(mx, output.dtype) for output in node.outputs]
    )

    if not full_matrices:
        raise TypeError("full_matrices=False is not supported in the mlx backend.")

    def svd_S_only(x):
        return mx.linalg.svd(x, compute_uv=False, stream=mx.cpu).astype(
            otype, stream=mx.cpu
        )

    def svd_full(x):
        outputs = mx.linalg.svd(x, compute_uv=True, stream=mx.cpu)
        return tuple(
            output.astype(typ, stream=mx.cpu)
            for output, typ in zip(outputs, otype, strict=True)
        )

    if compute_uv:
        return svd_full
    else:
        return svd_S_only
