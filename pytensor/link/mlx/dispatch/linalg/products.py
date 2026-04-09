import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor._linalg.products import KroneckerProduct


@mlx_funcify.register(KroneckerProduct)
def mlx_funcify_KroneckerProduct(op, node, **kwargs):
    otype = node.outputs[0].dtype
    stream = mx.cpu if otype == "float64" else mx.gpu

    A_dtype = getattr(mx, node.inputs[0].dtype)
    B_dtype = getattr(mx, node.inputs[1].dtype)

    def kron(a, b):
        return mx.kron(
            a.astype(dtype=A_dtype, stream=stream),
            b.astype(dtype=B_dtype, stream=stream),
            stream=stream,
        )

    return kron
