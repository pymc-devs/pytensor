import mlx.core as mx

from pytensor.link.mlx.dispatch.basic import mlx_funcify
from pytensor.tensor.linalg.constructors import BlockDiagonal


@mlx_funcify.register(BlockDiagonal)
def mlx_funcify_BlockDiagonal(op, node, **kwargs):
    out_dtype = node.outputs[0].dtype
    stream = mx.cpu if out_dtype == "float64" else mx.gpu
    mx_dtype = getattr(mx, out_dtype)

    def block_diag(*matrices):
        n_cols = sum(matrix.shape[-1] for matrix in matrices)
        rows = []
        col = 0
        for matrix in matrices:
            width = matrix.shape[-1]
            rows.append(
                mx.pad(
                    matrix.astype(mx_dtype, stream=stream),
                    [(0, 0), (col, n_cols - col - width)],
                    stream=stream,
                )
            )
            col += width
        return mx.concatenate(rows, axis=0, stream=stream)

    return block_diag
