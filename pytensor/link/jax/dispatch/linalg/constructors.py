import jax

from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor._linalg.constructors import BlockDiagonal


@jax_funcify.register(BlockDiagonal)
def jax_funcify_BlockDiagonalMatrix(op, **kwargs):
    def block_diag(*inputs):
        return jax.scipy.linalg.block_diag(*inputs)

    return block_diag
