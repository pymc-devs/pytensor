import torch

from pytensor.link.pytorch.dispatch import pytorch_funcify
from pytensor.tensor.linalg.constructors import BlockDiagonal


@pytorch_funcify.register(BlockDiagonal)
def pytorch_funcify_BlockDiagonal(op, node, **kwargs):
    out_dtype = getattr(torch, node.outputs[0].dtype)

    def block_diag(*matrices):
        return torch.block_diag(*(matrix.to(out_dtype) for matrix in matrices))

    return block_diag
