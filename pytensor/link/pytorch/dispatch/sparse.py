import torch
from scipy.sparse import spmatrix

from pytensor.graph.basic import Constant
from pytensor.link.pytorch.dispatch import pytorch_funcify, pytorch_typify
from pytensor.sparse.basic import Dot, StructuredDot
from pytensor.sparse.type import SparseTensorType


@pytorch_typify.register(spmatrix)
def pytorch_typify_spmatrix(matrix, dtype=None, **kwargs):
    # Note: This changes the type of the constants from CSR/CSC to COO
    # We could add COO as a PyTensor type but this would only be useful for PyTorch graphs
    # and it would break the premise of one graph -> multiple backends.
    # The same situation happens with RandomGenerators...
    return torch.sparse_coo_tensor(matrix.indices, matrix.data, matrix.shape)


@pytorch_funcify.register(Dot)
@pytorch_funcify.register(StructuredDot)
def pytorch_funcify_sparse_dot(op, node, **kwargs):
    for input in node.inputs:
        if isinstance(input.type, SparseTensorType) and not isinstance(input, Constant):
            raise NotImplementedError(
                "PyTorch sparse dot only implemented for constant sparse inputs"
            )

    if isinstance(node.outputs[0].type, SparseTensorType):
        raise NotImplementedError("PyTorch sparse dot only implemented for dense outputs")

    def sparse_dot(x, y):
        out = torch.sparse.mm(x, y)
        if out.is_sparse:
            out = out.to_dense()
        return out

    return sparse_dot