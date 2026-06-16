from pytensor.assumptions.alloc import eye_identity_rule
from pytensor.assumptions.core import (
    ORTHOGONAL,
    FactState,
    all_inputs_have_key,
    propagate_first,
    register_assumption,
)
from pytensor.assumptions.dimshuffle import left_expand_dims_propagates_matrix_property
from pytensor.assumptions.subtensor import subtensor_propagates_matrix_property
from pytensor.tensor.basic import Eye
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.decomposition.eigen import Eigh
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.schur import QZ, Schur
from pytensor.tensor.linalg.decomposition.svd import SVD
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.math import Dot
from pytensor.tensor.subtensor import Subtensor


register_assumption(ORTHOGONAL, Eye)(eye_identity_rule)
register_assumption(ORTHOGONAL, MatrixInverse)(propagate_first)
register_assumption(ORTHOGONAL, MatrixPinv)(propagate_first)
register_assumption(ORTHOGONAL, BlockDiagonal)(all_inputs_have_key)
register_assumption(ORTHOGONAL, Subtensor)(subtensor_propagates_matrix_property)
# The orthogonal group is closed under matmul: (A B)ᵀ (A B) = Bᵀ AᵀA B = I.
register_assumption(ORTHOGONAL, Dot)(all_inputs_have_key)


@register_assumption(ORTHOGONAL, DimShuffle)
def _dimshuffle(key, op, feature, fgraph, node, input_states):
    """Orthogonality survives matrix transpose and left-expand-dims (batch broadcast)."""
    if input_states[0] is not FactState.TRUE:
        return [FactState.UNKNOWN]
    if op.is_matrix_transpose:
        return [FactState.TRUE]
    if left_expand_dims_propagates_matrix_property(op):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(ORTHOGONAL, Schur)
def _schur(key, op, feature, fgraph, node, input_states):
    # Schur: A = Z @ T @ Z.T -> outputs are (T, Z); Z is orthogonal
    return [FactState.UNKNOWN, FactState.TRUE]


@register_assumption(ORTHOGONAL, QZ)
def _qz(key, op, feature, fgraph, node, input_states):
    # QZ: outputs are (AA, BB, Q, Z) or (AA, BB, alpha, beta, Q, Z)
    n_out = len(node.outputs)
    states = [FactState.UNKNOWN] * n_out
    # Q and Z are always the last two outputs
    states[-2] = FactState.TRUE
    states[-1] = FactState.TRUE
    return states


@register_assumption(ORTHOGONAL, QR)
def _qr(key, op, feature, fgraph, node, input_states):
    # QR full mode: Q is square and orthogonal (output 0)
    # Economic/other modes: Q has orthonormal columns but isn't square
    n_out = len(node.outputs)
    if op.mode == "full":
        return [FactState.TRUE] + [FactState.UNKNOWN] * (n_out - 1)
    return [FactState.UNKNOWN] * n_out


@register_assumption(ORTHOGONAL, SVD)
def _svd(key, op, feature, fgraph, node, input_states):
    # SVD full_matrices: outputs are (U, S, V) where U and V are orthogonal
    # Reduced: U and V have orthonormal columns/rows but aren't square
    if not op.compute_uv:
        return [FactState.UNKNOWN]
    if op.full_matrices:
        return [FactState.TRUE, FactState.UNKNOWN, FactState.TRUE]
    return [FactState.UNKNOWN] * 3


@register_assumption(ORTHOGONAL, Eigh)
def _eigh(key, op, feature, fgraph, node, input_states):
    # Eigh: outputs are (w, v) where v is the orthogonal eigenvector matrix
    return [FactState.UNKNOWN, FactState.TRUE]
