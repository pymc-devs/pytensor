from pytensor.tensor.assumptions.core import (
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.assumptions.utils import eye_is_identity, true_if
from pytensor.tensor.basic import Eye
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.schur import QZ, Schur
from pytensor.tensor.linalg.decomposition.svd import SVD
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv


ORTHOGONAL = AssumptionKey("orthogonal")


@register_assumption(ORTHOGONAL, Eye)
def _eye(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(ORTHOGONAL, MatrixInverse)
@register_assumption(ORTHOGONAL, MatrixPinv)
def _inv(op, feature, fgraph, node, input_states):
    return true_if(feature.check(node.inputs[0], ORTHOGONAL))


@register_assumption(ORTHOGONAL, DimShuffle)
def _transpose(op, feature, fgraph, node, input_states):
    if op.is_matrix_transpose:
        return true_if(input_states[0])
    return [FactState.UNKNOWN]


@register_assumption(ORTHOGONAL, Schur)
def _schur(op, feature, fgraph, node, input_states):
    # Schur: A = Z @ T @ Z.T -> outputs are (T, Z); Z is orthogonal
    return [FactState.UNKNOWN, FactState.TRUE]


@register_assumption(ORTHOGONAL, QZ)
def _qz(op, feature, fgraph, node, input_states):
    # QZ: outputs are (AA, BB, Q, Z) or (AA, BB, alpha, beta, Q, Z)
    n_out = len(node.outputs)
    states = [FactState.UNKNOWN] * n_out
    # Q and Z are always the last two outputs
    states[-2] = FactState.TRUE
    states[-1] = FactState.TRUE
    return states


@register_assumption(ORTHOGONAL, QR)
def _qr(op, feature, fgraph, node, input_states):
    # QR full mode: Q is square and orthogonal (output 0)
    # Economic/other modes: Q has orthonormal columns but isn't square
    n_out = len(node.outputs)
    if op.mode == "full":
        return [FactState.TRUE] + [FactState.UNKNOWN] * (n_out - 1)
    return [FactState.UNKNOWN] * n_out


@register_assumption(ORTHOGONAL, SVD)
def _svd(op, feature, fgraph, node, input_states):
    # SVD full_matrices: outputs are (U, S, V) where U and V are orthogonal
    # Reduced: U and V have orthonormal columns/rows but aren't square
    if not op.compute_uv:
        return [FactState.UNKNOWN]
    if op.full_matrices:
        return [FactState.TRUE, FactState.UNKNOWN, FactState.TRUE]
    return [FactState.UNKNOWN] * 3
