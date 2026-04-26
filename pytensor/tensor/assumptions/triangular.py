from pytensor.tensor.assumptions._elemwise import elemwise_preserves_zero_pattern
from pytensor.tensor.assumptions.core import (
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.assumptions.utils import (
    all_inputs_have_key,
    alloc_of_zero,
    eye_is_identity,
    propagate_first,
    true_if,
)
from pytensor.tensor.basic import Alloc, AllocDiag, Eye
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.lu import LU
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.schur import QZ, Schur
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.math import Dot


LOWER_TRIANGULAR = AssumptionKey("lower_triangular")
UPPER_TRIANGULAR = AssumptionKey("upper_triangular")


@register_assumption(LOWER_TRIANGULAR, Eye)
def _eye_lower(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(UPPER_TRIANGULAR, Eye)
def _eye_upper(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(LOWER_TRIANGULAR, AllocDiag)
def _alloc_diag_lower(op, feature, fgraph, node, input_states):
    return true_if(op.offset == 0)


@register_assumption(UPPER_TRIANGULAR, AllocDiag)
def _alloc_diag_upper(op, feature, fgraph, node, input_states):
    return true_if(op.offset == 0)


register_assumption(LOWER_TRIANGULAR, Alloc)(alloc_of_zero)
register_assumption(UPPER_TRIANGULAR, Alloc)(alloc_of_zero)
register_assumption(LOWER_TRIANGULAR, BlockDiagonal)(all_inputs_have_key)
register_assumption(UPPER_TRIANGULAR, BlockDiagonal)(all_inputs_have_key)
register_assumption(LOWER_TRIANGULAR, MatrixInverse)(propagate_first)
register_assumption(UPPER_TRIANGULAR, MatrixInverse)(propagate_first)
register_assumption(LOWER_TRIANGULAR, Dot)(all_inputs_have_key)
register_assumption(UPPER_TRIANGULAR, Dot)(all_inputs_have_key)


@register_assumption(LOWER_TRIANGULAR, Cholesky)
def _chol_lower(op, feature, fgraph, node, input_states):
    return true_if(op.lower)


@register_assumption(UPPER_TRIANGULAR, Cholesky)
def _chol_upper(op, feature, fgraph, node, input_states):
    return true_if(not op.lower)


@register_assumption(UPPER_TRIANGULAR, QR)
def _qr_upper(op, feature, fgraph, node, input_states):
    # R (output 1) is always upper triangular in full/economic modes
    n_out = len(node.outputs)
    if op.mode in ("full", "economic"):
        states = [FactState.UNKNOWN] * n_out
        states[1] = FactState.TRUE
        return states
    return [FactState.UNKNOWN] * n_out


@register_assumption(UPPER_TRIANGULAR, Schur)
def _schur_upper(op, feature, fgraph, node, input_states):
    # T (output 0) is upper triangular only for complex Schur
    # Real Schur gives quasi-upper-triangular (2x2 blocks)
    if op.output == "complex":
        return [FactState.TRUE, FactState.UNKNOWN]
    return [FactState.UNKNOWN, FactState.UNKNOWN]


@register_assumption(UPPER_TRIANGULAR, QZ)
def _qz_upper(op, feature, fgraph, node, input_states):
    # AA (output 0) and BB (output 1) are upper triangular
    n_out = len(node.outputs)
    states = [FactState.UNKNOWN] * n_out
    states[0] = FactState.TRUE
    states[1] = FactState.TRUE
    return states


@register_assumption(LOWER_TRIANGULAR, LU)
def _lu_lower(op, feature, fgraph, node, input_states):
    n_out = len(node.outputs)
    states = [FactState.UNKNOWN] * n_out
    if op.permute_l:
        # outputs are (PL, U) — PL is not lower triangular
        return states
    elif op.p_indices:
        # outputs are (p_indices, L, U) — L is output 1
        states[1] = FactState.TRUE
    else:
        # outputs are (P, L, U) — L is output 1
        states[1] = FactState.TRUE
    return states


@register_assumption(UPPER_TRIANGULAR, LU)
def _lu_upper(op, feature, fgraph, node, input_states):
    n_out = len(node.outputs)
    states = [FactState.UNKNOWN] * n_out
    # U is always the last output
    states[-1] = FactState.TRUE
    return states


@register_assumption(LOWER_TRIANGULAR, Elemwise)
def _elemwise_lower(op, feature, fgraph, node, input_states):
    return elemwise_preserves_zero_pattern(
        LOWER_TRIANGULAR, op, feature, node, input_states
    )


@register_assumption(UPPER_TRIANGULAR, Elemwise)
def _elemwise_upper(op, feature, fgraph, node, input_states):
    return elemwise_preserves_zero_pattern(
        UPPER_TRIANGULAR, op, feature, node, input_states
    )
