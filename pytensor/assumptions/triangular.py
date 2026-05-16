from pytensor.assumptions.alloc import alloc_of_zero
from pytensor.assumptions.core import (
    LOWER_TRIANGULAR,
    UPPER_TRIANGULAR,
    FactState,
    all_inputs_have_key,
    propagate_first,
    register_assumption,
    true_if,
)
from pytensor.assumptions.dimshuffle import left_expand_dims_propagates_matrix_property
from pytensor.assumptions.elemwise import elemwise_preserves_zero_pattern
from pytensor.assumptions.subtensor import subtensor_propagates_matrix_property
from pytensor.tensor.basic import Alloc, AllocDiag, Eye
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.lu import LU
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.schur import QZ, Schur
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.products import KroneckerProduct
from pytensor.tensor.math import Dot
from pytensor.tensor.subtensor import Subtensor
from pytensor.tensor.variable import TensorConstant


@register_assumption(LOWER_TRIANGULAR, Eye)
def _eye_lower(key, op, feature, fgraph, node, input_states):
    # Eye is lower triangular when its diagonal sits on or below the main one.
    k = node.inputs[2]
    if not isinstance(k, TensorConstant):
        return [FactState.UNKNOWN]
    return true_if(k.data.item() <= 0, else_false=True)


@register_assumption(UPPER_TRIANGULAR, Eye)
def _eye_upper(key, op, feature, fgraph, node, input_states):
    # Eye is upper triangular when its diagonal sits on or above the main one.
    k = node.inputs[2]
    if not isinstance(k, TensorConstant):
        return [FactState.UNKNOWN]
    return true_if(k.data.item() >= 0, else_false=True)


@register_assumption(LOWER_TRIANGULAR, AllocDiag)
def _alloc_diag_lower(key, op, feature, fgraph, node, input_states):
    # offset <= 0 places values on or below the main diagonal; a positive
    # offset puts them strictly above, so the result is not lower-triangular.
    return true_if(op.offset <= 0, else_false=True)


@register_assumption(UPPER_TRIANGULAR, AllocDiag)
def _alloc_diag_upper(key, op, feature, fgraph, node, input_states):
    # offset >= 0 places values on or above the main diagonal; a negative
    # offset puts them strictly below, so the result is not upper-triangular.
    return true_if(op.offset >= 0, else_false=True)


register_assumption(LOWER_TRIANGULAR, Alloc)(alloc_of_zero)
register_assumption(UPPER_TRIANGULAR, Alloc)(alloc_of_zero)
register_assumption(LOWER_TRIANGULAR, BlockDiagonal)(all_inputs_have_key)
register_assumption(UPPER_TRIANGULAR, BlockDiagonal)(all_inputs_have_key)
register_assumption(LOWER_TRIANGULAR, MatrixInverse)(propagate_first)
register_assumption(UPPER_TRIANGULAR, MatrixInverse)(propagate_first)
register_assumption(LOWER_TRIANGULAR, Dot)(all_inputs_have_key)
register_assumption(UPPER_TRIANGULAR, Dot)(all_inputs_have_key)
register_assumption(LOWER_TRIANGULAR, KroneckerProduct)(all_inputs_have_key)
register_assumption(UPPER_TRIANGULAR, KroneckerProduct)(all_inputs_have_key)
register_assumption(LOWER_TRIANGULAR, Subtensor)(subtensor_propagates_matrix_property)
register_assumption(UPPER_TRIANGULAR, Subtensor)(subtensor_propagates_matrix_property)


@register_assumption(LOWER_TRIANGULAR, Cholesky)
def _chol_lower(key, op, feature, fgraph, node, input_states):
    return true_if(op.lower)


@register_assumption(UPPER_TRIANGULAR, Cholesky)
def _chol_upper(key, op, feature, fgraph, node, input_states):
    return true_if(not op.lower)


@register_assumption(UPPER_TRIANGULAR, QR)
def _qr_upper(key, op, feature, fgraph, node, input_states):
    # R (output 1) is always upper triangular in full/economic modes
    n_out = len(node.outputs)
    if op.mode in ("full", "economic"):
        states = [FactState.UNKNOWN] * n_out
        states[1] = FactState.TRUE
        return states
    return [FactState.UNKNOWN] * n_out


@register_assumption(UPPER_TRIANGULAR, Schur)
def _schur_upper(key, op, feature, fgraph, node, input_states):
    # T (output 0) is upper triangular only for complex Schur
    # Real Schur gives quasi-upper-triangular (2x2 blocks)
    if op.output == "complex":
        return [FactState.TRUE, FactState.UNKNOWN]
    return [FactState.UNKNOWN, FactState.UNKNOWN]


@register_assumption(UPPER_TRIANGULAR, QZ)
def _qz_upper(key, op, feature, fgraph, node, input_states):
    # AA (output 0) and BB (output 1) are upper triangular
    n_out = len(node.outputs)
    states = [FactState.UNKNOWN] * n_out
    states[0] = FactState.TRUE
    states[1] = FactState.TRUE
    return states


@register_assumption(LOWER_TRIANGULAR, LU)
def _lu_lower(key, op, feature, fgraph, node, input_states):
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
def _lu_upper(key, op, feature, fgraph, node, input_states):
    n_out = len(node.outputs)
    states = [FactState.UNKNOWN] * n_out
    # U is always the last output
    states[-1] = FactState.TRUE
    return states


def _dimshuffle_left_expand_dims(key, op, feature, fgraph, node, input_states):
    """Triangularity survives left-expand-dims (batch broadcast).

    Matrix transpose swaps lower<->upper, so it is *not* propagated under
    the same key.
    """
    if input_states[0] is not FactState.TRUE:
        return [FactState.UNKNOWN]
    if left_expand_dims_propagates_matrix_property(op):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


register_assumption(LOWER_TRIANGULAR, DimShuffle)(_dimshuffle_left_expand_dims)
register_assumption(UPPER_TRIANGULAR, DimShuffle)(_dimshuffle_left_expand_dims)


@register_assumption(LOWER_TRIANGULAR, Elemwise)
def _elemwise_lower(key, op, feature, fgraph, node, input_states):
    return elemwise_preserves_zero_pattern(
        LOWER_TRIANGULAR, op, feature, node, input_states
    )


@register_assumption(UPPER_TRIANGULAR, Elemwise)
def _elemwise_upper(key, op, feature, fgraph, node, input_states):
    return elemwise_preserves_zero_pattern(
        UPPER_TRIANGULAR, op, feature, node, input_states
    )
