import numpy as np

from pytensor.assumptions.alloc import _eye_is_square, alloc_has_zero_off_diagonal
from pytensor.assumptions.core import (
    TRIDIAGONAL,
    FactState,
    all_inputs_have_key,
    propagate_first,
    register_assumption,
    register_constant_inference,
    true_if,
)
from pytensor.assumptions.elemwise import elemwise_preserves_zero_pattern
from pytensor.assumptions.subtensor import subtensor_propagates_matrix_property
from pytensor.tensor.basic import Alloc, AllocDiag, Eye
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.subtensor import Subtensor
from pytensor.tensor.variable import TensorConstant


def _tridiagonal_from_constant(var: TensorConstant) -> FactState:
    """Recognize a literal :class:`TensorConstant` matrix as TRIDIAGONAL.

    The band scan is O(n^2), so the result is cached on ``var.tag.is_tridiagonal``.
    """
    cached: FactState | None = getattr(var.tag, "is_tridiagonal", None)
    if cached is not None:
        return cached

    data = np.asarray(var.data)
    if data.ndim < 2 or data.shape[-2] != data.shape[-1]:
        result = FactState.FALSE
    else:
        in_band = sum(
            np.count_nonzero(np.diagonal(data, offset=k, axis1=-2, axis2=-1))
            for k in (-1, 0, 1)
        )
        result = (
            FactState.TRUE if np.count_nonzero(data) == in_band else FactState.FALSE
        )

    var.tag.is_tridiagonal = result
    return result


register_constant_inference(TRIDIAGONAL, _tridiagonal_from_constant)


@register_assumption(TRIDIAGONAL, Eye)
def _eye_tridiagonal(key, op, feature, fgraph, node, input_states):
    # Eye is tridiagonal when it is square with its nonzero diagonal within one
    # of the main one. A larger offset is UNKNOWN, not FALSE: for small enough
    # matrices the offset diagonal is empty and the all-zero result is still banded
    n, m, k = node.inputs
    square = _eye_is_square(n, m)
    if square is not FactState.TRUE:
        return [square]
    if not isinstance(k, TensorConstant):
        return [FactState.UNKNOWN]
    return true_if(abs(k.data.item()) <= 1)


@register_assumption(TRIDIAGONAL, AllocDiag)
def _alloc_diag_tridiagonal(key, op, feature, fgraph, node, input_states):
    return true_if(abs(op.offset) <= 1)


@register_assumption(TRIDIAGONAL, Alloc)
def _alloc_tridiagonal(key, op, feature, fgraph, node, input_states):
    # The band covers the whole matrix when n <= 2, so a nonzero fill only
    # rules the property out for a statically larger matrix
    states = alloc_has_zero_off_diagonal(key, op, feature, fgraph, node, input_states)
    if states[0] is FactState.FALSE and node.outputs[0].type.shape[-1] <= 2:
        return [FactState.TRUE]
    return states


# Blocks sit on the diagonal, so block_diag preserves the bandwidth of its blocks
register_assumption(TRIDIAGONAL, BlockDiagonal)(all_inputs_have_key)
# The Cholesky factor of a tridiagonal matrix is bidiagonal
register_assumption(TRIDIAGONAL, Cholesky)(propagate_first)
register_assumption(TRIDIAGONAL, Subtensor)(subtensor_propagates_matrix_property)


@register_assumption(TRIDIAGONAL, DimShuffle)
def _dimshuffle(key, op, feature, fgraph, node, input_states):
    # The band is symmetric around the main diagonal, so the matrix transpose
    # preserves it; batch-axis shuffles are handled by the generic rule
    if op.is_left_expanded_matrix_transpose:
        return [input_states[0]]
    return [FactState.UNKNOWN]


@register_assumption(TRIDIAGONAL, Elemwise)
def _elemwise(key, op, feature, fgraph, node, input_states):
    return elemwise_preserves_zero_pattern(TRIDIAGONAL, op, feature, node, input_states)
