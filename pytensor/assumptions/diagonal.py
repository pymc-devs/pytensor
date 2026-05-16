import numpy as np

from pytensor.assumptions.alloc import (
    alloc_diag_at_offset_zero,
    alloc_of_zero,
    eye_identity_rule,
)
from pytensor.assumptions.core import (
    DIAGONAL,
    ORTHOGONAL,
    FactState,
    all_inputs_have_key,
    propagate_first,
    register_assumption,
    register_constant_inference,
)
from pytensor.assumptions.elemwise import elemwise_preserves_zero_pattern
from pytensor.assumptions.subtensor import subtensor_propagates_matrix_property
from pytensor.tensor.basic import Alloc, AllocDiag, Eye
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.linalg.products import KroneckerProduct
from pytensor.tensor.math import Dot
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    IncSubtensor,
    Subtensor,
)
from pytensor.tensor.variable import TensorConstant


def _diagonal_from_constant(var: TensorConstant) -> FactState:
    """Recognize a literal :class:`TensorConstant` matrix as DIAGONAL by inspecting its data.

    The off-diagonal scan is O(n^2), so the result is cached on ``var.tag.is_diagonal``.
    """
    cached: FactState | None = getattr(var.tag, "is_diagonal", None)
    if cached is not None:
        return cached

    data = np.asarray(var.data)
    if data.ndim < 2:
        # A scalar or vector is not a matrix, so it cannot be a diagonal matrix.
        result = FactState.FALSE
    else:
        m, n = data.shape[-2], data.shape[-1]
        if m != n:
            result = FactState.FALSE
        else:
            eye_mask = np.eye(n, dtype=bool)
            result = FactState.FALSE if np.any(data * ~eye_mask) else FactState.TRUE

    var.tag.is_diagonal = result
    return result


register_constant_inference(DIAGONAL, _diagonal_from_constant)


def indexes_diagonal(node) -> bool:
    """True when an ``*IncSubtensor*`` node modifies only diagonal entries."""
    op = node.op
    if not isinstance(op, AdvancedIncSubtensor | IncSubtensor):
        return False

    # A full slice (``[:]``) over a batch axis is a no-op; drop it so the
    # batched ``x[:, i, i]`` form is recognised, not just bare ``x[i, i]``.
    idx_list = [p for p in op.idx_list if p != slice(None)]
    if len(idx_list) < 2 or any(isinstance(p, slice) for p in idx_list):
        return False

    op_indices = node.inputs[2:]
    first = op_indices[idx_list[0]]
    return all(first is op_indices[p] for p in idx_list[1:])


register_assumption(DIAGONAL, Eye)(eye_identity_rule)
register_assumption(DIAGONAL, AllocDiag)(alloc_diag_at_offset_zero)
register_assumption(DIAGONAL, Alloc)(alloc_of_zero)
register_assumption(DIAGONAL, Cholesky)(propagate_first)
register_assumption(DIAGONAL, BlockDiagonal)(all_inputs_have_key)
register_assumption(DIAGONAL, MatrixInverse)(propagate_first)
register_assumption(DIAGONAL, MatrixPinv)(propagate_first)
register_assumption(DIAGONAL, KroneckerProduct)(all_inputs_have_key)
register_assumption(DIAGONAL, Dot)(all_inputs_have_key)
register_assumption(DIAGONAL, Subtensor)(subtensor_propagates_matrix_property)


@register_assumption(DIAGONAL, Dot)
def _dot_orthogonal_xxt(key, op, feature, fgraph, node, input_states):
    """x @ x.T is diagonal (identity) when x is orthogonal."""
    a, b = node.inputs
    if (
        feature.check(a, ORTHOGONAL)
        and b.owner is not None
        and isinstance(b.owner_op, DimShuffle)
        and b.owner_op.is_matrix_transpose
        and b.owner.inputs[0] is a
    ):
        return [FactState.TRUE]

    return [FactState.UNKNOWN]


@register_assumption(DIAGONAL, IncSubtensor)
@register_assumption(DIAGONAL, AdvancedIncSubtensor)
def _inc_subtensor(key, op, feature, fgraph, node, input_states):
    if input_states[0] is FactState.TRUE and indexes_diagonal(node):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(DIAGONAL, DimShuffle)
def _dimshuffle(key, op, feature, fgraph, node, input_states):
    # Matrix transpose and left-expand-dims are exact diagonality bijections\
    if op.is_transpose:
        nd = op.input_ndim
        if nd >= 2:
            last_two_swapped = (*tuple(range(nd - 2)), nd - 1, nd - 2)
            if op.new_order == last_two_swapped:
                return [input_states[0]]

    if op.is_expand_dims and op.input_ndim >= 2:
        if op.new_order[-op.input_ndim :] == tuple(range(op.input_ndim)):
            return [input_states[0]]

    return [FactState.UNKNOWN]


@register_assumption(DIAGONAL, Elemwise)
def _elemwise(key, op, feature, fgraph, node, input_states):
    return elemwise_preserves_zero_pattern(DIAGONAL, op, feature, node, input_states)
