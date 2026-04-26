from pytensor.tensor.assumptions._elemwise import elemwise_preserves_zero_pattern
from pytensor.tensor.assumptions.core import (
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.assumptions.orthogonal import ORTHOGONAL
from pytensor.tensor.assumptions.utils import (
    all_inputs_have_key,
    alloc_of_zero,
    eye_is_identity,
    propagate_first,
    true_if,
)
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
    indices_from_subtensor,
)


DIAGONAL = AssumptionKey("diagonal")


def indexes_diagonal(node) -> bool:
    """True when an ``*IncSubtensor*`` node modifies only diagonal entries."""

    op = node.op
    if not isinstance(op, AdvancedIncSubtensor | IncSubtensor):
        return False

    indices = indices_from_subtensor(node.inputs[2:], op.idx_list)
    if len(indices) < 2:
        return False
    if any(isinstance(idx, slice) for idx in indices):
        return False
    return all(indices[0] is idx for idx in indices[1:])


@register_assumption(DIAGONAL, Eye)
def _eye(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(DIAGONAL, AllocDiag)
def _alloc_diag(op, feature, fgraph, node, input_states):
    return true_if(op.offset == 0)


register_assumption(DIAGONAL, Alloc)(alloc_of_zero)
register_assumption(DIAGONAL, Cholesky)(propagate_first)
register_assumption(DIAGONAL, BlockDiagonal)(all_inputs_have_key)
register_assumption(DIAGONAL, MatrixInverse)(propagate_first)
register_assumption(DIAGONAL, MatrixPinv)(propagate_first)
register_assumption(DIAGONAL, KroneckerProduct)(all_inputs_have_key)
register_assumption(DIAGONAL, Dot)(all_inputs_have_key)


@register_assumption(DIAGONAL, Dot)
def _dot_orthogonal_xxt(op, feature, fgraph, node, input_states):
    """x @ x.T is diagonal (identity) when x is orthogonal."""
    a, b = node.inputs
    b_owner = b.owner
    if (
        feature.check(a, ORTHOGONAL)
        and b_owner is not None
        and isinstance(b_owner.op, DimShuffle)
        and b_owner.op.is_matrix_transpose
        and b_owner.inputs[0] is a
    ):
        return [FactState.TRUE]

    return [FactState.UNKNOWN]


@register_assumption(DIAGONAL, IncSubtensor)
@register_assumption(DIAGONAL, AdvancedIncSubtensor)
def _inc_subtensor(op, feature, fgraph, node, input_states):
    if input_states[0] and indexes_diagonal(node):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(DIAGONAL, DimShuffle)
def _dimshuffle(op, feature, fgraph, node, input_states):
    if not input_states[0]:
        return [FactState.UNKNOWN]

    if op.is_transpose:
        nd = op.input_ndim
        if nd >= 2:
            last_two_swapped = (*tuple(range(nd - 2)), nd - 1, nd - 2)
            if op.new_order == last_two_swapped:
                return [FactState.TRUE]

    if op.is_expand_dims and op.input_ndim >= 2:
        if op.new_order[-op.input_ndim :] == tuple(range(op.input_ndim)):
            return [FactState.TRUE]

    return [FactState.UNKNOWN]


@register_assumption(DIAGONAL, Elemwise)
def _elemwise(op, feature, fgraph, node, input_states):
    return elemwise_preserves_zero_pattern(DIAGONAL, op, feature, node, input_states)
