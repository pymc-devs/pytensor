from functools import wraps

from pytensor.assumptions.alloc import eye_identity_rule
from pytensor.assumptions.core import (
    NOT_ORTHOGONAL,
    ORTHOGONAL,
    ORTHONORMAL_COLUMNS,
    ORTHONORMAL_ROWS,
    FactState,
    OrthogonalFact,
    all_inputs_have_key,
    propagate_first,
    register_assumption,
)
from pytensor.assumptions.dimshuffle import left_expand_dims_propagates_matrix_property
from pytensor.assumptions.subtensor import (
    incsubtensor_propagates_matrix_property,
    subtensor_propagates_matrix_property,
)
from pytensor.tensor.basic import Eye
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.decomposition.eigen import Eigh
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.schur import QZ, Schur
from pytensor.tensor.linalg.decomposition.svd import SVD
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.math import Dot
from pytensor.tensor.subtensor import IncSubtensor, Subtensor


BOTH = ORTHOGONAL.true_value
COLUMNS = frozenset({ORTHONORMAL_COLUMNS})
NONE = ORTHOGONAL.unknown

_FROM_STATE = {
    FactState.UNKNOWN: NONE,
    FactState.TRUE: BOTH,
    FactState.FALSE: ORTHOGONAL.false_value,
    FactState.CONFLICT: BOTH | ORTHOGONAL.false_value,
}


def _to_state(value: frozenset[OrthogonalFact]) -> FactState:
    if ORTHOGONAL.is_true(value):
        return FactState.TRUE
    if ORTHOGONAL.is_false(value):
        return FactState.FALSE
    return FactState.UNKNOWN


def on_full_orthogonality(rule):
    """Run a boolean rule on whether each input is orthogonal on both sides."""

    @wraps(rule)
    def lifted(key, op, feature, fgraph, node, input_states):
        states = [_to_state(value) for value in input_states]
        return [
            _FROM_STATE[state] for state in rule(key, op, feature, fgraph, node, states)
        ]

    return lifted


register_assumption(ORTHOGONAL, Eye)(on_full_orthogonality(eye_identity_rule))
register_assumption(ORTHOGONAL, MatrixInverse)(on_full_orthogonality(propagate_first))
register_assumption(ORTHOGONAL, MatrixPinv)(on_full_orthogonality(propagate_first))
register_assumption(ORTHOGONAL, BlockDiagonal)(
    on_full_orthogonality(all_inputs_have_key)
)
register_assumption(ORTHOGONAL, Subtensor)(
    on_full_orthogonality(subtensor_propagates_matrix_property)
)
register_assumption(ORTHOGONAL, IncSubtensor)(
    on_full_orthogonality(incsubtensor_propagates_matrix_property)
)


_TRANSPOSED: dict[OrthogonalFact, OrthogonalFact] = {
    ORTHONORMAL_COLUMNS: ORTHONORMAL_ROWS,
    ORTHONORMAL_ROWS: ORTHONORMAL_COLUMNS,
    NOT_ORTHOGONAL: NOT_ORTHOGONAL,
}


@register_assumption(ORTHOGONAL, DimShuffle)
def _dimshuffle(key, op, feature, fgraph, node, input_states):
    """A matrix transpose swaps the orthonormal sides, and left-expand-dims (batch
    broadcast) keeps them."""
    if op.is_matrix_transpose:
        return [frozenset(_TRANSPOSED[fact] for fact in input_states[0])]
    if left_expand_dims_propagates_matrix_property(op):
        return [input_states[0]]
    return [NONE]


@register_assumption(ORTHOGONAL, Dot)
def _dot(key, op, feature, fgraph, node, input_states):
    """A product keeps the orthonormal sides its factors share: with orthonormal columns,
    (A B)^T (A B) = B^T A^T A B = B^T B = I, and rows follow by transposing."""
    left, right = input_states
    return [left & right & BOTH]


@register_assumption(ORTHOGONAL, Schur)
def _schur(key, op, feature, fgraph, node, input_states):
    # Schur: A = Z @ T @ Z.T -> outputs are (T, Z); Z is orthogonal
    return [NONE, BOTH]


@register_assumption(ORTHOGONAL, QZ)
def _qz(key, op, feature, fgraph, node, input_states):
    # QZ: outputs are (AA, BB, Q, Z) or (AA, BB, alpha, beta, Q, Z), Q and Z last
    return [NONE] * (len(node.outputs) - 2) + [BOTH, BOTH]


@register_assumption(ORTHOGONAL, QR)
def _qr(key, op, feature, fgraph, node, input_states):
    # Full mode Q (output 0) is square and orthogonal. Economic Q has orthonormal
    # columns, and the "r" and "raw" modes return no Q.
    n_out = len(node.outputs)
    if op.mode == "full":
        return [BOTH] + [NONE] * (n_out - 1)
    if op.mode == "economic":
        return [COLUMNS] + [NONE] * (n_out - 1)
    return [NONE] * n_out


@register_assumption(ORTHOGONAL, SVD)
def _svd(key, op, feature, fgraph, node, input_states):
    # SVD full_matrices: outputs are (U, S, V) where U and V are orthogonal
    # Reduced: U and V have orthonormal columns/rows but aren't square
    if not op.compute_uv:
        return [NONE]
    if op.full_matrices:
        return [BOTH, NONE, BOTH]
    return [NONE] * 3


@register_assumption(ORTHOGONAL, Eigh)
def _eigh(key, op, feature, fgraph, node, input_states):
    # Eigh: outputs are (w, v) where v is the orthogonal eigenvector matrix
    return [NONE, BOTH]
