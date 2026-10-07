import numpy as np

from pytensor.assumptions.core import (
    ALL_KEYS,
    FactState,
    register_assumption,
    true_if,
)
from pytensor.tensor.basic import (
    Alloc,
    NotScalarConstantError,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.variable import TensorConstant


def alloc_has_zero_off_diagonal(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """``Alloc`` rule for DIAGONAL / LOWER_TRIANGULAR / UPPER_TRIANGULAR: TRUE when
    the fill value is the scalar 0 (an all-zero square matrix) or the matrix is at
    most 1x1 (no off-diagonal entries), FALSE when it is a known non-zero scalar
    and the matrix is at least 2x2 -- the off-diagonal entries are then non-zero,
    so none of these properties holds.

    Requires the trailing two output dims to be statically known and equal; these
    properties apply only to square matrices. SYMMETRIC uses :func:`alloc_is_symmetric`
    instead, since a constant fill is symmetric whatever its value.
    """
    out_shape = node.outputs[0].type.shape
    if len(out_shape) < 2:
        return [FactState.UNKNOWN]
    m, n = out_shape[-2], out_shape[-1]
    if m is None or n is None or m != n:
        return [FactState.UNKNOWN]
    if n <= 1:
        return [FactState.TRUE]
    try:
        val = get_underlying_scalar_constant_value(node.inputs[0])
    except NotScalarConstantError:
        return [FactState.UNKNOWN]
    return true_if(val == 0, else_false=True)


def alloc_is_symmetric(key, op, feature, fgraph, node, input_states) -> list[FactState]:
    """``Alloc`` rule for SYMMETRIC: a scalar fill broadcasts one value to every
    entry, so a square output equals its transpose -- whatever the value.

    A non-scalar fill (e.g. a vector broadcast across rows) is left UNKNOWN.
    """
    out_shape = node.outputs[0].type.shape
    if len(out_shape) < 2:
        return [FactState.UNKNOWN]
    m, n = out_shape[-2], out_shape[-1]
    if m is None or n is None or m != n:
        return [FactState.UNKNOWN]
    return true_if(node.inputs[0].type.ndim == 0)


def _eye_is_square(n, m) -> FactState:
    """Static squareness of an :class:`Eye` from its row/column size inputs:
    TRUE when the two sizes are provably equal, FALSE when provably unequal,
    UNKNOWN while either remains symbolic."""
    if n is m:
        return FactState.TRUE
    if isinstance(n, TensorConstant) and isinstance(m, TensorConstant):
        return FactState.TRUE if n.data.item() == m.data.item() else FactState.FALSE
    return FactState.UNKNOWN


def eye_identity_rule(key, op, feature, fgraph, node, input_states) -> list[FactState]:
    """Rule for ORTHOGONAL / PERMUTATION / POSITIVE_DEFINITE: TRUE only when an
    :class:`Eye` is the identity matrix (square with ``k == 0``, or 0x0). Every
    other Eye -- rectangular, off-main, or the all-zero matrix of an off-shape
    band -- lacks all three properties, so it is FALSE once the shape is known
    and UNKNOWN while it is still symbolic.
    """
    n, m, k = node.inputs
    if not isinstance(k, TensorConstant):
        return [FactState.UNKNOWN]
    if k.data.item() == 0:
        return [_eye_is_square(n, m)]
    rows, cols = node.outputs[0].type.shape
    if rows is None or cols is None:
        return [FactState.UNKNOWN]
    return true_if(rows == cols == 0, else_false=True)


def eye_band_is_empty(node, k: int) -> FactState:
    """Decide statically whether an :class:`Eye`'s ``k``-th diagonal is empty."""
    rows, cols = node.outputs[0].type.shape
    if rows is None or cols is None:
        return FactState.UNKNOWN
    band_is_empty = rows == 0 or cols == 0 or k <= -rows or k >= cols
    return FactState.TRUE if band_is_empty else FactState.FALSE


def eye_zero_or_identity_rule(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """Rule for SYMMETRIC / DIAGONAL: a square :class:`Eye` has both properties
    when it is the identity (``k == 0``) *or* when the requested off-main band
    ``k`` falls entirely outside the shape, leaving an all-zero matrix. A
    non-square Eye, or a square one whose off-main band intersects the shape, is
    neither symmetric nor diagonal.
    """
    n, m, k = node.inputs
    if not isinstance(k, TensorConstant):
        return [FactState.UNKNOWN]
    square = _eye_is_square(n, m)
    if square is FactState.FALSE:
        return [FactState.FALSE]
    offset = k.data.item()
    if offset == 0:
        return [square]
    return [eye_band_is_empty(node, offset)]


def alloc_diag_band_is_zero(node) -> FactState:
    """Decide statically whether the vector an :class:`AllocDiag` places is zero."""
    diag = node.inputs[0]
    if 0 in diag.type.shape:
        return FactState.TRUE
    if isinstance(diag, TensorConstant):
        return FactState.FALSE if np.any(diag.data) else FactState.TRUE
    try:
        val = get_underlying_scalar_constant_value(diag)
    except NotScalarConstantError:
        return FactState.UNKNOWN
    if val == 0:
        return FactState.TRUE
    if None in diag.type.shape:
        # A non-zero fill of unknown size may still be empty.
        return FactState.UNKNOWN
    return FactState.FALSE


def alloc_diag_has_zero_off_diagonal(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """Rule for DIAGONAL / SYMMETRIC: TRUE when :class:`AllocDiag` places values on
    the main diagonal or places an all-zero vector, FALSE when it places a vector
    with a non-zero entry on any other diagonal."""
    if op.offset == 0:
        return [FactState.TRUE]
    return [alloc_diag_band_is_zero(node)]


def alloc_propagates_matrix_property(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """``Alloc`` broadcasts a value across new or expanded leading axes. When the
    value is already matrix-shaped with non-broadcastable core axes, the alloc
    cannot touch those axes, so the matrix property carries through unchanged."""
    value = node.inputs[0]
    if value.type.ndim >= 2 and not any(value.type.broadcastable[-2:]):
        return [input_states[0]]
    return [FactState.UNKNOWN]


for _key in ALL_KEYS:
    register_assumption(_key, Alloc)(alloc_propagates_matrix_property)
