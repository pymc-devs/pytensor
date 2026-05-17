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


def alloc_of_zero(key, op, feature, fgraph, node, input_states) -> list[FactState]:
    """``Alloc`` rule for DIAGONAL / LOWER_TRIANGULAR / UPPER_TRIANGULAR: TRUE when
    the fill value is the scalar 0 (an all-zero square matrix), FALSE when it is a
    known non-zero scalar -- the off-diagonal entries are then non-zero, so none of
    these properties holds.

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


def eye_identity_rule(key, op, feature, fgraph, node, input_states) -> list[FactState]:
    """Rule body: TRUE when an :class:`Eye` node produces the identity matrix (square, k == 0)."""
    n, m, k = node.inputs
    if not isinstance(k, TensorConstant):
        return [FactState.UNKNOWN]
    if k.data.item() != 0:
        # Ones sit on an off-main diagonal, so this is not the identity.
        return [FactState.FALSE]
    if n is m:
        return [FactState.TRUE]
    if isinstance(n, TensorConstant) and isinstance(m, TensorConstant):
        return true_if(n.data.item() == m.data.item(), else_false=True)
    return [FactState.UNKNOWN]


def alloc_diag_at_offset_zero(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """Rule body: TRUE when :class:`AllocDiag` places values on the main diagonal,
    FALSE when it places them on any other diagonal (off-main entries break diagonal /
    symmetric / PD structure regardless of the diagonal vector's values)."""
    return [FactState.TRUE if op.offset == 0 else FactState.FALSE]


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
