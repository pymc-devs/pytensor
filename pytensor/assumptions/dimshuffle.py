from pytensor.assumptions.core import MATRIX_KEYS, FactState, register_assumption
from pytensor.scalar.basic import Conj
from pytensor.tensor.elemwise import DimShuffle, Elemwise


def conjugate_transpose_input(x):
    """Return ``X`` when ``x`` is the conjugate transpose of ``X``, else ``None``.

    For real ``X`` that is ``X.mT``. For complex ``X`` it is ``X.conj().mT`` or
    ``X.mT.conj()``. The transpose may also add broadcastable dimensions on the left.
    """
    match x.owner_op_and_inputs:
        case (DimShuffle(is_left_expanded_matrix_transpose=True), inner):
            if not inner.type.dtype.startswith("complex"):
                return inner
            match inner.owner_op_and_inputs:
                case (Elemwise(Conj()), X):
                    return X
        case (Elemwise(Conj()), inner):
            match inner.owner_op_and_inputs:
                case (DimShuffle(is_left_expanded_matrix_transpose=True), X):
                    return X
    return None


def left_expand_dims_propagates_matrix_property(op) -> bool:
    """True iff ``op`` is a ``DimShuffle`` adding new broadcast dims only on the left of a matrix.

    Such a DimShuffle (e.g. ``M[None, ..., :, :]``) stacks the operand along new
    leading batch axes without touching the trailing two axes. Every per-matrix
    property (symmetric, PSD, triangular, orthogonal, diagonal) carries through
    elementwise across the batch, so the output inherits the assumption.
    """
    return bool(
        isinstance(op, DimShuffle) and op.is_left_expand_dims and op.input_ndim >= 2
    )


def dimshuffle_propagates_matrix_property(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """A ``DimShuffle`` that only adds, drops, or permutes batch axes -- leaving
    the trailing two axes last and in order -- forwards the matrix property.
    Matrix transpose, which swaps the core axes, is handled per-key."""
    input_ndim = op.input_ndim
    if input_ndim < 2:
        return [FactState.UNKNOWN]
    if tuple(op.new_order[-2:]) == (input_ndim - 2, input_ndim - 1):
        return [input_states[0]]
    return [FactState.UNKNOWN]


for _key in MATRIX_KEYS:
    register_assumption(_key, DimShuffle)(dimshuffle_propagates_matrix_property)
