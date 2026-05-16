from pytensor.tensor.elemwise import DimShuffle


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
