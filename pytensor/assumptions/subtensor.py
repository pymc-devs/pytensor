from pytensor.assumptions.core import FactState, true_if


def subtensor_propagates_matrix_property(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """Generic rule: matrix property propagates from ``input[0]`` iff the ``Subtensor``
    leaves the trailing two matrix axes untouched.

    Indexing only the leading batch axes (with scalars or slices) yields a stack
    or single instance of the original matrices, so any per-matrix property
    (symmetric, PSD, triangular, orthogonal, diagonal) carries through. Indexing
    into either of the last two axes — via a non-full slice or a scalar — breaks
    that guarantee.
    """
    if input_states[0] is not FactState.TRUE:
        return [FactState.UNKNOWN]
    input_ndim = node.inputs[0].type.ndim
    if input_ndim < 2:
        return [FactState.UNKNOWN]
    idx_list = op.idx_list
    n_explicit_matrix_axes = max(0, len(idx_list) - (input_ndim - 2))
    if n_explicit_matrix_axes == 0:
        return [FactState.TRUE]
    full_slice = slice(None, None, None)
    return true_if(
        all(idx_list[-(i + 1)] == full_slice for i in range(n_explicit_matrix_axes))
    )
