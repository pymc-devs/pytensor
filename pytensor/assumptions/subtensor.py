from pytensor.assumptions.core import (
    ALL_KEYS,
    ORTHOGONAL,
    FactState,
    register_assumption,
    true_if,
)
from pytensor.tensor.subtensor import IncSubtensor


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


def incsubtensor_propagates_matrix_property(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """``IncSubtensor`` writes ``value`` into a slice of ``base``. When the index
    leaves the trailing two (core) axes as full slices -- a batch-only write --
    the per-matrix property forwards:

    - ``set`` over the whole tensor: the output is just ``value``.
    - ``set`` over a partial batch region: every output slice comes from ``base``
      or ``value``, so both must carry the property.
    - ``inc``: the written region becomes ``base + value``; the property must be
      closed under addition -- true for every key except ``orthogonal``.

    An index that reaches into the core axes is left to the per-key rules (e.g.
    the diagonal-position write handled by ``indexes_diagonal``).
    """
    base_ndim = node.inputs[0].type.ndim
    if base_ndim < 2:
        return [FactState.UNKNOWN]
    idx_list = op.idx_list
    full_slice = slice(None, None, None)
    n_core = max(0, len(idx_list) - (base_ndim - 2))
    if n_core and not all(idx_list[-(i + 1)] == full_slice for i in range(n_core)):
        return [FactState.UNKNOWN]

    base_state, value_state = input_states[0], input_states[1]
    if op.set_instead_of_inc:
        if idx_list and all(idx == full_slice for idx in idx_list):
            return [value_state]
        return true_if(base_state is FactState.TRUE and value_state is FactState.TRUE)

    if key is ORTHOGONAL:
        return [FactState.UNKNOWN]
    return true_if(base_state is FactState.TRUE and value_state is FactState.TRUE)


for _key in ALL_KEYS:
    register_assumption(_key, IncSubtensor)(incsubtensor_propagates_matrix_property)
