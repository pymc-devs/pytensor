from pytensor.assumptions.core import ALL_KEYS, FactState, register_assumption
from pytensor.tensor.reshape import JoinDims, SplitDims


def join_dims_propagates_matrix_property(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """``JoinDims`` merges a run of axes. The matrix property survives iff that
    run stays clear of the trailing two (core) axes."""
    core_start = node.inputs[0].type.ndim - 2
    if op.start_axis + op.n_axes <= core_start:
        return [input_states[0]]
    return [FactState.UNKNOWN]


def split_dims_propagates_matrix_property(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """``SplitDims`` splits one axis. The matrix property survives iff that axis
    is not one of the trailing two (core) axes."""
    if op.axis < node.inputs[0].type.ndim - 2:
        return [input_states[0]]
    return [FactState.UNKNOWN]


for _key in ALL_KEYS:
    register_assumption(_key, JoinDims)(join_dims_propagates_matrix_property)
    register_assumption(_key, SplitDims)(split_dims_propagates_matrix_property)
