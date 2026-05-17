from pytensor.assumptions.core import ALL_KEYS, FactState, register_assumption
from pytensor.tensor.shape import Reshape, SpecifyShape


def specify_shape_propagates_matrix_property(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """``SpecifyShape`` only annotates a shape; the value and its axes are unchanged."""
    return [input_states[0]]


def reshape_propagates_matrix_property(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """``Reshape`` preserves the matrix property iff the trailing two dimensions
    are unchanged."""
    in_shape = node.inputs[0].type.shape
    out_shape = node.outputs[0].type.shape
    if len(in_shape) < 2 or len(out_shape) < 2:
        return [FactState.UNKNOWN]
    in_core = in_shape[-2:]
    out_core = out_shape[-2:]
    if None in in_core or None in out_core or in_core != out_core:
        return [FactState.UNKNOWN]
    return [input_states[0]]


for _key in ALL_KEYS:
    register_assumption(_key, SpecifyShape)(specify_shape_propagates_matrix_property)
    register_assumption(_key, Reshape)(reshape_propagates_matrix_property)
