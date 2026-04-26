from typing import Any

from pytensor.graph import FunctionGraph
from pytensor.graph.basic import Constant
from pytensor.tensor.assumptions import AssumptionFeature, AssumptionKey
from pytensor.tensor.assumptions.core import FactState
from pytensor.tensor.basic import (
    NotScalarConstantError,
    get_underlying_scalar_constant_value,
)


def true_if(cond: bool) -> list[FactState]:
    """``[TRUE]`` when *cond* holds, ``[UNKNOWN]`` otherwise."""
    return [FactState.TRUE] if cond else [FactState.UNKNOWN]


def propagate_first(op, feature, fgraph, node, input_states) -> list[FactState]:
    """Output inherits the assumption iff the first input has it."""
    return true_if(input_states[0])


def all_inputs_have_key(op, feature, fgraph, node, input_states) -> list[FactState]:
    """Output inherits the assumption iff *every* input has it."""
    return true_if(all(input_states))


def alloc_of_zero(op, feature, fgraph, node, input_states) -> list[FactState]:
    """``Alloc`` rule: TRUE iff the fill value is the scalar 0."""
    try:
        val = get_underlying_scalar_constant_value(node.inputs[0])
    except NotScalarConstantError:
        return [FactState.UNKNOWN]
    return true_if(val == 0)


def eye_is_identity(node) -> bool:
    """True when an :class:`Eye` node produces the identity matrix (square, k == 0)."""
    n, m, k = node.inputs
    if not (isinstance(k, Constant) and k.data.item() == 0):
        return False
    if n is m:
        return True
    if isinstance(n, Constant) and isinstance(m, Constant):
        return bool(n.data.item() == m.data.item())
    return False


def check_assumption(
    fgraph: FunctionGraph | None, var: Any, key: AssumptionKey
) -> bool:
    """Return True iff *key* is definitively TRUE for *var* in *fgraph*.

    Lazily attaches :class:`AssumptionFeature` to *fgraph* if it is not already present.
    """
    if fgraph is None:
        return False
    feature = getattr(fgraph, "assumption_feature", None)
    if feature is None:
        feature = AssumptionFeature()
        fgraph.attach_feature(feature)
    return feature.check(var, key)
