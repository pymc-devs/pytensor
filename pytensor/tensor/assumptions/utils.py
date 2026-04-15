from typing import Any

from pytensor.graph import FunctionGraph
from pytensor.graph.basic import Constant
from pytensor.tensor.assumptions import AssumptionFeature, AssumptionKey
from pytensor.tensor.assumptions.core import FactState


def true_if(cond: bool) -> list[FactState]:
    """``[TRUE]`` when *cond* holds, ``[UNKNOWN]`` otherwise."""
    return [FactState.TRUE] if cond else [FactState.UNKNOWN]


def propagate_first(op, feature, fgraph, node, input_states) -> list[FactState]:
    """Output inherits the assumption iff the first input has it."""
    return true_if(input_states[0])


def all_inputs_have_key(op, feature, fgraph, node, input_states) -> list[FactState]:
    """Output inherits the assumption iff *every* input has it."""
    return true_if(all(input_states))


def eye_is_identity(node) -> bool:
    """True when an :class:`Eye` node produces the identity matrix (square, k == 0)."""
    n, m, k = node.inputs
    if not (isinstance(k, Constant) and k.data.item() == 0):
        return False
    if n is m:
        return True
    if isinstance(n, Constant) and isinstance(m, Constant):
        return n.data.item() == m.data.item()
    return False


def _same_variable(a, b) -> bool:
    """True when *a* and *b* represent the same graph variable, including ``ScalarFromTensor`` wrappers."""
    if a is b:
        return True
    if (
        a.owner is not None
        and b.owner is not None
        and type(a.owner.op) is type(b.owner.op)
        and len(a.owner.inputs) == 1
        and len(b.owner.inputs) == 1
        and a.owner.inputs[0] is b.owner.inputs[0]
    ):
        return True
    return False


def indexes_diagonal(node) -> bool:
    """True when an ``*IncSubtensor*`` node modifies only diagonal entries."""
    from pytensor.tensor.subtensor import AdvancedIncSubtensor, IncSubtensor

    op = node.op
    if isinstance(op, AdvancedIncSubtensor):
        # inputs: (x, y, *index_arrays)
        index_arrays = node.inputs[2:]
        if len(index_arrays) >= 2:
            return _same_variable(index_arrays[-2], index_arrays[-1])
        return False

    if isinstance(op, IncSubtensor):
        # idx_list entries: int = scalar index (consumes a dynamic input),
        #                   slice = slice (no dynamic input for static parts)
        # Dynamic inputs are in node.inputs[2:], one per non-slice entry.
        idx_list = op.idx_list
        if len(idx_list) < 2:
            return False
        # Last two entries must both be scalar indices (not slices)
        if isinstance(idx_list[-1], slice) or isinstance(idx_list[-2], slice):
            return False
        # Map each non-slice idx_list entry to its dynamic input
        dynamic_inputs = list(node.inputs[2:])
        non_slice_positions = [
            i for i, entry in enumerate(idx_list) if not isinstance(entry, slice)
        ]
        if len(non_slice_positions) < 2:
            return False
        # The last two non-slice positions correspond to the last two dynamic inputs
        pos_a = non_slice_positions[-2]
        pos_b = non_slice_positions[-1]
        # idx in dynamic_inputs list = count of non-slice entries before this one
        dyn_idx_a = sum(1 for e in idx_list[:pos_a] if not isinstance(e, slice))
        dyn_idx_b = sum(1 for e in idx_list[:pos_b] if not isinstance(e, slice))
        if dyn_idx_a < len(dynamic_inputs) and dyn_idx_b < len(dynamic_inputs):
            return _same_variable(dynamic_inputs[dyn_idx_a], dynamic_inputs[dyn_idx_b])
        return False

    return False


def check_assumption(
    fgraph: FunctionGraph | None, var: Any, key: AssumptionKey
) -> bool:
    """Return True iff *key* is definitively TRUE for *var* in *fgraph*.

    Lazily attaches :class:`AssumptionFeature` to *fgraph* if it is not already present.
    """
    if fgraph is None:
        return None
    feature = getattr(fgraph, "assumption_feature", None)
    if feature is None:
        feature = AssumptionFeature()
        fgraph.attach_feature(feature)
    return feature.check(var, key)
