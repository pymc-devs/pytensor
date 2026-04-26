from pytensor.tensor.assumptions import ALL_KEYS
from pytensor.tensor.assumptions.core import (
    FactState,
    lookup_assumption_rules,
    register_assumption,
)
from pytensor.tensor.blockwise import Blockwise


def _blockwise_delegate(op, feature, fgraph, node, input_states):
    """Delegate assumption inference to the ``core_op`` of a Blockwise wrapper."""
    core_op = op.core_op
    key = feature._current_key

    for fn in lookup_assumption_rules(key, core_op):
        result = fn(core_op, feature, fgraph, node, input_states)
        if any(result):
            return result

    meth = getattr(core_op, "infer_assumption", None)
    if meth is not None:
        result = meth(key, feature, fgraph, node, input_states)
        if result is not NotImplemented:
            return result

    return [FactState.UNKNOWN] * len(node.outputs)


for _key in ALL_KEYS:
    register_assumption(_key, Blockwise)(_blockwise_delegate)
