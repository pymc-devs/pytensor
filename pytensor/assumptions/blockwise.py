from pytensor.assumptions.core import (
    ALL_KEYS,
    infer_assumption_for_node,
    register_assumption,
)
from pytensor.tensor.blockwise import Blockwise


def _blockwise_delegate(key, op, feature, fgraph, node, input_states):
    """Delegate assumption inference to the ``core_op`` of a Blockwise wrapper."""
    return infer_assumption_for_node(
        key, op.core_op, feature, fgraph, node, input_states
    )


for _key in ALL_KEYS:
    register_assumption(_key, Blockwise)(_blockwise_delegate)
