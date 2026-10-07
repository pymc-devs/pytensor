from pytensor.assumptions.core import (
    ALL_KEYS,
    infer_assumption_for_node,
    infer_assumption_from_client,
    register_assumption,
    register_client_inference,
)
from pytensor.tensor.blockwise import Blockwise


def _blockwise_delegate(key, op, feature, fgraph, node, input_states):
    """Delegate assumption inference to the ``core_op`` of a Blockwise wrapper."""
    return infer_assumption_for_node(
        key, op.core_op, feature, fgraph, node, input_states
    )


def _blockwise_client_delegate(key, op, node, input_index):
    """Delegate client inference to the ``core_op`` of a Blockwise wrapper."""
    return infer_assumption_from_client(key, op.core_op, node, input_index)


for _key in ALL_KEYS:
    register_assumption(_key, Blockwise)(_blockwise_delegate)
    register_client_inference(_key, Blockwise)(_blockwise_client_delegate)
