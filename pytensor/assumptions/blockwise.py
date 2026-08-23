from pytensor.assumptions.core import (
    infer_assumption_for_node,
    register_universal_assumption,
)
from pytensor.tensor.blockwise import Blockwise


@register_universal_assumption(Blockwise)
def _blockwise_delegate(key, op, feature, fgraph, node, input_states):
    """Delegate assumption inference to the ``core_op`` of a Blockwise wrapper."""
    return infer_assumption_for_node(
        key, op.core_op, feature, fgraph, node, input_states
    )
