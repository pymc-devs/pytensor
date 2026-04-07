from pytensor.tensor.assumptions.core import (
    FactState,
    lookup_assumption_rule,
    register_assumption,
)
from pytensor.tensor.assumptions.diagonal import DIAGONAL
from pytensor.tensor.assumptions.orthogonal import ORTHOGONAL
from pytensor.tensor.assumptions.positive_definite import (
    POSITIVE_DEFINITE,
)
from pytensor.tensor.assumptions.symmetric import SYMMETRIC
from pytensor.tensor.assumptions.triangular import (
    LOWER_TRIANGULAR,
    UPPER_TRIANGULAR,
)
from pytensor.tensor.blockwise import Blockwise


ALL_KEYS = (
    DIAGONAL,
    LOWER_TRIANGULAR,
    UPPER_TRIANGULAR,
    SYMMETRIC,
    POSITIVE_DEFINITE,
    ORTHOGONAL,
)


def _blockwise_delegate(op, feature, fgraph, node, input_states):
    """Delegate assumption inference to the ``core_op`` of a Blockwise wrapper."""
    core_op = op.core_op
    key = feature._current_key

    fn = lookup_assumption_rule(key, core_op)
    if fn is not None:
        return fn(core_op, feature, fgraph, node, input_states)

    meth = getattr(core_op, "infer_assumption", None)
    if meth is not None:
        result = meth(key, feature, fgraph, node, input_states)
        if result is not NotImplemented:
            return result

    return [FactState.UNKNOWN] * len(node.outputs)


for _key in ALL_KEYS:
    register_assumption(_key, Blockwise)(_blockwise_delegate)
