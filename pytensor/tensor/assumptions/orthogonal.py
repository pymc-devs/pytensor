from pytensor.tensor.assumptions.core import (
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.assumptions.utils import eye_is_identity, true_if
from pytensor.tensor.basic import Eye
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.inverse import MatrixInverse


ORTHOGONAL = AssumptionKey("orthogonal")


@register_assumption(ORTHOGONAL, Eye)
def _eye(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(ORTHOGONAL, MatrixInverse)
def _inv(op, feature, fgraph, node, input_states):
    return true_if(feature.check(node.inputs[0], ORTHOGONAL))


@register_assumption(ORTHOGONAL, DimShuffle)
def _transpose(op, feature, fgraph, node, input_states):
    if op.is_matrix_transpose:
        return true_if(input_states[0])
    return [FactState.UNKNOWN]
