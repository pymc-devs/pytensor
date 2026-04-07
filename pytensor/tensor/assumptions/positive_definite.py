from pytensor.tensor.assumptions.core import AssumptionKey, register_assumption
from pytensor.tensor.assumptions.utils import eye_is_identity, true_if
from pytensor.tensor.basic import Eye
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.products import KroneckerProduct


POSITIVE_DEFINITE = AssumptionKey("positive_definite")


@register_assumption(POSITIVE_DEFINITE, Eye)
def _eye(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(POSITIVE_DEFINITE, BlockDiagonal)
def _block_diag(op, feature, fgraph, node, input_states):
    return true_if(all(feature.check(inp, POSITIVE_DEFINITE) for inp in node.inputs))


@register_assumption(POSITIVE_DEFINITE, MatrixInverse)
def _inv(op, feature, fgraph, node, input_states):
    return true_if(feature.check(node.inputs[0], POSITIVE_DEFINITE))


@register_assumption(POSITIVE_DEFINITE, KroneckerProduct)
def _kron(op, feature, fgraph, node, input_states):
    return true_if(all(feature.check(inp, POSITIVE_DEFINITE) for inp in node.inputs))
