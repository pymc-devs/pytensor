from pytensor.tensor.assumptions.core import (
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.assumptions.utils import eye_is_identity, true_if
from pytensor.tensor.basic import (
    Alloc,
    AllocDiag,
    Eye,
    NotScalarConstantError,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.linalg.products import KroneckerProduct


SYMMETRIC = AssumptionKey("symmetric")


@register_assumption(SYMMETRIC, Eye)
def _eye(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(SYMMETRIC, AllocDiag)
def _alloc_diag(op, feature, fgraph, node, input_states):
    return true_if(AllocDiag.is_offset_zero(node))


@register_assumption(SYMMETRIC, Alloc)
def _alloc(op, feature, fgraph, node, input_states):
    try:
        val = get_underlying_scalar_constant_value(node.inputs[0])
        if val == 0:
            return [FactState.TRUE]
    except NotScalarConstantError:
        pass
    return [FactState.UNKNOWN]


@register_assumption(SYMMETRIC, BlockDiagonal)
def _block_diag(op, feature, fgraph, node, input_states):
    return true_if(all(feature.check(inp, SYMMETRIC) for inp in node.inputs))


@register_assumption(SYMMETRIC, MatrixInverse)
def _inv(op, feature, fgraph, node, input_states):
    return true_if(feature.check(node.inputs[0], SYMMETRIC))


@register_assumption(SYMMETRIC, MatrixPinv)
def _pinv(op, feature, fgraph, node, input_states):
    return true_if(feature.check(node.inputs[0], SYMMETRIC))


@register_assumption(SYMMETRIC, KroneckerProduct)
def _kron(op, feature, fgraph, node, input_states):
    return true_if(all(feature.check(inp, SYMMETRIC) for inp in node.inputs))
