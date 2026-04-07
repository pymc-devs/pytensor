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
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.math import Dot


LOWER_TRIANGULAR = AssumptionKey("lower_triangular")
UPPER_TRIANGULAR = AssumptionKey("upper_triangular")


@register_assumption(LOWER_TRIANGULAR, Eye)
def _eye_lower(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(UPPER_TRIANGULAR, Eye)
def _eye_upper(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(LOWER_TRIANGULAR, AllocDiag)
def _alloc_diag_lower(op, feature, fgraph, node, input_states):
    return true_if(AllocDiag.is_offset_zero(node))


@register_assumption(UPPER_TRIANGULAR, AllocDiag)
def _alloc_diag_upper(op, feature, fgraph, node, input_states):
    return true_if(AllocDiag.is_offset_zero(node))


@register_assumption(LOWER_TRIANGULAR, Alloc)
@register_assumption(UPPER_TRIANGULAR, Alloc)
def _alloc_tri(op, feature, fgraph, node, input_states):
    try:
        val = get_underlying_scalar_constant_value(node.inputs[0])
        if val == 0:
            return [FactState.TRUE]
    except NotScalarConstantError:
        pass
    return [FactState.UNKNOWN]


@register_assumption(LOWER_TRIANGULAR, Cholesky)
def _chol_lower(op, feature, fgraph, node, input_states):
    return true_if(op.lower)


@register_assumption(UPPER_TRIANGULAR, Cholesky)
def _chol_upper(op, feature, fgraph, node, input_states):
    return true_if(not op.lower)


@register_assumption(LOWER_TRIANGULAR, BlockDiagonal)
def _block_diag_lower(op, feature, fgraph, node, input_states):
    return true_if(all(input_states))


@register_assumption(UPPER_TRIANGULAR, BlockDiagonal)
def _block_diag_upper(op, feature, fgraph, node, input_states):
    return true_if(all(input_states))


@register_assumption(LOWER_TRIANGULAR, MatrixInverse)
def _inv_lower(op, feature, fgraph, node, input_states):
    return true_if(input_states[0])


@register_assumption(UPPER_TRIANGULAR, MatrixInverse)
def _inv_upper(op, feature, fgraph, node, input_states):
    return true_if(input_states[0])


@register_assumption(LOWER_TRIANGULAR, Dot)
def _dot_lower(op, feature, fgraph, node, input_states):
    return true_if(all(input_states))


@register_assumption(UPPER_TRIANGULAR, Dot)
def _dot_upper(op, feature, fgraph, node, input_states):
    return true_if(all(input_states))
