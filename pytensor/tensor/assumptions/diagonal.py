from pytensor.scalar.basic import (
    Abs,
    Add,
    ArcSin,
    ArcSinh,
    ArcTan,
    ArcTanh,
    Ceil,
    Conj,
    Deg2Rad,
    Expm1,
    Floor,
    Identity,
    Imag,
    Log1p,
    Mul,
    Neg,
    Pow,
    Rad2Deg,
    Real,
    RoundHalfAwayFromZero,
    RoundHalfToEven,
    Sign,
    Sin,
    Sinh,
    Sqr,
    Sqrt,
    Sub,
    Tan,
    Tanh,
    TrueDiv,
    Trunc,
)
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
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.linalg.products import KroneckerProduct
from pytensor.tensor.math import Dot
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    IncSubtensor,
    indices_from_subtensor,
)


DIAGONAL = AssumptionKey("diagonal")


def indexes_diagonal(node) -> bool:
    """True when an ``*IncSubtensor*`` node modifies only diagonal entries."""

    op = node.op
    if not isinstance(op, AdvancedIncSubtensor | IncSubtensor):
        return False

    indices = indices_from_subtensor(node.inputs[2:], op.idx_list)
    if len(indices) < 2:
        return False
    if any(isinstance(idx, slice) for idx in indices):
        return False
    return all(indices[0] is idx for idx in indices[1:])


@register_assumption(DIAGONAL, Eye)
def _eye(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(DIAGONAL, AllocDiag)
def _alloc_diag(op, feature, fgraph, node, input_states):
    return true_if(op.offset == 0)


@register_assumption(DIAGONAL, Alloc)
def _alloc(op, feature, fgraph, node, input_states):
    try:
        val = get_underlying_scalar_constant_value(node.inputs[0])
        if val == 0:
            return [FactState.TRUE]
    except NotScalarConstantError:
        pass
    return [FactState.UNKNOWN]


@register_assumption(DIAGONAL, Cholesky)
def _cholesky(op, feature, fgraph, node, input_states):
    return true_if(input_states[0])


@register_assumption(DIAGONAL, BlockDiagonal)
def _block_diag(op, feature, fgraph, node, input_states):
    return true_if(all(input_states))


@register_assumption(DIAGONAL, MatrixInverse)
def _inv(op, feature, fgraph, node, input_states):
    return true_if(input_states[0])


@register_assumption(DIAGONAL, MatrixPinv)
def _pinv(op, feature, fgraph, node, input_states):
    return true_if(input_states[0])


@register_assumption(DIAGONAL, KroneckerProduct)
def _kron(op, feature, fgraph, node, input_states):
    return true_if(all(feature.check(inp, DIAGONAL) for inp in node.inputs))


@register_assumption(DIAGONAL, Dot)
def _dot(op, feature, fgraph, node, input_states):
    return true_if(all(input_states))


@register_assumption(DIAGONAL, IncSubtensor)
@register_assumption(DIAGONAL, AdvancedIncSubtensor)
def _inc_subtensor(op, feature, fgraph, node, input_states):
    if input_states[0] and indexes_diagonal(node):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(DIAGONAL, DimShuffle)
def _dimshuffle(op, feature, fgraph, node, input_states):
    if not input_states[0]:
        return [FactState.UNKNOWN]

    if op.is_transpose:
        nd = op.input_ndim
        if nd >= 2:
            last_two_swapped = (*tuple(range(nd - 2)), nd - 1, nd - 2)
            if op.new_order == last_two_swapped:
                return [FactState.TRUE]

    if op.is_expand_dims and op.input_ndim >= 2:
        if op.new_order[-op.input_ndim :] == tuple(range(op.input_ndim)):
            return [FactState.TRUE]

    return [FactState.UNKNOWN]


_ZERO_PRESERVING_UNARY = (
    Abs,
    ArcSin,
    ArcSinh,
    ArcTan,
    ArcTanh,
    Ceil,
    Conj,
    Deg2Rad,
    Expm1,
    Floor,
    Identity,
    Imag,
    Log1p,
    Neg,
    Rad2Deg,
    Real,
    RoundHalfAwayFromZero,
    RoundHalfToEven,
    Sign,
    Sin,
    Sinh,
    Sqr,
    Sqrt,
    Tan,
    Tanh,
    Trunc,
)


@register_assumption(DIAGONAL, Elemwise)
def _elemwise(op, feature, fgraph, node, input_states):
    scalar_op = op.scalar_op

    if isinstance(scalar_op, Mul):
        return true_if(
            any(
                feature.check(inp, DIAGONAL)
                and inp.type.ndim >= 2
                and not any(inp.type.broadcastable[-2:])
                for inp in node.inputs
            )
        )

    if isinstance(scalar_op, (Add, Sub)):
        return true_if(all(feature.check(inp, DIAGONAL) for inp in node.inputs))

    if isinstance(scalar_op, TrueDiv):
        return true_if(feature.check(node.inputs[0], DIAGONAL))

    if isinstance(scalar_op, Pow):
        if not feature.check(node.inputs[0], DIAGONAL):
            return [FactState.UNKNOWN]
        try:
            k = get_underlying_scalar_constant_value(node.inputs[1])
            if k > 0:
                return [FactState.TRUE]
        except NotScalarConstantError:
            pass
        return [FactState.UNKNOWN]

    if isinstance(scalar_op, _ZERO_PRESERVING_UNARY) and len(node.inputs) == 1:
        return true_if(input_states[0])

    return [FactState.UNKNOWN]
