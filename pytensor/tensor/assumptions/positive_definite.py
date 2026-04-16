from pytensor.scalar.basic import Add, Conj, Mul
from pytensor.tensor.assumptions.core import (
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.assumptions.utils import eye_is_identity, true_if
from pytensor.tensor.basic import (
    AllocDiag,
    Eye,
    NotScalarConstantError,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.products import KroneckerProduct
from pytensor.tensor.math import Dot
from pytensor.tensor.variable import TensorConstant


POSITIVE_DEFINITE = AssumptionKey("positive_definite")


@register_assumption(POSITIVE_DEFINITE, Eye)
def _eye(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(POSITIVE_DEFINITE, AllocDiag)
def _alloc_diag(op, feature, fgraph, node, input_states):
    """diag(v) is positive definite if all elements of v are positive."""
    if op.offset != 0:
        # Off-diagonal matrices are not positive definite
        return [FactState.UNKNOWN]

    [diag_values] = node.inputs
    if isinstance(diag_values, TensorConstant):
        # Check if all diagonal values are positive
        data = diag_values.data
        if data.size > 0 and (data > 0).all():
            return [FactState.TRUE]

    return [FactState.UNKNOWN]


@register_assumption(POSITIVE_DEFINITE, BlockDiagonal)
def _block_diag(op, feature, fgraph, node, input_states):
    return true_if(all(feature.check(inp, POSITIVE_DEFINITE) for inp in node.inputs))


@register_assumption(POSITIVE_DEFINITE, MatrixInverse)
def _inv(op, feature, fgraph, node, input_states):
    return true_if(feature.check(node.inputs[0], POSITIVE_DEFINITE))


@register_assumption(POSITIVE_DEFINITE, KroneckerProduct)
def _kron(op, feature, fgraph, node, input_states):
    return true_if(all(feature.check(inp, POSITIVE_DEFINITE) for inp in node.inputs))


@register_assumption(POSITIVE_DEFINITE, Elemwise)
def _elemwise(op, feature, fgraph, node, input_states):
    """Elementwise operations that preserve positive definiteness.

    - Add: Sum of PD matrices is PD
    - Mul by positive scalar: c * A is PD if c > 0 and A is PD
    """
    scalar_op = op.scalar_op

    if isinstance(scalar_op, Add):
        # Sum of PD matrices is PD
        return true_if(
            all(feature.check(inp, POSITIVE_DEFINITE) for inp in node.inputs)
        )

    if isinstance(scalar_op, Mul):
        # c * A is PD if c > 0 and A is PD
        # Check if one input is a positive scalar constant and the other is PD
        for i, inp in enumerate(node.inputs):
            try:
                val = get_underlying_scalar_constant_value(inp)
                if val > 0:
                    # Check if all other inputs are PD
                    other_inputs = [
                        node.inputs[j] for j in range(len(node.inputs)) if j != i
                    ]
                    if all(
                        feature.check(other, POSITIVE_DEFINITE)
                        for other in other_inputs
                    ):
                        return [FactState.TRUE]
            except (NotScalarConstantError, TypeError):
                pass
        return [FactState.UNKNOWN]

    return [FactState.UNKNOWN] * len(node.outputs)


def _is_conjugate_transpose_of(y, x):
    """Check if y is the conjugate transpose of x (x.T.conj() or x.conj().T)."""
    match y.owner_op_and_inputs:
        case (Elemwise(scalar_op=Conj()), y_inner):  # x.T.conj()
            match y_inner.owner_op_and_inputs:
                case (DimShuffle(is_left_expanded_matrix_transpose=True), inner) if (
                    inner is x
                ):
                    return True
        case (
            DimShuffle(is_left_expanded_matrix_transpose=True),
            y_inner,
        ):  # x.conj().T
            match y_inner.owner_op_and_inputs:
                case (Elemwise(scalar_op=Conj()), inner) if inner is x:
                    return True
    return False


@register_assumption(POSITIVE_DEFINITE, Dot)
def _dot(op, feature, fgraph, node, input_states):
    """Detect positive semi-definite matrices from dot products.

    Patterns detected:
    - dot(x, x.T) for real matrices (and symmetric counterpart)
    - dot(x, x.conj().T) or dot(x, x.T.conj()) for complex matrices

    Technically these are semi-definite, and only positive definite if the input matrix is full rank.
    I'm adding this to have a conversation about it. We could add an @unsafe_assumption tag that could be excluded,
    or we could just ask users to tag their matrices in this case. But it's common enough in applications (Grahm matrices)
    to merit some thought.
    """
    x, y = node.inputs
    is_complex = x.type.dtype.startswith("complex")

    for left, right in [(x, y), (y, x)]:
        if is_complex:
            if _is_conjugate_transpose_of(right, left):
                return true_if(True)
        else:
            match right.owner_op_and_inputs:
                case (DimShuffle(is_left_expanded_matrix_transpose=True), inner) if (
                    inner is left
                ):
                    return true_if(True)

    return true_if(False)
