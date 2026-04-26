from pytensor.scalar.basic import Add, Conj, Mul
from pytensor.tensor.assumptions.core import (
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.assumptions.utils import (
    all_inputs_have_key,
    eye_is_identity,
    propagate_first,
    true_if,
)
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


register_assumption(POSITIVE_DEFINITE, BlockDiagonal)(all_inputs_have_key)
register_assumption(POSITIVE_DEFINITE, MatrixInverse)(propagate_first)
register_assumption(POSITIVE_DEFINITE, KroneckerProduct)(all_inputs_have_key)


def _is_psd_full_shape(inp, feature) -> bool:
    """Whether ``inp`` is a known-PD matrix at full shape over the last two axes.

    PD is a property of square matrices; scalar/vector inputs that broadcast into a matrix shape do not preserve
    definiteness (e.g. ``A + c`` adds the all-ones matrix scaled by c, which is not generally PD-preserving). Only
    inputs whose last two axes are both present (non-broadcastable) and that are themselves marked PD count.
    """
    if inp.type.ndim < 2:
        return False
    if any(inp.type.broadcastable[-2:]):
        return False
    return bool(feature.check(inp, POSITIVE_DEFINITE))


@register_assumption(POSITIVE_DEFINITE, Elemwise)
def _elemwise(op, feature, fgraph, node, input_states):
    """Elementwise rules that preserve positive definiteness.

    - Add: every input is a full-shape PD matrix.
    - Mul: a positive scalar constant times all-PD-matrix factors.
    """
    scalar_op = op.scalar_op

    if isinstance(scalar_op, Add):
        return true_if(all(_is_psd_full_shape(inp, feature) for inp in node.inputs))

    if isinstance(scalar_op, Mul):
        for i, inp in enumerate(node.inputs):
            try:
                val = get_underlying_scalar_constant_value(inp)
            except (NotScalarConstantError, TypeError):
                continue
            if val <= 0:
                continue
            other_inputs = [node.inputs[j] for j in range(len(node.inputs)) if j != i]
            if all(_is_psd_full_shape(other, feature) for other in other_inputs):
                return [FactState.TRUE]
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
