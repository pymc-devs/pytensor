from pytensor.assumptions.alloc import eye_identity_rule
from pytensor.assumptions.core import (
    POSITIVE_DEFINITE,
    FactState,
    all_inputs_have_key,
    propagate_first,
    register_assumption,
    true_if,
)
from pytensor.assumptions.dimshuffle import left_expand_dims_propagates_matrix_property
from pytensor.assumptions.dot import match_congruence
from pytensor.assumptions.subtensor import subtensor_propagates_matrix_property
from pytensor.scalar.basic import Add, Conj, Mul
from pytensor.tensor.basic import AllocDiag, Eye
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.products import KroneckerProduct
from pytensor.tensor.linalg.solvers.linear_control import (
    SolveBilinearDiscreteLyapunov,
)
from pytensor.tensor.math import Dot
from pytensor.tensor.subtensor import Subtensor, _is_provably_positive


register_assumption(POSITIVE_DEFINITE, Eye)(eye_identity_rule)


@register_assumption(POSITIVE_DEFINITE, AllocDiag)
def _alloc_diag(key, op, feature, fgraph, node, input_states):
    """``diag(v)`` is positive definite iff it lies on the main diagonal and every
    element of ``v`` is provably positive."""
    if op.offset != 0:
        return [FactState.FALSE]

    [diag_values] = node.inputs
    return true_if(_is_provably_positive(diag_values))


register_assumption(POSITIVE_DEFINITE, BlockDiagonal)(all_inputs_have_key)
register_assumption(POSITIVE_DEFINITE, MatrixInverse)(propagate_first)
register_assumption(POSITIVE_DEFINITE, KroneckerProduct)(all_inputs_have_key)
register_assumption(POSITIVE_DEFINITE, Subtensor)(subtensor_propagates_matrix_property)


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
def _elemwise(key, op, feature, fgraph, node, input_states):
    """Elementwise rules that preserve positive definiteness.

    - Add: every input is a full-shape PD matrix.
    - Mul: a provably-positive scalar times all-PD-matrix factors.
    """
    scalar_op = op.scalar_op

    if isinstance(scalar_op, Add):
        return true_if(all(_is_psd_full_shape(inp, feature) for inp in node.inputs))

    if isinstance(scalar_op, Mul):
        for i, inp in enumerate(node.inputs):
            # Scaling a PD matrix by a positive scalar keeps it PD. The factor
            # must be constant across the matrix axes (both broadcastable);
            # per-batch variation is fine since every batch slice stays PD.
            if not (all(inp.type.broadcastable[-2:]) and _is_provably_positive(inp)):
                continue
            other_inputs = [node.inputs[j] for j in range(len(node.inputs)) if j != i]
            if other_inputs and all(
                _is_psd_full_shape(other, feature) for other in other_inputs
            ):
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


@register_assumption(POSITIVE_DEFINITE, DimShuffle)
def _dimshuffle(key, op, feature, fgraph, node, input_states):
    """PSD survives matrix transpose and left-expand-dims (batch broadcast)."""
    if input_states[0] is not FactState.TRUE:
        return [FactState.UNKNOWN]
    if op.is_matrix_transpose:
        return [FactState.TRUE]
    if left_expand_dims_propagates_matrix_property(op):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(POSITIVE_DEFINITE, Dot)
def _dot(key, op, feature, fgraph, node, input_states):
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
                return [FactState.TRUE]
        else:
            match right.owner_op_and_inputs:
                case (DimShuffle(is_left_expanded_matrix_transpose=True), inner) if (
                    inner is left
                ):
                    return [FactState.TRUE]

    # Congruence: M @ S @ M.T (or its mirror) is PSD when S is PSD. Only valid
    # for real matrices here -- the complex analogue requires M^H, which would
    # need a separate Hermitian-transpose match.
    if not is_complex:
        s = match_congruence(node)
        if s is not None and feature.check(s, POSITIVE_DEFINITE):
            return [FactState.TRUE]

    return [FactState.UNKNOWN]


@register_assumption(POSITIVE_DEFINITE, SolveBilinearDiscreteLyapunov)
def _discrete_lyapunov(key, op, feature, fgraph, node, input_states):
    """Propagate PSD from ``Q`` to the solution ``X`` of ``A X A^T - X = Q``.

    Mathematically, ``X`` is PSD when ``Q`` is PSD *and* the spectral radius of ``A``
    is strictly less than 1; without a stability assumption on ``A`` we propagate
    optimistically. Users with non-stable ``A`` can override with
    ``assume(X, positive_definite=False)``.
    """
    return true_if(input_states[1] is FactState.TRUE)
