from pytensor.assumptions.alloc import (
    alloc_diag_at_offset_zero,
    alloc_is_symmetric,
    eye_identity_rule,
)
from pytensor.assumptions.core import (
    SYMMETRIC,
    FactState,
    all_inputs_have_key,
    propagate_first,
    register_assumption,
    true_if,
)
from pytensor.assumptions.dimshuffle import left_expand_dims_propagates_matrix_property
from pytensor.assumptions.dot import match_congruence
from pytensor.assumptions.subtensor import subtensor_propagates_matrix_property
from pytensor.tensor.basic import Alloc, AllocDiag, Eye
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.linalg.products import KroneckerProduct
from pytensor.tensor.linalg.solvers.linear_control import (
    SolveBilinearDiscreteLyapunov,
)
from pytensor.tensor.math import Dot
from pytensor.tensor.subtensor import Subtensor


def _preserves_symmetry_under_broadcast(inp, feature) -> bool:
    """Whether ``inp``'s contribution to an Elemwise output is invariant under swapping the last two axes.

    A 0d scalar broadcasts to a constant matrix (symmetric). A 1d vector broadcasts as a row and breaks symmetry. For
    ndim >= 2, both trailing axes must either be jointly broadcastable (scalar in the matrix dims) or the input itself
    must be known symmetric; a single broadcastable trailing axis (row/col vector) breaks symmetry.
    """
    ndim = inp.type.ndim
    if ndim < 2:
        return bool(ndim == 0)

    last_two_bc = inp.type.broadcastable[-2:]
    if all(last_two_bc):
        return True
    if any(last_two_bc):
        return False
    return bool(feature.check(inp, SYMMETRIC))


register_assumption(SYMMETRIC, Eye)(eye_identity_rule)
register_assumption(SYMMETRIC, AllocDiag)(alloc_diag_at_offset_zero)
register_assumption(SYMMETRIC, Alloc)(alloc_is_symmetric)
register_assumption(SYMMETRIC, BlockDiagonal)(all_inputs_have_key)
register_assumption(SYMMETRIC, MatrixInverse)(propagate_first)
register_assumption(SYMMETRIC, MatrixPinv)(propagate_first)
register_assumption(SYMMETRIC, KroneckerProduct)(all_inputs_have_key)
register_assumption(SYMMETRIC, Subtensor)(subtensor_propagates_matrix_property)


@register_assumption(SYMMETRIC, DimShuffle)
def _dimshuffle(key, op, feature, fgraph, node, input_states):
    """Symmetry survives matrix transpose and left-expand-dims (batch broadcast)."""
    if input_states[0] is not FactState.TRUE:
        return [FactState.UNKNOWN]
    if op.is_matrix_transpose:
        return [FactState.TRUE]
    if left_expand_dims_propagates_matrix_property(op):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(SYMMETRIC, SolveBilinearDiscreteLyapunov)
def _discrete_lyapunov(key, op, feature, fgraph, node, input_states):
    """The unique solution ``X`` of ``A X A^T - X = Q`` is symmetric when ``Q`` is symmetric.

    Inputs are ``(A, Q)``; the output's symmetry is governed entirely by ``Q``.
    """
    return true_if(input_states[1] is FactState.TRUE)


@register_assumption(SYMMETRIC, Dot)
def _dot_congruence(key, op, feature, fgraph, node, input_states):
    """``M @ S @ M.T`` (and the mirror ``M.T @ S @ M``) is symmetric when ``S`` is symmetric.

    Matches both Python associativities so the rule fires regardless of how the
    user parenthesized the triple product.
    """
    s = match_congruence(node)
    if s is None:
        return [FactState.UNKNOWN]
    return true_if(feature.check(s, SYMMETRIC))


@register_assumption(SYMMETRIC, Dot)
def _dot_self_product(key, op, feature, fgraph, node, input_states):
    """``A @ A`` is symmetric when ``A`` is symmetric -- ``(A A)ᵀ = Aᵀ Aᵀ = A A``.

    The product of two *different* symmetric matrices is not generally symmetric,
    so this fires only when both factors are the same variable.
    """
    a, b = node.inputs
    return true_if(a is b and input_states[0] is FactState.TRUE)


@register_assumption(SYMMETRIC, Elemwise)
def _elemwise(key, op, feature, fgraph, node, input_states):
    """Any elementwise op preserves symmetry when every input broadcasts to a
    symmetric pattern: ``out[i,j] = f(*xs[i,j])`` equals ``out[j,i]`` because
    each input contributes the same value at (i,j) and (j,i).
    """
    n_out = len(node.outputs)
    if node.outputs[0].type.ndim < 2:
        return [FactState.UNKNOWN] * n_out

    state = (
        FactState.TRUE
        if all(_preserves_symmetry_under_broadcast(inp, feature) for inp in node.inputs)
        else FactState.UNKNOWN
    )
    return [state] * n_out
