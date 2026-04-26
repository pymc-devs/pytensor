from pytensor.tensor.assumptions.core import (
    AssumptionKey,
    FactState,
    register_assumption,
)
from pytensor.tensor.assumptions.utils import (
    all_inputs_have_key,
    alloc_of_zero,
    eye_is_identity,
    propagate_first,
    true_if,
)
from pytensor.tensor.basic import Alloc, AllocDiag, Eye
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.linalg.products import KroneckerProduct


SYMMETRIC = AssumptionKey("symmetric")


def _preserves_symmetry_under_broadcast(inp, feature) -> bool:
    """Whether ``inp``'s contribution to an Elemwise output is invariant under swapping the last two axes.

    A 0d scalar broadcasts to a constant matrix (symmetric). A 1d vector broadcasts as a row and breaks symmetry. For
    ndim >= 2, both trailing axes must either be jointly broadcastable (scalar in the matrix dims) or the input itself
    must be known symmetric; a single broadcastable trailing axis (row/col vector) breaks symmetry.
    """
    ndim = inp.type.ndim
    if ndim < 2:
        return ndim == 0

    last_two_bc = inp.type.broadcastable[-2:]
    if all(last_two_bc):
        return True
    if any(last_two_bc):
        return False
    return feature.check(inp, SYMMETRIC)


@register_assumption(SYMMETRIC, Eye)
def _eye(op, feature, fgraph, node, input_states):
    return true_if(eye_is_identity(node))


@register_assumption(SYMMETRIC, AllocDiag)
def _alloc_diag(op, feature, fgraph, node, input_states):
    return true_if(op.offset == 0)


register_assumption(SYMMETRIC, Alloc)(alloc_of_zero)
register_assumption(SYMMETRIC, BlockDiagonal)(all_inputs_have_key)
register_assumption(SYMMETRIC, MatrixInverse)(propagate_first)
register_assumption(SYMMETRIC, MatrixPinv)(propagate_first)
register_assumption(SYMMETRIC, KroneckerProduct)(all_inputs_have_key)


@register_assumption(SYMMETRIC, Elemwise)
def _elemwise(op, feature, fgraph, node, input_states):
    """Any elementwise op preserves symmetry when every input broadcasts to a
    symmetric pattern: ``out[i,j] = f(*xs[i,j])`` equals ``out[j,i]`` because
    each input contributes the same value at (i,j) and (j,i).
    """
    if node.outputs[0].type.ndim < 2:
        return [FactState.UNKNOWN]

    return true_if(
        all(_preserves_symmetry_under_broadcast(inp, feature) for inp in node.inputs)
    )
