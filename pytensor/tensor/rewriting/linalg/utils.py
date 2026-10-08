import logging
from collections.abc import Iterator

from pytensor import tensor as pt
from pytensor.assumptions import (
    LOWER_TRIANGULAR,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    UPPER_TRIANGULAR,
    check_assumption,
)
from pytensor.graph import Apply, Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.scalar.basic import Mul
from pytensor.tensor.basic import (
    Eye,
    TensorVariable,
    atleast_Nd,
    diagonal,
)
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.math import variadic_mul
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)


logger = logging.getLogger(__name__)

MATRIX_INVERSE_OPS = (MatrixInverse, MatrixPinv)

# Mapping of ``solve``'s ``assume_a`` for ``A`` to the one valid for ``A.mT``: triangular
# flips direction, sym/pos/gen are preserved (since A.mT == A for sym/pos).
ASSUME_A_OF_TRANSPOSE = {
    "lower triangular": "upper triangular",
    "upper triangular": "lower triangular",
    "pos": "pos",
    "sym": "sym",
    "gen": "gen",
}


def matrix_diagonal_product(x):
    return pt.prod(diagonal(x, axis1=-2, axis2=-1), axis=-1)


def get_assume_a(fgraph, A):
    """Return the most-specific ``solve`` ``assume_a`` for ``A`` from tags and assumptions."""
    if getattr(A.tag, "lower_triangular", False) or check_assumption(
        fgraph, A, LOWER_TRIANGULAR
    ):
        return "lower triangular"
    if getattr(A.tag, "upper_triangular", False) or check_assumption(
        fgraph, A, UPPER_TRIANGULAR
    ):
        return "upper triangular"
    if getattr(A.tag, "psd", None) is True or check_assumption(
        fgraph, A, POSITIVE_DEFINITE
    ):
        return "pos"
    if getattr(A.tag, "symmetric", False) or check_assumption(fgraph, A, SYMMETRIC):
        return "sym"
    return "gen"


def strip_left_expand_dims(x: TensorVariable) -> tuple[TensorVariable, bool]:
    """Peel left expand_dims and left-expanded matrix transposes off ``x``.

    Batched graphs pad matrix operands with left ``expand_dims``
    (``RandomVariable.make_node`` and ``Blockwise.make_node`` both do), hiding
    the owner that structural matchers look for.

    Returns
    -------
    core : TensorVariable
        ``x`` with all left padding and left-expanded matrix transposes peeled.
    transposed : bool
        Whether ``core``'s last two axes are transposed relative to ``x``.
    """
    transposed = False
    while True:
        match x.owner_op_and_inputs:
            case (DimShuffle(is_left_expand_dims=True), inner):
                x = inner  # type: ignore[assignment]
            case (DimShuffle(is_left_expanded_matrix_transpose=True), inner):
                x = inner  # type: ignore[assignment]
                transposed = not transposed
            case _:
                return x, transposed


def clients_through_padding(
    fgraph: FunctionGraph, var: TensorVariable
) -> Iterator[tuple[Apply, int]]:
    """Iterate over clients of ``var``, looking through left expand_dims clients.

    Yields
    ------
    client : Apply
        A client of ``var`` or of one of its left expand_dims aliases.
    index : int
        Index at which ``var`` (or its padded alias) enters ``client``.
    """
    for client, idx in fgraph.clients[var]:
        match client.op:
            case DimShuffle(is_left_expand_dims=True):
                yield from clients_through_padding(fgraph, client.outputs[0])  # type: ignore[arg-type]
            case _:
                yield client, idx


def rebroadcast_like(new: TensorVariable, old: TensorVariable) -> TensorVariable:
    """Match ``new``'s type to ``old``'s so it is a valid replacement for ``old``.

    The counterpart of `strip_left_expand_dims`: a rewrite that matched
    through padding builds its replacement from core variables, then restores
    the original output type here. Leading broadcastable dims are added or
    squeezed as needed.
    """
    if new.type == old.type:
        return new

    if new.type.ndim > old.type.ndim:
        new = new.squeeze(axis=tuple(range(new.type.ndim - old.type.ndim)))
    else:
        new = atleast_Nd(new, n=old.type.ndim)
    if new.type.dtype != old.type.dtype:
        new = new.astype(old.type.dtype)
    if not old.type.is_super(new.type):
        new = pt.broadcast_to(new, old.shape)
    return new


def is_matrix_transpose(x: TensorVariable) -> bool:
    """Check if a variable corresponds to a transpose of the last two axes"""
    match x.owner_op:
        case DimShuffle(is_left_expanded_matrix_transpose=True):
            return True
    return False


def is_eye_mul(x) -> None | tuple[TensorVariable, TensorVariable]:
    # Check if we have a Multiplication with an Eye inside
    # Note: This matches for cases like (eye * 0)!
    match x.owner_op_and_inputs:
        case Elemwise(Mul()), *mul_inputs:
            pass
        case _:
            return None

    x_bcast = x.type.broadcastable[-2:]
    eye_input = None
    non_eye_inputs = []
    for mul_input in mul_inputs:
        # We only care about Eye if it's not broadcasting in the multiplication
        if mul_input.type.broadcastable[-2:] == x_bcast:
            match mul_input.owner_op_and_inputs:
                case (Eye(), _, _, Constant(0)):
                    eye_input = mul_input
                    continue
                # This whole condition checks if there is an Eye hiding inside a DimShuffle.
                # This arises from batched elementwise multiplication between a tensor and an eye, e.g.:
                # tensor(shape=(None, 3, 3) * eye(3). This is still potentially valid for diag rewrites.
                case (DimShuffle(is_left_expand_dims=True), ds_input):
                    match ds_input.owner_op_and_inputs:
                        case (Eye(), _, _, Constant(0)):
                            eye_input = mul_input
                            continue
        # If no match:
        non_eye_inputs.append(mul_input)

    if eye_input is None:
        return None

    return eye_input, variadic_mul(*non_eye_inputs)


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([OpPattern(DimShuffle, is_left_expanded_matrix_transpose=True)])
def useless_symmetric_transpose(fgraph, node):
    x = node.inputs[0]
    if getattr(x.tag, "symmetric", False) or check_assumption(fgraph, x, SYMMETRIC):
        return [atleast_Nd(x, n=node.outputs[0].type.ndim)]
