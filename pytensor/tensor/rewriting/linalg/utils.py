import logging
from collections.abc import Iterator
from functools import reduce

from pytensor import tensor as pt
from pytensor.assumptions import (
    LOWER_TRIANGULAR,
    POSITIVE_DEFINITE,
    SYMMETRIC,
    TRIDIAGONAL,
    UPPER_TRIANGULAR,
    assume,
    check_assumption,
)
from pytensor.graph import Apply, Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.scalar.basic import Add, Mul
from pytensor.tensor.basic import (
    Eye,
    TensorVariable,
    atleast_Nd,
    diagonal,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.decomposition.eigen import eigh
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.linalg.products import KroneckerProduct, kron
from pytensor.tensor.math import outer, variadic_mul
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
    "tridiagonal": "tridiagonal",
    "pos": "pos",
    "sym": "sym",
    "gen": "gen",
}


def matrix_diagonal_product(x):
    return pt.prod(diagonal(x, axis1=-2, axis2=-1), axis=-1)


def permutation_indices_to_det_sign(indices):
    """Permutation indices -> (-1) ** n_inversions, declaring unit magnitude."""
    inversions = pt.triu(
        (indices[..., :, None] > indices[..., None, :]).astype("int64"), k=1
    ).sum(axis=(-2, -1))
    sign = 1 - 2 * (inversions % 2)
    return assume(sign, unit_magnitude=True)


def get_assume_a(fgraph, A):
    """Return the most-specific ``solve`` ``assume_a`` for ``A`` from tags and assumptions."""
    if check_assumption(fgraph, A, TRIDIAGONAL):
        return "tridiagonal"
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


def strip_left_expand_dims_and_transpose(
    x: TensorVariable,
) -> tuple[TensorVariable, bool]:
    """Peel a left expand_dims or left-expanded matrix transpose off ``x``.

    Batched graphs insert dummy leading dims on matrix operands with left
    ``expand_dims`` (``RandomVariable.make_node`` and ``Blockwise.make_node``
    both do), hiding the owner that structural matchers look for. One peel
    suffices: adjacent DimShuffles are merged by ``local_dimshuffle_lift``.

    Returns
    -------
    core : TensorVariable
        ``x`` without the leading dummy dims and matrix transpose, if any.
    transposed : bool
        Whether ``core``'s last two axes are transposed relative to ``x``.
    """
    match x.owner_op_and_inputs:
        case (DimShuffle(is_left_expand_dims=True), inner):
            return inner, False  # type: ignore[return-value]
        case (DimShuffle(is_left_expanded_matrix_transpose=True), inner):
            return inner, True  # type: ignore[return-value]
    return x, False


def clients_through_expand_dims(
    fgraph: FunctionGraph, var: TensorVariable
) -> Iterator[tuple[Apply, int]]:
    """Iterate over clients of ``var``, looking through left expand_dims clients.

    Yields
    ------
    client : Apply
        A client of ``var`` or of one of its left expand_dims aliases.
    index : int
        Index at which ``var`` (or its expanded alias) enters ``client``.
    """
    for client, idx in fgraph.clients[var]:
        match client.op:
            case DimShuffle(is_left_expand_dims=True):
                yield from clients_through_expand_dims(fgraph, client.outputs[0])  # type: ignore[arg-type]
            case _:
                yield client, idx


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


def kron_leaves(x: TensorVariable) -> list[TensorVariable]:
    """Return the factors of a nested ``kron(kron(K1, K2), K3)`` chain in order."""
    match x.owner_op_and_inputs:
        case (KroneckerProduct(), left, right):
            return kron_leaves(left) + kron_leaves(right)  # type: ignore[arg-type]
    return [x]


def kron_plus_scaled_eye_consumers(
    fgraph: FunctionGraph, kron_node: Apply, core_op_type: type
) -> tuple[list[TensorVariable], list[tuple[Apply, TensorVariable]]] | None:
    """Find ``Blockwise(core_op_type)`` consumers of ``kron(*Ks) + c * I``.

    Every factor must be real and symmetric, because the eigh form built from the
    factors reads one triangle of each and treats it as Hermitian. Returns None
    when the pattern or the consumers are absent.

    Returns
    -------
    Ks : list of TensorVariable
        The Kronecker factors.
    consumers : list of (Apply, TensorVariable)
        Each node whose first input is a noisy matrix, possibly left-expanded or
        transposed, paired with the scalar ``c`` of that noisy matrix.
        Transposition is harmless because the noisy matrix is symmetric.
    """
    [kron_out] = kron_node.outputs
    Ks = kron_leaves(kron_out)  # type: ignore[arg-type]
    if any(K.type.ndim != 2 or K.type.dtype.startswith("complex") for K in Ks):
        return None
    if not all(check_assumption(fgraph, K, SYMMETRIC) for K in Ks):
        return None

    consumers: list[tuple[Apply, TensorVariable]] = []
    # The kron may reach the Add through left expand_dims lifted out of a batched
    # consumer.
    for client, kron_idx in clients_through_expand_dims(fgraph, kron_out):  # type: ignore[arg-type]
        match client.op:
            case Elemwise(Add()) if len(client.inputs) == 2:
                pass
            case _:
                continue
        kron_term, eye_term = client.inputs[kron_idx], client.inputs[1 - kron_idx]
        # A broadcasting operand, such as eye(1) or a 1x1 kron, spreads over every
        # entry of the other.
        if any(kron_term.type.broadcastable[-2:]) or any(
            eye_term.type.broadcastable[-2:]
        ):
            continue
        eye_match = is_eye_mul(eye_term)
        if eye_match is None:
            continue
        _, c = eye_match
        # The eigh form adds c to a 1-D eigenvalue vector.
        if not all(c.type.broadcastable):
            continue
        c = c.squeeze()

        [noisy] = client.outputs
        aliases = [noisy] + [
            transpose_client.outputs[0]
            for transpose_client, _ in fgraph.clients[noisy]
            if is_matrix_transpose(transpose_client.outputs[0])  # type: ignore[arg-type]
        ]
        consumers.extend(
            (consumer, c)
            for alias in aliases
            for consumer, idx in clients_through_expand_dims(fgraph, alias)  # type: ignore[arg-type]
            if idx == 0
            and isinstance(consumer.op, Blockwise)
            and isinstance(consumer.op.core_op, core_op_type)
        )

    if not consumers:
        return None
    return Ks, consumers


def kron_eigh(
    Ks: list[TensorVariable], dtype: str
) -> tuple[TensorVariable, TensorVariable]:
    """Return ``(Q, d)`` with ``kron(*Ks) == Q @ diag(d) @ Q.mT``, from eigh of each K.

    The factors are cast to ``dtype``, or to its real counterpart for a complex
    ``dtype``, so the decomposition runs at the precision of the consumer.
    """
    real_dtype = {"complex64": "float32", "complex128": "float64"}.get(dtype, dtype)
    ds, Qs = zip(*(eigh(K.astype(real_dtype)) for K in Ks), strict=True)
    Q = reduce(kron, Qs)
    d = reduce(lambda a, b: outer(a, b).ravel(), ds)
    return Q, d


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([OpPattern(DimShuffle, is_left_expanded_matrix_transpose=True)])
def useless_symmetric_transpose(fgraph, node):
    x = node.inputs[0]
    if getattr(x.tag, "symmetric", False) or check_assumption(fgraph, x, SYMMETRIC):
        return [atleast_Nd(x, n=node.outputs[0].type.ndim)]
