import numpy as np

from pytensor import tensor as pt
from pytensor.assumptions import (
    LOWER_TRIANGULAR,
    UPPER_TRIANGULAR,
    check_assumption,
)
from pytensor.compile.mode import optdb
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    dfs_rewriter,
    node_rewriter,
)
from pytensor.scalar.basic import Abs, Log, Sign, Sqr
from pytensor.tensor.basic import ones
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.lu import LU, LUFactor
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.svd import SVD
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.summary import SLogDet, det
from pytensor.tensor.math import Prod, log, prod, sqr
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.linalg.utils import (
    lu_pivots_to_det_sign,
    matrix_diagonal_product,
    permutation_indices_to_det_sign,
)
from pytensor.tensor.subtensor import _is_provably_non_negative


@register_stabilize
@register_specialize
@node_rewriter([log])
def local_log_prod_to_sum_log(fgraph, node):
    """log(prod(x)) -> sum(log(x)), if x >= 0.

    Also log(abs(prod(x))) -> sum(log(abs(x))).
    """
    [p] = node.inputs
    match p.owner_op_and_inputs:
        case (Prod(axis=axis), x):
            # TODO: have a reduction like prod and sum that simply
            # returns the sign of the prod multiplication.

            # TODO: The product of diagonals of a Cholesky(A) are also strictly positive
            if _is_provably_non_negative(x):
                return [log(x).sum(axis=axis)]

        # Special case for log(abs(prod(x))) -> sum(log(abs(x))) that shows up in slogdet
        case (Elemwise(Abs()), p):
            match p.owner_op_and_inputs:
                case (Prod(axis=axis), x):
                    return [log(abs(x)).sum(axis=axis)]


@register_stabilize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter([det])
def det_of_matrix_factorized_elsewhere(fgraph, node):
    """det(X) -> p * prod(diag(U)), where p is the existing LU permutation's sign.

    For an existing Cholesky factor L, det(X) -> sqr(prod(diag(L))).
    Magnitude-only factorizations require consumers that discard the sign.
    """
    [det] = node.outputs
    [x] = node.inputs

    sign_not_needed = all(
        isinstance(client.op, Elemwise) and isinstance(client.op.scalar_op, (Abs, Sqr))
        for client, _ in fgraph.clients[det]
    )

    new_det = None
    for client, _ in fgraph.clients[x]:
        core_op = client.op.core_op if isinstance(client.op, Blockwise) else client.op
        match core_op:
            case Cholesky():
                L = client.outputs[0]
                factor_diag = pt.diagonal(L, axis1=-2, axis2=-1)
                new_det = sqr(prod(factor_diag, axis=-1))
            case LUFactor():
                LU_packed, pivots = client.outputs
                factor_diag = pt.diagonal(LU_packed, axis1=-2, axis2=-1)
                permutation_sign = lu_pivots_to_det_sign(pivots, x.shape[-1]).astype(
                    LU_packed.dtype
                )
                new_det = permutation_sign * prod(factor_diag, axis=-1)
            case LU(permute_l=False, p_indices=p_indices):
                P_or_indices, _, U = client.outputs
                factor_diag = pt.diagonal(U, axis1=-2, axis2=-1)
                indices = (
                    P_or_indices if p_indices else pt.argmax(P_or_indices, axis=-1)
                )
                permutation_sign = permutation_indices_to_det_sign(indices).astype(
                    U.dtype
                )
                new_det = permutation_sign * prod(factor_diag, axis=-1)
            case _:
                if not sign_not_needed:
                    continue
                match core_op:
                    case LU():
                        U = client.outputs[-1]
                        new_det = matrix_diagonal_product(U)
                    case SVD():
                        lmbda = (
                            client.outputs[1]
                            if core_op.compute_uv
                            else client.outputs[0]
                        )
                        new_det = prod(lmbda, axis=-1)
                    case QR():
                        R = client.outputs[-1]
                        # if mode == "economic", R may not be square and this rewrite could hide a shape error
                        # That's why it's tagged as `shape_unsafe`
                        new_det = matrix_diagonal_product(R)

        if new_det is not None:
            # found a match
            break
    else:  # no-break (i.e., no-match)
        return None

    copy_stack_trace(det, new_det)
    return [new_det]


@register_stabilize("shape_unsafe")
@register_specialize("shape_unsafe")
@node_rewriter(tracks=[det])
def det_of_factorized_matrix(fgraph, node):
    """Introduce special forms for det(decomposition(X)).

    Some cases are only known up to a sign change such as det(QR(X)),
    and are only introduced if the determinant sign is discarded downstream (e.g., abs, sqr)
    """
    [det] = node.outputs
    [x] = node.inputs

    sign_not_needed = all(
        isinstance(client.op, Elemwise) and isinstance(client.op.scalar_op, (Abs, Sqr))
        for client, _ in fgraph.clients[det]
    )

    x_node = x.owner
    if x_node is None:
        return None

    x_op = x_node.op
    core_op = x_op.core_op if isinstance(x_op, Blockwise) else x_op

    new_det = None
    match core_op:
        case Cholesky():
            new_det = matrix_diagonal_product(x)
        case LU():
            if not core_op.permute_l and x is x_node.outputs[-2]:
                # x is L
                new_det = ones(x.shape[:-2], dtype=det.dtype)
            elif x is x_node.outputs[-1]:
                # x is U
                new_det = matrix_diagonal_product(x)
        case SVD():
            if not core_op.compute_uv or x is x_node.outputs[1]:
                # x is lambda
                new_det = prod(x, axis=-1)
            elif sign_not_needed:
                # x is either U or Vt and sign is discarded downstream
                new_det = ones(x.shape[:-2], dtype=det.dtype)
        case QR():
            # if mode == "economic", Q/R may not be square and this rewrite could hide a shape error
            # That's why it's tagged as `shape_unsafe`
            if x is x_node.outputs[-1]:
                # x is R
                new_det = matrix_diagonal_product(x)
            elif (
                sign_not_needed
                and core_op.mode in ("economic", "full")
                and x is x_node.outputs[0]
            ):
                # x is Q and sign is discarded downstream
                new_det = ones(x.shape[:-2], dtype=det.dtype)

    if new_det is None:
        return None

    copy_stack_trace(det, new_det)
    return [new_det]


@register_canonicalize("shape_unsafe")
@register_stabilize("shape_unsafe")
@node_rewriter([det])
def det_of_triangular(fgraph, node):
    """det(T) -> prod(diagonal(T)) when T is triangular (incl. diagonal)."""
    inp = node.inputs[0]

    if not (
        check_assumption(fgraph, inp, LOWER_TRIANGULAR)
        or check_assumption(fgraph, inp, UPPER_TRIANGULAR)
    ):
        return None

    det_val = matrix_diagonal_product(inp).astype(node.outputs[0].type.dtype)
    return [det_val]


@register_canonicalize
@register_stabilize
@node_rewriter([det])
def det_of_inv(fgraph, node):
    """Replace det(matrix_inverse(X)) with reciprocal(det(X))."""
    match node.inputs[0].owner_op_and_inputs:
        case (Blockwise(MatrixInverse()), X):
            return [1 / det(X)]


@node_rewriter([det])
def slogdet_specialization(fgraph, node):
    """sign(det(X)), log(abs(det(X))) -> the two outputs of SLogDet(X).

    Also replace log(det(X)) using sign/phase and logabsdet. Requires all
    determinant consumers to match, avoiding an additional factorization.
    """
    dummy_replacements = {}
    for client, _ in fgraph.clients[node.outputs[0]]:
        match (client.op, *client.outputs):
            # Check for sign(det)
            case (Elemwise(Sign()), sign):
                dummy_replacements[sign] = "sign"

            # Check for log(abs(det))
            case (Elemwise(Abs()), potential_log):
                for client_2, _ in fgraph.clients[potential_log]:
                    match (client_2.op, *client_2.outputs):
                        case (Elemwise(Log()), log_abs_det):
                            dummy_replacements[log_abs_det] = "log_abs_det"
                        case _:
                            return None

            case (Elemwise(Log()), log_det):
                dummy_replacements[log_det] = "log_det"

            case _:
                # Det is used directly for something else, don't rewrite to avoid computing two dets
                return None

    if not dummy_replacements:
        return None

    [x] = node.inputs
    sign_det_x, log_abs_det_x = Blockwise(SLogDet())(x)
    if x.dtype.startswith("complex"):
        log_det_x = log(sign_det_x) + log_abs_det_x
    else:
        log_det_x = pt.where(pt.eq(sign_det_x, -1), np.nan, log_abs_det_x)
    slogdet_specialization_map = {
        "sign": sign_det_x,
        "log_abs_det": log_abs_det_x,
        "log_det": log_det_x,
    }
    return {k: slogdet_specialization_map[v] for k, v in dummy_replacements.items()}


# Keep the naive determinant graph available to all specialization rewrites.
optdb.register(
    "slogdet_specialization",
    dfs_rewriter(slogdet_specialization),
    "fast_run",
    position=2.1,
)
