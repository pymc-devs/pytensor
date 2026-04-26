import numpy as np

from pytensor import tensor as pt
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    node_rewriter,
)
from pytensor.scalar.basic import Abs, Exp, Log, Sign, Sqr
from pytensor.tensor.assumptions.diagonal import DIAGONAL
from pytensor.tensor.assumptions.triangular import LOWER_TRIANGULAR, UPPER_TRIANGULAR
from pytensor.tensor.assumptions.utils import check_assumption
from pytensor.tensor.basic import ones
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.lu import LU, LUFactor
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.svd import SVD
from pytensor.tensor.linalg.summary import SLogDet, det
from pytensor.tensor.math import Prod, log, prod
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.linalg.utils import matrix_diagonal_product


@register_stabilize
@register_specialize
@node_rewriter([log])
def local_log_prod_to_sum_log(fgraph, node):
    """Rewrite log(prod(x)) as sum(log(x)), when x is known to be positive."""
    [p] = node.inputs
    match p.owner_op_and_inputs:
        case (Prod(axis=axis), x):
            # TODO: have a reduction like prod and sum that simply
            # returns the sign of the prod multiplication.

            # TODO: The product of diagonals of a Cholesky(A) are also strictly positive
            match x.owner_op:
                case Elemwise(Abs() | Sqr() | Exp()):
                    return [log(x).sum(axis=axis)]

            if getattr(x.tag, "positive", False):
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
    """
    If we have det(X) or abs(det(X)) and there is already a nice decomposition(X) floating around,
    use it to compute it more cheaply

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
                new_det = matrix_diagonal_product(L) ** 2
            case LU():
                U = client.outputs[-1]
                new_det = matrix_diagonal_product(U)
            case LUFactor():
                LU_packed = client.outputs[0]
                new_det = matrix_diagonal_product(LU_packed)
            case _:
                if not sign_not_needed:
                    continue
                match core_op:
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

    [det] = node.outputs
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
            if x is x_node.outputs[-2]:
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
def det_of_diag(fgraph, node):
    """det(D) -> prod(diagonal(D)) for diagonal D."""
    inp = node.inputs[0]

    if not check_assumption(fgraph, inp, DIAGONAL):
        return None

    det_val = pt.diagonal(inp, axis1=-2, axis2=-1).prod(axis=-1)
    det_val = det_val.astype(node.outputs[0].type.dtype)
    return [det_val]


@register_stabilize("shape_unsafe")
@node_rewriter([det])
def det_of_triangular(fgraph, node):
    """det(T) -> prod(diagonal(T)) for triangular T."""
    inp = node.inputs[0]

    if not (
        check_assumption(fgraph, inp, LOWER_TRIANGULAR)
        or check_assumption(fgraph, inp, UPPER_TRIANGULAR)
    ):
        return None

    det_val = matrix_diagonal_product(inp)
    det_val = det_val.astype(node.outputs[0].type.dtype)
    return [det_val]


@register_specialize
@node_rewriter([det])
def slogdet_specialization(fgraph, node):
    """
    This rewrite targets specific operations related to slogdet i.e sign(det), log(det) and log(abs(det)) and rewrites them using the SLogDet operation.

    Parameters
    ----------
    fgraph: FunctionGraph
        Function graph being optimized
    node: Apply
        Node of the function graph to be optimized

    Returns
    -------
    dictionary of Variables, optional
        Dictionary of nodes and what they should be replaced with, or None if no optimization was performed
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
    sign_det_x, log_abs_det_x = SLogDet()(x)
    log_det_x = pt.where(pt.eq(sign_det_x, -1), np.nan, log_abs_det_x)
    slogdet_specialization_map = {
        "sign": sign_det_x,
        "log_abs_det": log_abs_det_x,
        "log_det": log_det_x,
    }
    return {k: slogdet_specialization_map[v] for k, v in dummy_replacements.items()}
