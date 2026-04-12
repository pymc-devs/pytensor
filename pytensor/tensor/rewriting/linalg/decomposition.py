from pytensor import tensor as pt
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.tensor.assumptions.diagonal import DIAGONAL
from pytensor.tensor.assumptions.utils import check_assumption
from pytensor.tensor.basic import alloc_diag
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.eigen import Eigh, Eigvalsh
from pytensor.tensor.linalg.decomposition.svd import SVD, svd
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(Cholesky)])
def cholesky_ldotlt(fgraph, node):
    """
    rewrite cholesky(dot(L, L.T), lower=True) = L, where L is lower triangular,
    or cholesky(dot(U.T, U), upper=True) = U where U is upper triangular.

    Also works with matmul.

    This utilizes a boolean `lower_triangular` or `upper_triangular` tag on matrices.
    """
    A = node.inputs[0]
    lower = node.op.core_op.lower

    match A.owner_op_and_inputs:
        case (Blockwise(Dot()) | Dot(), l, r):
            lower_triangular = getattr(l.tag, "lower_triangular", False)
            match (lower_triangular, r.owner_op_and_inputs):
                # cholesky(dot(L,L.T)) case
                case (
                    True,
                    (DimShuffle(is_left_expanded_matrix_transpose=True), l_T),
                ) if l_T == l:
                    return [l] if lower else [r]

            upper_triangular = getattr(r.tag, "upper_triangular", False)
            match (upper_triangular, l.owner_op_and_inputs):
                # cholesky(dot(U.T,U)) case
                case (
                    True,
                    (DimShuffle(is_left_expanded_matrix_transpose=True), r_T),
                ) if r_T == r:
                    return [l] if lower else [r]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(Cholesky)])
def cholesky_of_diag(fgraph, node):
    """cholesky(D) -> diag(sqrt(diagonal(D))) for diagonal D."""
    [X] = node.inputs

    if all(X.type.broadcastable[-2:]):
        return [pt.sqrt(X)]

    if not check_assumption(fgraph, X, DIAGONAL):
        return None

    diag_vals = pt.sqrt(pt.diagonal(X, axis1=-2, axis2=-1))
    return [alloc_diag(diag_vals, axis1=-2, axis2=-1)]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([blockwise_of(SVD)])
def svd_uv_merge(fgraph, node):
    """If we have more than one `SVD` `Op`s and at least one has keyword argument
    `compute_uv=True`, then we can change `compute_uv = False` to `True` everywhere
    and allow `pytensor` to re-use the decomposition outputs instead of recomputing.
    """
    [x] = node.inputs

    if node.op.core_op.compute_uv:
        # compute_uv=True returns [u, s, v].
        u, s, v = node.outputs

        # if at least u or v is used, no need to rewrite this node.
        if fgraph.clients[u] or fgraph.clients[v]:
            return None

        # Else, has to replace the s of this node with s of an SVD Op that compute_uv=False.
        # First, iterate to see if there is an SVD Op that can be reused.
        for cl, _ in fgraph.clients[x]:
            if cl is node:
                continue
            match (cl.op, *cl.outputs):
                case (Blockwise(SVD(compute_uv=False)), replacement_s):
                    break
        else:
            # If no SVD reusable, return a new one.
            replacement_s = svd(
                x,
                full_matrices=node.op.core_op.full_matrices,
                compute_uv=False,
            )
        return {s: replacement_s}

    else:
        # compute_uv=False returns [s].
        # We want rewrite if there is another one with compute_uv=True.
        # For this case, just reuse the `s` from the one with compute_uv=True.
        for cl, _ in fgraph.clients[x]:
            if cl is node:
                continue
            match (cl.op, *cl.outputs):
                case (Blockwise(SVD(compute_uv=True)), u, s, v) if (
                    fgraph.clients[u] or fgraph.clients[v]
                ):
                    return [s]


@register_canonicalize
@register_stabilize
@node_rewriter([Eigh])
def eigh_of_diag(fgraph, node):
    """eigh(D) -> (sort(diagonal(D)), eye permuted by argsort) for diagonal D."""
    [X] = node.inputs

    if not check_assumption(fgraph, X, DIAGONAL):
        return None

    w, v = node.outputs
    diag_vals = pt.diagonal(X, axis1=-2, axis2=-1)
    sort_idx = pt.argsort(diag_vals)

    new_w = diag_vals[sort_idx]
    new_v = pt.eye(X.shape[-1])[:, sort_idx]

    copy_stack_trace(w, new_w)
    copy_stack_trace(v, new_v)

    return [new_w, new_v]


@register_canonicalize
@register_stabilize
@node_rewriter([Eigvalsh])
def eigvalsh_of_diag(fgraph, node):
    """eigvalsh(D) -> sort(diagonal(D)) for diagonal D.

    Also handles the generalized case eigvalsh(D, B) when both are diagonal:
    sort(diagonal(D) / diagonal(B)).
    """
    if len(node.inputs) == 1:
        [X] = node.inputs
        if not check_assumption(fgraph, X, DIAGONAL):
            return None
        diag_vals = pt.diagonal(X, axis1=-2, axis2=-1)
        new_w = pt.sort(diag_vals)
    elif len(node.inputs) == 2:
        X, B = node.inputs
        if not check_assumption(fgraph, X, DIAGONAL):
            return None
        if not check_assumption(fgraph, B, DIAGONAL):
            return None
        diag_x = pt.diagonal(X, axis1=-2, axis2=-1)
        diag_b = pt.diagonal(B, axis1=-2, axis2=-1)
        new_w = pt.sort(diag_x / diag_b)
    else:
        return None

    copy_stack_trace(node.outputs[0], new_w)
    return [new_w]
