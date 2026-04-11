from pytensor import tensor as pt
from pytensor.graph import Constant
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.basic import AllocDiag, Eye
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.svd import SVD, svd
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.rewriting.linalg.utils import is_eye_mul


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
    [X] = node.inputs

    # Check if input is a (1, 1) matrix
    if all(X.type.broadcastable[-2:]):
        return [pt.sqrt(X)]

    match X.owner_op_and_inputs:
        # Check whether input to Cholesky is Eye and the 1's are on main diagonal
        case (Eye(), _, _, Constant(0)):
            return [X]
        case (AllocDiag(offset=0, axis1=axis1, axis2=axis2), diag_input):
            ndim = diag_input.ndim
            if axis1 == ndim - 1 and axis2 == ndim:
                return [pt.diag(diag_input**0.5)]

    # Check if the input is an elemwise multiply with identity matrix -- this also results in a diagonal matrix
    match is_eye_mul(X):
        case (eye_input, non_eye_input):
            # Now, we can simply return the matrix consisting of sqrt values of the original diagonal elements
            # For a matrix, we have to first extract the diagonal (non-zero values) and then only use those
            if non_eye_input.type.broadcastable[-2:] == (False, False):
                non_eye_input = non_eye_input.diagonal(axis1=-1, axis2=-2)
                if eye_input.type.ndim > 2:
                    non_eye_input = pt.shape_padaxis(non_eye_input, -2)

            return [eye_input * (non_eye_input**0.5)]


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
