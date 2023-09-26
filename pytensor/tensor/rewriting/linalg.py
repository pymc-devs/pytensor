import logging
from typing import cast

from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.basic import TensorVariable, diagonal, swapaxes
from pytensor.tensor.blas import Dot22
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.math import Dot, Prod, _matrix_matrix_matmul, log, prod
from pytensor.tensor.nlinalg import MatrixInverse, det
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.slinalg import Cholesky, Solve, cholesky, solve, solve_triangular


logger = logging.getLogger(__name__)


def is_matrix_transpose(x: TensorVariable) -> bool:
    """Check if a variable corresponds to a transpose of the last two axes"""
    node = x.owner
    if (
        node
        and isinstance(node.op, DimShuffle)
        and not (node.op.drop or node.op.augment)
    ):
        [inp] = node.inputs
        ndims = inp.type.ndim
        if ndims < 2:
            return False
        transpose_order = tuple(range(ndims - 2)) + (ndims - 1, ndims - 2)
        return cast(bool, node.op.new_order == transpose_order)
    return False


def _T(x: TensorVariable) -> TensorVariable:
    """Matrix transpose for potentially higher dimensionality tensors"""
    return swapaxes(x, -1, -2)


@register_canonicalize
@node_rewriter([DimShuffle])
def transinv_to_invtrans(fgraph, node):
    if is_matrix_transpose(node.outputs[0]):
        (A,) = node.inputs
        if (
            A.owner
            and isinstance(A.owner.op, Blockwise)
            and isinstance(A.owner.op.core_op, MatrixInverse)
        ):
            (X,) = A.owner.inputs
            return [A.owner.op(node.op(X))]


@register_stabilize
@node_rewriter([Dot, Dot22])
def inv_as_solve(fgraph, node):
    """
    This utilizes a boolean `symmetric` tag on the matrices.
    """
    if isinstance(node.op, (Dot, Dot22)):
        l, r = node.inputs
        if (
            l.owner
            and isinstance(l.owner.op, Blockwise)
            and isinstance(l.owner.op.core_op, MatrixInverse)
        ):
            return [solve(l.owner.inputs[0], r)]
        if (
            r.owner
            and isinstance(r.owner.op, Blockwise)
            and isinstance(r.owner.op.core_op, MatrixInverse)
        ):
            x = r.owner.inputs[0]
            if getattr(x.tag, "symmetric", None) is True:
                return [_T(solve(x, _T(l)))]
            else:
                return [_T(solve(_T(x), _T(l)))]


@register_stabilize
@register_canonicalize
@node_rewriter([Blockwise])
def generic_solve_to_solve_triangular(fgraph, node):
    """
    If any solve() is applied to the output of a cholesky op, then
    replace it with a triangular solve.

    """
    if isinstance(node.op.core_op, Solve):
        if node.op.core_op.assume_a == "gen":
            A, b = node.inputs  # result is solution Ax=b
            if (
                A.owner
                and isinstance(A.owner.op, Blockwise)
                and isinstance(A.owner.op.core_op, Cholesky)
            ):
                if A.owner.op.core_op.lower:
                    return [
                        solve_triangular(
                            A, b, lower=True, b_ndim=node.op.core_op.b_ndim
                        )
                    ]
                else:
                    return [
                        solve_triangular(
                            A, b, lower=False, b_ndim=node.op.core_op.b_ndim
                        )
                    ]
            if is_matrix_transpose(A):
                (A_T,) = A.owner.inputs
                if (
                    A_T.owner
                    and isinstance(A_T.owner.op, Blockwise)
                    and isinstance(A_T.owner.op, Cholesky)
                ):
                    if A_T.owner.op.lower:
                        return [
                            solve_triangular(
                                A, b, lower=False, b_ndim=node.op.core_op.b_ndim
                            )
                        ]
                    else:
                        return [
                            solve_triangular(
                                A, b, lower=True, b_ndim=node.op.core_op.b_ndim
                            )
                        ]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([DimShuffle])
def no_transpose_symmetric(fgraph, node):
    if is_matrix_transpose(node.outputs[0]):
        x = node.inputs[0]
        if getattr(x.tag, "symmetric", None):
            return [x]


@register_stabilize
@node_rewriter([Blockwise])
def psd_solve_with_chol(fgraph, node):
    """
    This utilizes a boolean `psd` tag on matrices.
    """
    if isinstance(node.op.core_op, Solve) and node.op.core_op.b_ndim == 2:
        A, b = node.inputs  # result is solution Ax=b
        if getattr(A.tag, "psd", None) is True:
            L = cholesky(A)
            # N.B. this can be further reduced to a yet-unwritten cho_solve Op
            #     __if__ no other Op makes use of the L matrix during the
            #     stabilization
            Li_b = solve(L, b, assume_a="sym", lower=True, b_ndim=2)
            x = solve(_T(L), Li_b, assume_a="sym", lower=False, b_ndim=2)
            return [x]


@register_canonicalize
@register_stabilize
@node_rewriter([Blockwise])
def cholesky_ldotlt(fgraph, node):
    """
    rewrite cholesky(dot(L, L.T), lower=True) = L, where L is lower triangular,
    or cholesky(dot(U.T, U), upper=True) = U where U is upper triangular.

    Also works with matmul.

    This utilizes a boolean `lower_triangular` or `upper_triangular` tag on matrices.
    """
    if not isinstance(node.op.core_op, Cholesky):
        return

    A = node.inputs[0]
    if not (
        A.owner is not None
        and (
            (
                isinstance(A.owner.op, (Dot, Dot22))
                # This rewrite only applies to matrix Dot
                and A.owner.inputs[0].type.ndim == 2
            )
            or (A.owner.op == _matrix_matrix_matmul)
        )
    ):
        return

    l, r = A.owner.inputs

    # cholesky(dot(L,L.T)) case
    if (
        getattr(l.tag, "lower_triangular", False)
        and is_matrix_transpose(r)
        and r.owner.inputs[0] == l
    ):
        if node.op.core_op.lower:
            return [l]
        return [r]

    # cholesky(dot(U.T,U)) case
    if (
        getattr(r.tag, "upper_triangular", False)
        and is_matrix_transpose(l)
        and l.owner.inputs[0] == r
    ):
        if node.op.core_op.lower:
            return [l]
        return [r]


@register_stabilize
@register_specialize
@node_rewriter([det])
def local_det_chol(fgraph, node):
    """
    If we have det(X) and there is already an L=cholesky(X)
    floating around, then we can use prod(diag(L)) to get the determinant.

    """
    (x,) = node.inputs
    for cl, xpos in fgraph.clients[x]:
        if cl == "output":
            continue
        if isinstance(cl.op, Blockwise) and isinstance(cl.op.core_op, Cholesky):
            L = cl.outputs[0]
            return [prod(diagonal(L, axis1=-2, axis2=-1) ** 2, axis=-1)]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([log])
def local_log_prod_sqr(fgraph, node):
    """
    This utilizes a boolean `positive` tag on matrices.
    """
    (x,) = node.inputs
    if x.owner and isinstance(x.owner.op, Prod):
        # we cannot always make this substitution because
        # the prod might include negative terms
        p = x.owner.inputs[0]

        # p is the matrix we're reducing with prod
        if getattr(p.tag, "positive", None) is True:
            return [log(p).sum(axis=x.owner.op.axis)]

        # TODO: have a reduction like prod and sum that simply
        # returns the sign of the prod multiplication.
