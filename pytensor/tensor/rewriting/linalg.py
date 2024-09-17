import logging
from collections.abc import Callable
from typing import cast

from pytensor import Variable
from pytensor import tensor as pt
from pytensor.graph import Apply, FunctionGraph
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    node_rewriter,
)
from pytensor.scalar.basic import Mul
from pytensor.tensor.basic import (
    AllocDiag,
    Eye,
    TensorVariable,
    diagonal,
)
from pytensor.tensor.blas import Dot22
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Dot, Prod, _matrix_matrix_matmul, log, prod
from pytensor.tensor.nlinalg import (
    SVD,
    KroneckerProduct,
    MatrixInverse,
    MatrixPinv,
    det,
    inv,
    kron,
    pinv,
    svd,
)
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.slinalg import (
    BlockDiagonal,
    Cholesky,
    Solve,
    SolveBase,
    block_diag,
    cholesky,
    solve,
    solve_triangular,
)


logger = logging.getLogger(__name__)
ALL_INVERSE_OPS = (MatrixInverse, MatrixPinv)


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
        transpose_order = (*range(ndims - 2), ndims - 1, ndims - 2)
        return cast(bool, node.op.new_order == transpose_order)
    return False


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
    if isinstance(node.op, Dot | Dot22):
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
                return [solve(x, (l.mT)).mT]
            else:
                return [solve((x.mT), (l.mT)).mT]


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


@register_specialize
@node_rewriter([Blockwise])
def batched_vector_b_solve_to_matrix_b_solve(fgraph, node):
    """Replace a batched Solve(a, b, b_ndim=1) by Solve(a, b.T, b_ndim=2).T

    `a` must have no batched dimensions, while `b` can have arbitrary batched dimensions.
    """
    core_op = node.op.core_op

    if not isinstance(core_op, SolveBase):
        return None

    if node.op.core_op.b_ndim != 1:
        return None

    [a, b] = node.inputs

    # Check `b` is actually batched
    if b.type.ndim == 1:
        return None

    # Check `a` is a matrix (possibly with degenerate dims on the left)
    a_bcast_batch_dims = a.type.broadcastable[:-2]
    if not all(a_bcast_batch_dims):
        return None
    # We squeeze degenerate dims, any that are still needed will be introduced by the new_solve
    elif len(a_bcast_batch_dims):
        a = a.squeeze(axis=tuple(range(len(a_bcast_batch_dims))))

    # Recreate solve Op with b_ndim=2
    props = core_op._props_dict()
    props["b_ndim"] = 2
    new_core_op = type(core_op)(**props)
    matrix_b_solve = Blockwise(new_core_op)

    # Ravel any batched dims
    original_b_shape = tuple(b.shape)
    if len(original_b_shape) > 2:
        b = b.reshape((-1, original_b_shape[-1]))

    # Apply the rewrite
    new_solve = matrix_b_solve(a, b.T).T

    # Unravel any batched dims
    if len(original_b_shape) > 2:
        new_solve = new_solve.reshape(original_b_shape)

    old_solve = node.outputs[0]
    copy_stack_trace(old_solve, new_solve)

    return [new_solve]


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
            Li_b = solve_triangular(L, b, lower=True, b_ndim=2)
            x = solve_triangular((L.mT), Li_b, lower=False, b_ndim=2)
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
                isinstance(A.owner.op, Dot | Dot22)
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


@register_specialize
@node_rewriter([Blockwise])
def local_lift_through_linalg(
    fgraph: FunctionGraph, node: Apply
) -> list[Variable] | None:
    """
    Rewrite compositions of linear algebra operations by lifting expensive operations (Cholesky, Inverse) through Ops
    that join matrices (KroneckerProduct, BlockDiagonal).

    This rewrite takes advantage of commutation between certain linear algebra operations to do several smaller matrix
    operations on component matrices instead of one large one. For example, when taking the inverse of Kronecker
    product, we can take the inverse of each component matrix and then take the Kronecker product of the inverses. This
    reduces the cost of the inverse from O((n*m)^3) to O(n^3 + m^3) where n and m are the dimensions of the component
    matrices.

    Parameters
    ----------
    fgraph: FunctionGraph
        Function graph being optimized
    node: Apply
        Node of the function graph to be optimized

    Returns
    -------
    list of Variable, optional
        List of optimized variables, or None if no optimization was performed
    """

    # TODO: Simplify this if we end up Blockwising KroneckerProduct
    if not isinstance(node.op.core_op, MatrixInverse | Cholesky | MatrixPinv):
        return None

    y = node.inputs[0]
    outer_op = node.op

    if y.owner and (
        isinstance(y.owner.op, Blockwise)
        and isinstance(y.owner.op.core_op, BlockDiagonal)
        or isinstance(y.owner.op, KroneckerProduct)
    ):
        input_matrices = y.owner.inputs

        if isinstance(outer_op.core_op, MatrixInverse):
            outer_f = cast(Callable, inv)
        elif isinstance(outer_op.core_op, Cholesky):
            outer_f = cast(Callable, cholesky)
        elif isinstance(outer_op.core_op, MatrixPinv):
            outer_f = cast(Callable, pinv)
        else:
            raise NotImplementedError  # pragma: no cover

        inner_matrices = [cast(TensorVariable, outer_f(m)) for m in input_matrices]

        if isinstance(y.owner.op, KroneckerProduct):
            return [kron(*inner_matrices)]
        elif isinstance(y.owner.op.core_op, BlockDiagonal):
            return [block_diag(*inner_matrices)]
        else:
            raise NotImplementedError  # pragma: no cover
    return None


def _find_diag_from_eye_mul(potential_mul_input):
    # Check if the op is Elemwise and mul
    if not (
        potential_mul_input.owner is not None
        and isinstance(potential_mul_input.owner.op, Elemwise)
        and isinstance(potential_mul_input.owner.op.scalar_op, Mul)
    ):
        return None

    # Find whether any of the inputs to mul is Eye
    inputs_to_mul = potential_mul_input.owner.inputs
    eye_input = [
        mul_input
        for mul_input in inputs_to_mul
        if mul_input.owner
        and (
            isinstance(mul_input.owner.op, Eye)
            or
            # This whole condition checks if there is an Eye hiding inside a DimShuffle.
            # This arises from batched elementwise multiplication between a tensor and an eye, e.g.:
            # tensor(shape=(None, 3, 3) * eye(3). This is still potentially valid for diag rewrites.
            (
                isinstance(mul_input.owner.op, DimShuffle)
                and (
                    mul_input.owner.op.is_left_expand_dims
                    or mul_input.owner.op.is_right_expand_dims
                )
                and mul_input.owner.inputs[0].owner is not None
                and isinstance(mul_input.owner.inputs[0].owner.op, Eye)
            )
        )
    ]

    if not eye_input:
        return None

    eye_input = eye_input[0]
    # If eye_input is an Eye Op (it's not wrapped in a DimShuffle), check it doesn't have an offset
    if isinstance(eye_input.owner.op, Eye) and (
        not Eye.is_offset_zero(eye_input.owner)
        or eye_input.broadcastable[-2:] != (False, False)
    ):
        return None

    # Otherwise, an Eye was found but it is wrapped in a DimShuffle (i.e. there was some broadcasting going on).
    # We have to look inside DimShuffle to decide if the rewrite can be applied
    if isinstance(eye_input.owner.op, DimShuffle) and (
        eye_input.owner.op.is_left_expand_dims
        or eye_input.owner.op.is_right_expand_dims
    ):
        inner_eye = eye_input.owner.inputs[0]
        # We can only rewrite when the Eye is on the main diagonal (the offset is zero) and the identity isn't
        # degenerate
        if not Eye.is_offset_zero(inner_eye.owner) or inner_eye.broadcastable[-2:] != (
            False,
            False,
        ):
            return None

    # Get all non Eye inputs (scalars/matrices/vectors)
    non_eye_inputs = list(set(inputs_to_mul) - {eye_input})
    return eye_input, non_eye_inputs


@register_canonicalize("shape_unsafe")
@register_stabilize("shape_unsafe")
@node_rewriter([det])
def rewrite_det_diag_to_prod_diag(fgraph, node):
    """
     This rewrite takes advantage of the fact that for a diagonal matrix, the determinant value is the product of its
     diagonal elements.

    The presence of a diagonal matrix is detected by inspecting the graph. This rewrite can identify diagonal matrices
    that arise as the result of elementwise multiplication with an identity matrix. Specialized computation is used to
    make this rewrite as efficient as possible, depending on whether the multiplication was with a scalar,
    vector or a matrix.

    Parameters
    ----------
    fgraph: FunctionGraph
        Function graph being optimized
    node: Apply
        Node of the function graph to be optimized

    Returns
    -------
    list of Variable, optional
        List of optimized variables, or None if no optimization was performed
    """
    inputs = node.inputs[0]

    # Check for use of pt.diag first
    if (
        inputs.owner
        and isinstance(inputs.owner.op, AllocDiag)
        and AllocDiag.is_offset_zero(inputs.owner)
    ):
        diag_input = inputs.owner.inputs[0]
        det_val = diag_input.prod(axis=-1)
        return [det_val]

    # Check if the input is an elemwise multiply with identity matrix -- this also results in a diagonal matrix
    inputs_or_none = _find_diag_from_eye_mul(inputs)
    if inputs_or_none is None:
        return None

    eye_input, non_eye_inputs = inputs_or_none

    # Dealing with only one other input
    if len(non_eye_inputs) != 1:
        return None

    eye_input, non_eye_input = eye_input[0], non_eye_inputs[0]

    # Checking if original x was scalar/vector/matrix
    if non_eye_input.type.broadcastable[-2:] == (True, True):
        # For scalar
        det_val = non_eye_input.squeeze(axis=(-1, -2)) ** (eye_input.shape[0])
    elif non_eye_input.type.broadcastable[-2:] == (False, False):
        # For Matrix
        det_val = non_eye_input.diagonal(axis1=-1, axis2=-2).prod(axis=-1)
    else:
        # For vector
        det_val = non_eye_input.prod(axis=(-1, -2))
    det_val = det_val.astype(node.outputs[0].type.dtype)
    return [det_val]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([Blockwise])
def svd_uv_merge(fgraph, node):
    """If we have more than one `SVD` `Op`s and at least one has keyword argument
    `compute_uv=True`, then we can change `compute_uv = False` to `True` everywhere
    and allow `pytensor` to re-use the decomposition outputs instead of recomputing.
    """
    if not isinstance(node.op.core_op, SVD):
        return

    (x,) = node.inputs

    if node.op.core_op.compute_uv:
        # compute_uv=True returns [u, s, v].
        # if at least u or v is used, no need to rewrite this node.
        if (
            len(fgraph.clients[node.outputs[0]]) > 0
            or len(fgraph.clients[node.outputs[2]]) > 0
        ):
            return

        # Else, has to replace the s of this node with s of an SVD Op that compute_uv=False.
        # First, iterate to see if there is an SVD Op that can be reused.
        for cl, _ in fgraph.clients[x]:
            if isinstance(cl.op, Blockwise) and isinstance(cl.op.core_op, SVD):
                if not cl.op.core_op.compute_uv:
                    return {
                        node.outputs[1]: cl.outputs[0],
                    }

        # If no SVD reusable, return a new one.
        return {
            node.outputs[1]: svd(
                x, full_matrices=node.op.core_op.full_matrices, compute_uv=False
            ),
        }

    else:
        # compute_uv=False returns [s].
        # We want rewrite if there is another one with compute_uv=True.
        # For this case, just reuse the `s` from the one with compute_uv=True.
        for cl, _ in fgraph.clients[x]:
            if isinstance(cl.op, Blockwise) and isinstance(cl.op.core_op, SVD):
                if cl.op.core_op.compute_uv and (
                    len(fgraph.clients[cl.outputs[0]]) > 0
                    or len(fgraph.clients[cl.outputs[2]]) > 0
                ):
                    return [cl.outputs[1]]


@register_canonicalize
@register_stabilize
@node_rewriter([Blockwise])
def rewrite_inv_inv(fgraph, node):
    """
    This rewrite takes advantage of the fact that if there are two consecutive inverse operations (inv(inv(input))), we get back our original input without having to compute inverse once.

    Here, we check for direct inverse operations (inv/pinv)  and allows for any combination of these "inverse" nodes to be simply rewritten.

    Parameters
    ----------
    fgraph: FunctionGraph
        Function graph being optimized
    node: Apply
        Node of the function graph to be optimized

    Returns
    -------
    list of Variable, optional
        List of optimized variables, or None if no optimization was performed
    """
    # Check if its a valid inverse operation (either inv/pinv)
    # In case the outer operation is an inverse, it directly goes to the next step of finding inner operation
    # If the outer operation is not a valid inverse, we do not apply this rewrite
    if not isinstance(node.op.core_op, ALL_INVERSE_OPS):
        return None

    potential_inner_inv = node.inputs[0].owner
    if potential_inner_inv is None or potential_inner_inv.op is None:
        return None

    # Check if inner op is blockwise and and possible inv
    if not (
        potential_inner_inv
        and isinstance(potential_inner_inv.op, Blockwise)
        and isinstance(potential_inner_inv.op.core_op, ALL_INVERSE_OPS)
    ):
        return None
    return [potential_inner_inv.inputs[0]]


@register_canonicalize
@register_stabilize
@node_rewriter([Blockwise])
def rewrite_inv_eye_to_eye(fgraph, node):
    """
     This rewrite takes advantage of the fact that the inverse of an identity matrix is the matrix itself
    The presence of an identity matrix is identified by checking whether we have k = 0 for an Eye Op inside an inverse op.
    Parameters
    ----------
    fgraph: FunctionGraph
        Function graph being optimized
    node: Apply
        Node of the function graph to be optimized
    Returns
    -------
    list of Variable, optional
        List of optimized variables, or None if no optimization was performed
    """
    core_op = node.op.core_op
    if not (isinstance(core_op, ALL_INVERSE_OPS)):
        return None

    # Check whether input to inverse is Eye and the 1's are on main diagonal
    potential_eye = node.inputs[0]
    if not (
        potential_eye.owner
        and isinstance(potential_eye.owner.op, Eye)
        and getattr(potential_eye.owner.inputs[-1], "data", -1).item() == 0
    ):
        return None
    return [potential_eye]


@register_canonicalize
@register_stabilize
@node_rewriter([Blockwise])
def rewrite_inv_diag_to_diag_reciprocal(fgraph, node):
    """
     This rewrite takes advantage of the fact that for a diagonal matrix, the inverse is a diagonal matrix with the new diagonal entries as reciprocals of the original diagonal elements.
     This function deals with diagonal matrix arising from the multiplicaton of eye with a scalar/vector/matrix

    Parameters
    ----------
    fgraph: FunctionGraph
        Function graph being optimized
    node: Apply
        Node of the function graph to be optimized

    Returns
    -------
    list of Variable, optional
        List of optimized variables, or None if no optimization was performed
    """
    core_op = node.op.core_op
    if not (isinstance(core_op, ALL_INVERSE_OPS)):
        return None

    inputs = node.inputs[0]
    # Check for use of pt.diag first
    if (
        inputs.owner
        and isinstance(inputs.owner.op, AllocDiag)
        and AllocDiag.is_offset_zero(inputs.owner)
    ):
        inv_input = inputs.owner.inputs[0]
        inv_val = pt.diag(1 / inv_input)
        return [inv_val]

    # Check if the input is an elemwise multiply with identity matrix -- this also results in a diagonal matrix
    inputs_or_none = _find_diag_from_eye_mul(inputs)
    if inputs_or_none is None:
        return None

    eye_input, non_eye_inputs = inputs_or_none

    # Dealing with only one other input
    if len(non_eye_inputs) != 1:
        return None

    non_eye_input = non_eye_inputs[0]

    # For a matrix, we have to first extract the diagonal (non-zero values) and then only use those
    if non_eye_input.type.broadcastable[-2:] == (False, False):
        non_eye_diag = non_eye_input.diagonal(axis1=-1, axis2=-2)
        non_eye_input = pt.shape_padaxis(non_eye_diag, -2)

    return [eye_input / non_eye_input]
