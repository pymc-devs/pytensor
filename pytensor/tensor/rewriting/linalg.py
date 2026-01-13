import logging
from collections.abc import Callable
from typing import cast

import numpy as np

from pytensor import Variable
from pytensor import tensor as pt
from pytensor.graph import Apply, FunctionGraph
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    node_rewriter,
)
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.scalar.basic import Abs, Exp, Log, Mul, Sign, Sqr
from pytensor.tensor.basic import (
    AllocDiag,
    ExtractDiag,
    Eye,
    TensorVariable,
    concatenate,
    diag,
    diagonal,
    ones,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Dot, Prod, _matmul, log, outer, prod
from pytensor.tensor.nlinalg import (
    SVD,
    KroneckerProduct,
    MatrixInverse,
    MatrixPinv,
    SLogDet,
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
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.slinalg import (
    LU,
    QR,
    BlockDiagonal,
    Cholesky,
    CholeskySolve,
    LUFactor,
    Solve,
    SolveBase,
    SolveTriangular,
    block_diag,
    cholesky,
    solve,
    solve_triangular,
)


logger = logging.getLogger(__name__)
MATRIX_INVERSE_OPS = (MatrixInverse, MatrixPinv)


def matrix_diagonal_product(x):
    return pt.prod(diagonal(x, axis1=-2, axis2=-1), axis=-1)


@register_canonicalize
@node_rewriter([BlockDiagonal])
def fuse_blockdiagonal(fgraph, node):
    """Fuse nested BlockDiagonal ops into a single BlockDiagonal."""

    new_inputs = []
    changed = False

    for inp in node.inputs:
        if inp.owner and isinstance(inp.owner.op, BlockDiagonal):
            new_inputs.extend(inp.owner.inputs)
            changed = True
        else:
            new_inputs.append(inp)

    if changed:
        fused_op = BlockDiagonal(len(new_inputs))
        new_output = fused_op(*new_inputs)
        copy_stack_trace(node.outputs[0], new_output)
        return [new_output]

    return None


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

        # Allow expand_dims on the left of the transpose
        if (diff := len(transpose_order) - len(node.op.new_order)) > 0:
            transpose_order = (
                *(["x"] * diff),
                *transpose_order,
            )
        return node.op.new_order == transpose_order
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
@node_rewriter([Dot])
def inv_as_solve(fgraph, node):
    """
    This utilizes a boolean `symmetric` tag on the matrices.
    """
    if isinstance(node.op, Dot):
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
@node_rewriter([blockwise_of(OpPattern(Solve, assume_a="gen"))])
def generic_solve_to_solve_triangular(fgraph, node):
    """
    If any solve() is applied to the output of a cholesky op, then
    replace it with a triangular solve.

    """
    A, b = node.inputs  # result is the solution to Ax=b
    if (
        A.owner
        and isinstance(A.owner.op, Blockwise)
        and isinstance(A.owner.op.core_op, Cholesky)
    ):
        if A.owner.op.core_op.lower:
            return [solve_triangular(A, b, lower=True, b_ndim=node.op.core_op.b_ndim)]
        else:
            return [solve_triangular(A, b, lower=False, b_ndim=node.op.core_op.b_ndim)]
    if is_matrix_transpose(A):
        (A_T,) = A.owner.inputs
        if (
            A_T.owner
            and isinstance(A_T.owner.op, Blockwise)
            and isinstance(A_T.owner.op, Cholesky)
        ):
            if A_T.owner.op.lower:
                return [
                    solve_triangular(A, b, lower=False, b_ndim=node.op.core_op.b_ndim)
                ]
            else:
                return [
                    solve_triangular(A, b, lower=True, b_ndim=node.op.core_op.b_ndim)
                ]


@register_specialize
@node_rewriter([blockwise_of(OpPattern(SolveBase, b_ndim=1))])
def batched_vector_b_solve_to_matrix_b_solve(fgraph, node):
    """Replace a batched Solve(a, b, b_ndim=1) by Solve(a, b.T, b_ndim=2).T

    `a` must have no batched dimensions, while `b` can have arbitrary batched dimensions.
    """
    core_op = node.op.core_op
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
@node_rewriter([blockwise_of(OpPattern(Solve, b_ndim=2))])
def psd_solve_with_chol(fgraph, node):
    """
    This utilizes a boolean `psd` tag on matrices.
    """
    A, b = node.inputs  # result is the solution to Ax=b
    if getattr(A.tag, "psd", None) is True:
        L = cholesky(A)
        # N.B. this can be further reduced to cho_solve Op
        #     if no other Op makes use of the L matrix
        Li_b = solve_triangular(L, b, lower=True, b_ndim=2)
        x = solve_triangular((L.mT), Li_b, lower=False, b_ndim=2)
        return [x]


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
    if not (
        A.owner is not None and (isinstance(A.owner.op, Dot) or (A.owner.op == _matmul))
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
@node_rewriter([log])
def local_log_prod_to_sum_log(fgraph, node):
    """Rewrite log(prod(x)) as sum(log(x)), when x is known to be positive."""
    [p] = node.inputs
    p_node = p.owner

    if p_node is None:
        return None

    p_op = p_node.op

    if isinstance(p_op, Prod):
        x = p_node.inputs[0]

        # TODO: The product of diagonals of a Cholesky(A) are also strictly positive
        if (
            x.owner is not None
            and isinstance(x.owner.op, Elemwise)
            and isinstance(x.owner.op.scalar_op, Abs | Sqr | Exp)
        ) or getattr(x.tag, "positive", False):
            return [log(x).sum(axis=p_node.op.axis)]

        # TODO: have a reduction like prod and sum that simply
        # returns the sign of the prod multiplication.

    # Special case for log(abs(prod(x))) -> sum(log(abs(x))) that shows up in slogdet
    elif isinstance(p_op, Elemwise) and isinstance(p_op.scalar_op, Abs):
        [p] = p_node.inputs
        p_node = p.owner
        if p_node is not None and isinstance(p_node.op, Prod):
            [x] = p.owner.inputs
            return [log(abs(x)).sum(axis=p_node.op.axis)]


@register_specialize
@node_rewriter([blockwise_of(MatrixInverse | Cholesky | MatrixPinv)])
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
    y = node.inputs[0]
    outer_op = node.op

    if y.owner and (
        (
            isinstance(y.owner.op, Blockwise)
            and isinstance(y.owner.op.core_op, BlockDiagonal)
        )
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
@node_rewriter([blockwise_of(SVD)])
def svd_uv_merge(fgraph, node):
    """If we have more than one `SVD` `Op`s and at least one has keyword argument
    `compute_uv=True`, then we can change `compute_uv = False` to `True` everywhere
    and allow `pytensor` to re-use the decomposition outputs instead of recomputing.
    """
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
@node_rewriter([blockwise_of(MATRIX_INVERSE_OPS)])
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
    potential_inner_inv = node.inputs[0].owner
    if potential_inner_inv is None or potential_inner_inv.op is None:
        return None

    # Check if inner op is blockwise and and possible inv
    if not (
        potential_inner_inv
        and isinstance(potential_inner_inv.op, Blockwise)
        and isinstance(potential_inner_inv.op.core_op, MATRIX_INVERSE_OPS)
    ):
        return None
    return [potential_inner_inv.inputs[0]]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(MATRIX_INVERSE_OPS)])
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
@node_rewriter([blockwise_of(MATRIX_INVERSE_OPS)])
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


@register_canonicalize
@register_stabilize
@node_rewriter([ExtractDiag])
def rewrite_diag_blockdiag(fgraph, node):
    """
    This rewrite simplifies extracting the diagonal of a blockdiagonal matrix by concatening the diagonal values of all of the individual sub matrices.

    diag(block_diag(a,b,c,....)) = concat(diag(a), diag(b), diag(c),...)

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
    # Check for inner block_diag operation
    potential_block_diag = node.inputs[0].owner
    if not (
        potential_block_diag
        and isinstance(potential_block_diag.op, Blockwise)
        and isinstance(potential_block_diag.op.core_op, BlockDiagonal)
    ):
        return None

    # Find the composing sub_matrices
    submatrices = potential_block_diag.inputs
    submatrices_diag = [diag(submatrices[i]) for i in range(len(submatrices))]

    return [concatenate(submatrices_diag)]


@register_canonicalize
@register_stabilize
@node_rewriter([det])
def rewrite_det_blockdiag(fgraph, node):
    """
    This rewrite simplifies the determinant of a blockdiagonal matrix by extracting the individual sub matrices and returning the product of all individual determinant values.

    det(block_diag(a,b,c,....)) = prod(det(a), det(b), det(c),...)

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
    # Check for inner block_diag operation
    potential_block_diag = node.inputs[0].owner
    if not (
        potential_block_diag
        and isinstance(potential_block_diag.op, Blockwise)
        and isinstance(potential_block_diag.op.core_op, BlockDiagonal)
    ):
        return None

    # Find the composing sub_matrices
    sub_matrices = potential_block_diag.inputs
    det_sub_matrices = [det(sub_matrices[i]) for i in range(len(sub_matrices))]

    return [prod(det_sub_matrices)]


@register_canonicalize
@register_stabilize
@node_rewriter([ExtractDiag])
def rewrite_diag_kronecker(fgraph, node):
    """
    This rewrite simplifies the diagonal of the kronecker product of 2 matrices by extracting the individual sub matrices and returning their outer product as a vector.

    diag(kron(a,b)) -> outer(diag(a), diag(b))

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
    # Check for inner kron operation
    potential_kron = node.inputs[0].owner
    if not (potential_kron and isinstance(potential_kron.op, KroneckerProduct)):
        return None

    # Find the matrices
    a, b = potential_kron.inputs
    diag_a, diag_b = diag(a), diag(b)
    outer_prod_as_vector = outer(diag_a, diag_b).flatten()

    return [outer_prod_as_vector]


@register_canonicalize
@register_stabilize
@node_rewriter([det])
def rewrite_det_kronecker(fgraph, node):
    """
    This rewrite simplifies the determinant of a kronecker-structured matrix by extracting the individual sub matrices and returning the det values computed using those

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
    # Check for inner kron operation
    potential_kron = node.inputs[0].owner
    if not (potential_kron and isinstance(potential_kron.op, KroneckerProduct)):
        return None

    # Find the matrices
    a, b = potential_kron.inputs
    dets = [det(a), det(b)]
    sizes = [a.shape[-1], b.shape[-1]]
    prod_sizes = prod(sizes, no_zeros_in_input=True)
    det_final = prod([dets[i] ** (prod_sizes / sizes[i]) for i in range(2)])

    return [det_final]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(Cholesky)])
def rewrite_remove_useless_cholesky(fgraph, node):
    """
     This rewrite takes advantage of the fact that the cholesky decomposition of an identity matrix is the matrix itself

    The presence of an identity matrix is identified by checking whether we have k = 0 for an Eye Op inside Cholesky.

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
    # Find whether cholesky op is being applied

    # Check whether input to Cholesky is Eye and the 1's are on main diagonal
    potential_eye = node.inputs[0]
    if not (
        potential_eye.owner
        and isinstance(potential_eye.owner.op, Eye)
        and hasattr(potential_eye.owner.inputs[-1], "data")
        and potential_eye.owner.inputs[-1].data.item() == 0
    ):
        return None
    return [potential_eye]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(Cholesky)])
def rewrite_cholesky_diag_to_sqrt_diag(fgraph, node):
    [input] = node.inputs

    # Check if input is a (1, 1) matrix
    if all(input.type.broadcastable[-2:]):
        return [pt.sqrt(input)]

    # Check for use of pt.diag first
    if (
        input.owner
        and isinstance(input.owner.op, AllocDiag)
        and AllocDiag.is_offset_zero(input.owner)
    ):
        diag_input = input.owner.inputs[0]
        cholesky_val = pt.diag(diag_input**0.5)
        return [cholesky_val]

    # Check if the input is an elemwise multiply with identity matrix -- this also results in a diagonal matrix
    inputs_or_none = _find_diag_from_eye_mul(input)
    if inputs_or_none is None:
        return None

    eye_input, non_eye_inputs = inputs_or_none

    # Dealing with only one other input
    if len(non_eye_inputs) != 1:
        return None

    [non_eye_input] = non_eye_inputs

    # Now, we can simply return the matrix consisting of sqrt values of the original diagonal elements
    # For a matrix, we have to first extract the diagonal (non-zero values) and then only use those
    if non_eye_input.type.broadcastable[-2:] == (False, False):
        non_eye_input = non_eye_input.diagonal(axis1=-1, axis2=-2)
        if eye_input.type.ndim > 2:
            non_eye_input = pt.shape_padaxis(non_eye_input, -2)

    return [eye_input * (non_eye_input**0.5)]


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
        # Check for sign(det)
        if isinstance(client.op, Elemwise) and isinstance(client.op.scalar_op, Sign):
            dummy_replacements[client.outputs[0]] = "sign"

        # Check for log(abs(det))
        elif isinstance(client.op, Elemwise) and isinstance(client.op.scalar_op, Abs):
            potential_log = None
            for client_2, _ in fgraph.clients[client.outputs[0]]:
                if isinstance(client_2.op, Elemwise) and isinstance(
                    client_2.op.scalar_op, Log
                ):
                    potential_log = client_2
            if potential_log:
                dummy_replacements[potential_log.outputs[0]] = "log_abs_det"
            else:
                return None

        # Check for log(det)
        elif isinstance(client.op, Elemwise) and isinstance(client.op.scalar_op, Log):
            dummy_replacements[client.outputs[0]] = "log_det"

        # Det is used directly for something else, don't rewrite to avoid computing two dets
        else:
            return None

    if not dummy_replacements:
        return None
    else:
        [x] = node.inputs
        sign_det_x, log_abs_det_x = SLogDet()(x)
        log_det_x = pt.where(pt.eq(sign_det_x, -1), np.nan, log_abs_det_x)
        slogdet_specialization_map = {
            "sign": sign_det_x,
            "log_abs_det": log_abs_det_x,
            "log_det": log_det_x,
        }
        replacements = {
            k: slogdet_specialization_map[v] for k, v in dummy_replacements.items()
        }
        return replacements


@register_stabilize
@register_canonicalize
@node_rewriter([blockwise_of(SolveBase)])
def scalar_solve_to_division(fgraph, node):
    """
    Replace solve(a, b) with b / a if a is a (1, 1) matrix
    """

    core_op = node.op.core_op
    if not isinstance(core_op, SolveBase):
        return None

    a, b = node.inputs
    old_out = node.outputs[0]
    if not all(a.broadcastable[-2:]):
        return None

    if core_op.b_ndim == 1:
        # Convert b to a column matrix
        b = b[..., None]

    # Special handling for different types of solve
    match core_op:
        case SolveTriangular():
            # Corner case: if user asked for a triangular solve with a unit diagonal, a is taken to be 1
            new_out = b / a if not core_op.unit_diagonal else pt.second(a, b)
        case CholeskySolve():
            new_out = b / a**2
        case Solve():
            new_out = b / a
        case _:
            raise NotImplementedError(
                f"Unsupported core_op type: {type(core_op)} in scalar_solve_to_divison"
            )

    if core_op.b_ndim == 1:
        # Squeeze away the column dimension added earlier
        new_out = new_out.squeeze(-1)

    copy_stack_trace(old_out, new_out)

    return [new_out]
