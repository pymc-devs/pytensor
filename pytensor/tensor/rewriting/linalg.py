import logging

import numpy as np

from pytensor import tensor as pt
from pytensor.compile import optdb
from pytensor.graph import Apply, Constant, FunctionGraph
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    dfs_rewriter,
    node_rewriter,
)
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.scalar.basic import Abs, Exp, Log, Mul, Sign, Sqr
from pytensor.tensor.basic import (
    AllocDiag,
    ExtractDiag,
    Eye,
    TensorVariable,
    atleast_Nd,
    concatenate,
    diag,
    diagonal,
    ones,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Dot, Prod, log, outer, prod, variadic_mul
from pytensor.tensor.nlinalg import (
    SVD,
    KroneckerProduct,
    MatrixInverse,
    MatrixPinv,
    SLogDet,
    det,
    kron,
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
    _bilinear_solve_discrete_lyapunov,
    block_diag,
    cholesky,
    solve,
    solve_discrete_lyapunov,
    solve_triangular,
)


logger = logging.getLogger(__name__)
# TODO: Make this inherit from a common abstract base class
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
@node_rewriter([OpPattern(DimShuffle, is_left_expanded_matrix_transpose=True)])
def transpose_of_inv(fgraph, node):
    # TODO: Transpose is much more frequent that MatrixInverse, flip the rewrite pattern matching.
    [A] = node.inputs
    match A.owner_op_and_inputs:
        case (Blockwise(MatrixInverse()) as inv_op, X):
            return [inv_op(node.op(X))]


@register_stabilize
@node_rewriter([Dot])
def inv_to_solve(fgraph, node):
    """
    This utilizes a boolean `symmetric` tag on the matrices.

    TODO: Exploit other assumptions like 'triangular' and 'psd'
    TODO: Handle expand_dims / matrix transpose between inv and dot
    """
    l, r = node.inputs
    match l.owner_op_and_inputs:
        case (Blockwise(MatrixInverse()), X):
            assume_a = "sym" if getattr(X.tag, "symmetric", False) else "gen"
            return [solve(X, r, assume_a=assume_a)]

    match r.owner_op_and_inputs:
        case (Blockwise(MatrixInverse()), X):
            if getattr(X.tag, "symmetric", False):
                return [solve(X, (l.mT), assume_a="sym").mT]
            else:
                return [solve((X.mT), (l.mT)).mT]

    return None


@register_stabilize
@register_canonicalize
@node_rewriter([blockwise_of(OpPattern(Solve, assume_a="gen"))])
def generic_solve_to_solve_triangular(fgraph, node):
    """
    If any solve() is applied to the output of a cholesky op, then
    replace it with a triangular solve.

    """
    b_ndim = node.op.core_op.b_ndim
    A, b = node.inputs  # result is the solution to Ax=b
    match A.owner_op_and_inputs:
        case (Blockwise(Cholesky(lower=lower)), _):
            return [solve_triangular(A, b, lower=lower, b_ndim=b_ndim)]
        case (DimShuffle(is_left_expanded_matrix_transpose=True), A_T):
            match A_T.owner_op:
                case Blockwise(Cholesky(lower=lower)):
                    return [solve_triangular(A, b, lower=not lower, b_ndim=b_ndim)]


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
    elif a_bcast_batch_dims:
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
@node_rewriter([OpPattern(DimShuffle, is_left_expanded_matrix_transpose=True)])
def useless_symmetric_transpose(fgraph, node):
    x = node.inputs[0]
    if getattr(x.tag, "symmetric", False):
        return [atleast_Nd(x, n=node.outputs[0].type.ndim)]


@register_stabilize
@node_rewriter([blockwise_of(OpPattern(Solve, b_ndim=2))])
def psd_solve_to_chol_solve(fgraph, node):
    """
    This utilizes the Solve assume_a flag or a boolean `psd` tag on matrices.
    """
    assume_a = node.op.core_op.assume_a
    A, b = node.inputs  # result is the solution to Ax=b
    if assume_a == "pos" or getattr(A.tag, "psd", None) is True:
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
                    return [log(p).sum(axis=axis)]

            if getattr(x.tag, "positive", False):
                return [log(x).sum(axis=axis)]

        # Special case for log(abs(prod(x))) -> sum(log(abs(x))) that shows up in slogdet
        case (Elemwise(Abs()), p):
            match p.owner_op_and_inputs:
                case (Prod(axis=axis), x):
                    return [log(abs(x)).sum(axis=axis)]


@register_specialize
@node_rewriter([blockwise_of(MatrixInverse | Cholesky | MatrixPinv)])
def lift_linalg_of_expanded_matrices(fgraph: FunctionGraph, node: Apply):
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
    outer_op = node.op
    [y] = node.inputs

    match y.owner_op_and_inputs:
        case (Blockwise(BlockDiagonal()), *inner_matrices):
            return [block_diag(*(outer_op(m) for m in inner_matrices))]
        case (KroneckerProduct(), *inner_matrices):
            return [kron(*(outer_op(m) for m in inner_matrices))]  # type: ignore[unreachable]


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
    inp = node.inputs[0]

    match inp.owner_op_and_inputs:
        # Check for use of pt.diag first
        case (AllocDiag(offset=0, axis1=axis1, axis2=axis2), diag_input):
            ndim = diag_input.ndim
            if axis1 == ndim - 1 and axis2 == ndim:
                return [diag_input.prod(axis=-1)]

    # Check if the input is an elemwise multiply with identity matrix -- this also results in a diagonal matrix
    match is_eye_mul(inp):
        case (eye_term, non_eye_term):
            # Checking if original x was scalar/vector/matrix
            match non_eye_term.type.broadcastable[-2:]:
                case (True, True):
                    # For scalar
                    det_val = (
                        non_eye_term.squeeze(axis=(-1, -2)) ** (eye_term.shape[-1])
                    )
                case (False, False):
                    # For Matrix
                    det_val = non_eye_term.diagonal(axis1=-1, axis2=-2).prod(axis=-1)
                case _:
                    # For vector
                    det_val = non_eye_term.prod(axis=(-1, -2))

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
@node_rewriter([blockwise_of(MATRIX_INVERSE_OPS)])
def inv_of_inv(fgraph, node):
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
    # Check if inner op is blockwise and possible inv
    match node.inputs[0].owner_op_and_inputs:
        case (Blockwise(MatrixInverse() | MatrixPinv()), X):
            return [X]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(MATRIX_INVERSE_OPS)])
def inv_of_diag_to_diag_reciprocal(fgraph, node):
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
    inp = node.inputs[0]

    # Check for diagonal constructors first
    match inp.owner_op_and_inputs:
        case (Eye(), _, _, Constant(0)):
            return [inp]
        case (AllocDiag(offset=0, axis1=axis1, axis2=axis2), inv_input):
            ndim = inv_input.type.ndim
            if axis1 == ndim - 1 and axis2 == ndim:
                return [pt.diag(1 / inv_input)]

    # Check if the input is an elemwise multiply with identity matrix -- this also results in a diagonal matrix
    match is_eye_mul(inp):
        case (eye_term, non_eye_term):
            # For a matrix, we have to first extract the diagonal (non-zero values) and then only use those
            if non_eye_term.type.broadcastable[-2:] == (False, False):
                non_eye_diag = non_eye_term.diagonal(axis1=-1, axis2=-2)
                non_eye_term = pt.shape_padaxis(non_eye_diag, -2)

            return [eye_term / non_eye_term]


@register_canonicalize
@register_stabilize
@node_rewriter([ExtractDiag])
def diag_of_blockdiag(fgraph, node):
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
    match node.inputs[0].owner_op_and_inputs:
        case (Blockwise(BlockDiagonal()), *submatrices):
            submatrices_diag = [diag(m) for m in submatrices]
            return [concatenate(submatrices_diag, axis=-1)]


@register_canonicalize
@register_stabilize
@node_rewriter([det])
def det_of_blockdiag(fgraph, node):
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
    match node.inputs[0].owner_op_and_inputs:
        case (Blockwise(BlockDiagonal()), *sub_matrices):
            det_sub_matrices = [det(m) for m in sub_matrices]
            return [prod(det_sub_matrices, axis=-1)]


@register_canonicalize
@register_stabilize
@node_rewriter([ExtractDiag])
def diag_of_kronecker(fgraph, node):
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
    match node.inputs[0].owner_op_and_inputs:
        case (KroneckerProduct(), a, b):
            diag_a, diag_b = diag(a), diag(b)
            outer_prod_as_vector = outer(diag_a, diag_b).flatten()
            return [outer_prod_as_vector]


@register_canonicalize
@register_stabilize
@node_rewriter([det])
def det_of_kronecker(fgraph, node):
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
    match node.inputs[0].owner_op_and_inputs:
        case (KroneckerProduct(), a, b):
            dets = [det(a), det(b)]
            sizes = [a.shape[-1], b.shape[-1]]
            prod_sizes = prod(sizes, no_zeros_in_input=True)
            det_final = prod(
                [dets[i] ** (prod_sizes / sizes[i]) for i in range(2)], axis=-1
            )
            return [det_final]


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


@node_rewriter([_bilinear_solve_discrete_lyapunov])
def jax_bilinaer_lyapunov_to_direct(fgraph: FunctionGraph, node: Apply):
    """
    Replace BilinearSolveDiscreteLyapunov with a direct computation that is supported by JAX
    """
    A, B = node.inputs
    result = solve_discrete_lyapunov(A, B, method="direct")
    return [result]


optdb.register(
    "jax_bilinaer_lyapunov_to_direct",
    dfs_rewriter(jax_bilinaer_lyapunov_to_direct),
    "jax",
    position=0.9,  # Run before canonicalization
)


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


@register_stabilize
@register_canonicalize
@node_rewriter([blockwise_of(SolveBase)])
def scalar_solve_to_division(fgraph, node):
    """
    Replace solve(a, b) with b / a if a is a (1, 1) matrix
    """
    a, b = node.inputs
    old_out = node.outputs[0]
    if not all(a.broadcastable[-2:]):
        return None

    core_op = node.op.core_op

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
