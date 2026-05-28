import numpy as np

from pytensor import tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    ORTHOGONAL,
    PERMUTATION,
    SELECTION,
    check_assumption,
)
from pytensor.assumptions.selection import column_selection_index
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    node_rewriter,
)
from pytensor.tensor.basic import ExtractDiag, Eye, alloc_diag, concatenate, diag
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.products import Expm, KroneckerProduct
from pytensor.tensor.linalg.summary import det
from pytensor.tensor.math import Dot, outer, prod
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.subtensor import AdvancedSubtensor
from pytensor.tensor.variable import TensorConstant


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
@node_rewriter([Dot, blockwise_of(Dot)])
def dot_of_kron(fgraph, node):
    r"""Decompose ``kron(A, B) @ X`` into two matmuls.

    Applies the identity :math:`(A \otimes B)\, \mathrm{vec}_{\mathrm{row}}(X)
    = \mathrm{vec}_{\mathrm{row}}(A X B^\top)` column-wise across the RHS:

    .. math::
        (A \otimes B)\, Y
            = \mathrm{reshape}\bigl(
                  A\, \mathrm{reshape}(Y,\, (m, p, k))\, B^\top,\,
                  (m p,\, k)
              \bigr).

    Cost drops from :math:`O(k\, (m p)^2)` to :math:`O(k\, m p\, (m + p))`,
    and the :math:`(m p) \times (m p)` Kronecker matrix is never formed.
    """
    K, X = node.inputs

    # Peel Blockwise(Dot)'s batch-broadcast wrapper (plain expand or matrix-transposed).
    transposed = False
    match K.owner_op_and_inputs:
        case (DimShuffle(is_left_expand_dims=True), inner):
            K = inner
        case (DimShuffle(is_left_expanded_matrix_transpose=True), inner):
            K = inner
            transposed = True

    match K.owner_op_and_inputs:
        case (KroneckerProduct(), A, B):
            pass
        case _:
            return None

    # ``kron(A, B).mT == kron(A.mT, B.mT)`` for 2-D ``A, B``.
    if transposed:
        A, B = A.mT, B.mT

    if A.type.ndim != 2 or B.type.ndim != 2:
        return None

    m = A.shape[-1]
    p = B.shape[-1]

    # Bring k to the front so each (m, p) slice is a batched matmul argument.
    batch_shape = tuple(X.shape[i] for i in range(X.type.ndim - 2))
    k = X.shape[-1]
    X_3d = X.reshape((*batch_shape, m, p, k))
    X_3d = pt.moveaxis(X_3d, -1, -3)

    Z = A @ X_3d
    Z = Z @ B.mT

    Z = pt.moveaxis(Z, -3, -1)
    new_out = Z.reshape((*batch_shape, m * p, k))

    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


@register_canonicalize
@register_stabilize
@node_rewriter([KroneckerProduct])
def kron_of_diagonal_to_diagonal(fgraph, node):
    """kron(D1, D2) -> alloc_diag(outer_product) when both inputs are diagonal."""
    a, b = node.inputs

    if not (
        check_assumption(fgraph, a, DIAGONAL) and check_assumption(fgraph, b, DIAGONAL)
    ):
        return None

    diag_a = pt.diagonal(a, axis1=-2, axis2=-1)
    diag_b = pt.diagonal(b, axis1=-2, axis2=-1)

    kron_diag = pt.join_dims(diag_a[..., :, None] * diag_b[..., None, :], start_axis=-2)
    return [alloc_diag(kron_diag, axis1=-2, axis2=-1)]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(Expm)])
def expm_of_diag(fgraph, node):
    """expm(D) -> diag(exp(diagonal(D))) for diagonal D."""
    [X] = node.inputs

    if all(X.type.broadcastable[-2:]):
        return [pt.exp(X)]

    if not check_assumption(fgraph, X, DIAGONAL):
        return None

    diag_vals = pt.exp(pt.diagonal(X, axis1=-2, axis2=-1))
    new_out = alloc_diag(diag_vals, axis1=-2, axis2=-1)
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


@register_canonicalize
@register_stabilize
@node_rewriter([Dot, blockwise_of(Dot)])
def orthogonal_dot_transpose_to_eye(fgraph, node):
    """Replace X @ X.T -> eye and X.T @ X -> eye when X is orthogonal."""
    a, b = node.inputs
    if not check_assumption(fgraph, a, ORTHOGONAL) or not check_assumption(
        fgraph, b, ORTHOGONAL
    ):
        return None

    # X @ X.T case
    match b.owner_op_and_inputs:
        case (DimShuffle(is_matrix_transpose=True), b_inner) if b_inner is a:
            result = pt.eye(a.shape[-2], dtype=a.type.dtype)
            if a.type.ndim > 2:
                result = pt.broadcast_to(result, (*a.shape[:-2], *result.shape[-2:]))
            copy_stack_trace(node.outputs[0], result)
            return [result]

    # X.T @ X case
    match a.owner_op_and_inputs:
        case (DimShuffle(is_matrix_transpose=True), a_inner) if a_inner is b:
            result = pt.eye(b.shape[-1], dtype=b.type.dtype)
            if b.type.ndim > 2:
                result = pt.broadcast_to(result, (*b.shape[:-2], *result.shape[-2:]))
            copy_stack_trace(node.outputs[0], result)
            return [result]


def _selection_operand(fgraph, var):
    """Return ``(idx, transposed, n_rows)`` for a selection matmul operand, or ``None``.

    Non-``None`` when ``var`` is a selection ``S = eye(n)[:, idx]`` or its transpose
    (``transposed`` True). ``idx`` is read off the graph when free (an ``Eye`` column-selection
    or a literal constant); for an opaque assumed selection it is recovered with
    ``argmax(S, axis=-2)``.
    """

    def recover_index(S):
        match S.owner_op_and_inputs:
            case (AdvancedSubtensor() as op, base, *_):
                match base.owner_op_and_inputs:
                    case (Eye(), _, _, TensorConstant(0)):
                        return column_selection_index(op, S.owner)
        if isinstance(S, TensorConstant):
            return pt.constant(np.argmax(S.data, axis=-2))
        return pt.argmax(S, axis=-2)

    # A batched matmul left-expands a 2-D selection to batch rank with a single
    # broadcast DimShuffle; peel it to reach the underlying matrix.
    match var.owner_op_and_inputs:
        case (DimShuffle(is_left_expand_dims=True), inner):
            core = inner
        case _:
            core = var

    if core.type.ndim != 2:
        return None

    if check_assumption(fgraph, core, SELECTION):
        return recover_index(core), False, core.shape[0]

    match core.owner_op_and_inputs:
        case (DimShuffle(is_matrix_transpose=True), s) if (
            s.type.ndim == 2 and check_assumption(fgraph, s, SELECTION)
        ):
            return recover_index(s), True, s.shape[0]

    return None


@register_canonicalize
@register_stabilize
@node_rewriter([Dot, blockwise_of(Dot)])
def selection_dot_to_indexing(fgraph, node):
    """Replace a matmul by a selection matrix ``S = eye(n)[:, idx]`` with indexing.

    Gathers (turn an ``O(n k m)`` matmul into an ``O(k m)`` take):

    - ``x @ S``   -> ``x[..., idx]``      (gather columns)
    - ``S.T @ x`` -> ``x[..., idx, :]``   (gather rows)

    Scatters (turn it into a zero-fill plus an accumulate):

    - ``S @ y``   -> rows of ``y`` accumulated into a zero matrix at positions ``idx``
    - ``x @ S.T`` -> columns of ``x`` accumulated into a zero matrix at positions ``idx``
    """
    a, b = node.inputs
    out_dtype = node.outputs[0].type.dtype

    a_sel = _selection_operand(fgraph, a)
    b_sel = _selection_operand(fgraph, b)

    # Gathers first -- they index without allocating. A gather keeps the operand's dtype,
    # so cast up to the matmul's output dtype when a mixed-dtype product would have upcast.
    if b_sel is not None and not b_sel[1]:  # x @ S
        out = a[..., b_sel[0]]
        if a.type.dtype != out_dtype:
            out = out.astype(out_dtype)
        copy_stack_trace(node.outputs[0], out)
        return [out]

    if a_sel is not None and a_sel[1]:  # S.T @ x
        out = b[..., a_sel[0], :]
        if b.type.dtype != out_dtype:
            out = out.astype(out_dtype)
        copy_stack_trace(node.outputs[0], out)
        return [out]

    # Scatters: fill zeros, then accumulate the operand at the selected positions.
    if a_sel is not None and not a_sel[1]:  # S @ y
        idx, _, n = a_sel
        batch = [b.shape[i] for i in range(b.type.ndim - 2)]
        z = pt.zeros((*batch, n, b.shape[-1]), dtype=out_dtype)
        out = pt.inc_subtensor(z[..., idx, :], b)
        copy_stack_trace(node.outputs[0], out)
        return [out]

    if b_sel is not None and b_sel[1]:  # x @ S.T
        idx, _, n = b_sel
        batch = [a.shape[i] for i in range(a.type.ndim - 2)]
        z = pt.zeros((*batch, a.shape[-2], n), dtype=out_dtype)
        out = pt.inc_subtensor(z[..., idx], a)
        copy_stack_trace(node.outputs[0], out)
        return [out]

    return None


@register_canonicalize
@register_stabilize
@node_rewriter([det])
def det_of_permutation(fgraph, node):
    """Replace ``det(P)`` by the sign of the permutation for a permutation matrix ``P``.

    The determinant of a permutation is its sign: :math:`(-1)^k` where :math:`k` counts the
    out-of-order pairs (``i < j`` with ``idx[i] > idx[j]``) of the index that orders its
    columns, ``P = eye(n)[:, idx]``.
    """
    [x] = node.inputs
    if x.type.ndim != 2 or not check_assumption(fgraph, x, PERMUTATION):
        return None
    operand = _selection_operand(fgraph, x)
    if operand is None:
        return None

    idx = operand[0]
    inversions = pt.triu((idx[:, None] > idx[None, :]).astype("int64"), k=1).sum()
    sign = (1 - 2 * (inversions % 2)).astype(node.outputs[0].type.dtype)
    copy_stack_trace(node.outputs[0], sign)
    return [sign]
