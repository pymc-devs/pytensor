from pytensor import tensor as pt
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.tensor.assumptions.diagonal import DIAGONAL
from pytensor.tensor.assumptions.symmetric import SYMMETRIC
from pytensor.tensor.assumptions.triangular import LOWER_TRIANGULAR, UPPER_TRIANGULAR
from pytensor.tensor.assumptions.utils import check_assumption
from pytensor.tensor.basic import alloc_diag
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.eigen import Eig, Eigh, Eigvalsh, eigh
from pytensor.tensor.linalg.decomposition.lu import LU, LUFactor
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.schur import QZ, Schur
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
    """
    A = node.inputs[0]
    lower = node.op.core_op.lower

    match A.owner_op_and_inputs:
        case (Blockwise(Dot()) | Dot(), l, r):
            if check_assumption(fgraph, l, LOWER_TRIANGULAR):
                match r.owner_op_and_inputs:
                    # cholesky(dot(L,L.T)) case
                    case (
                        DimShuffle(is_left_expanded_matrix_transpose=True),
                        l_T,
                    ) if l_T == l:
                        return [l] if lower else [r]

            if check_assumption(fgraph, r, UPPER_TRIANGULAR):
                match l.owner_op_and_inputs:
                    # cholesky(dot(U.T,U)) case
                    case (
                        DimShuffle(is_left_expanded_matrix_transpose=True),
                        r_T,
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
@node_rewriter([blockwise_of(SVD)])
def svd_of_diag(fgraph, node):
    """
    svd(D) has three return values:
        - S = diag(sort(abs(d)))
        - U = I
        - V = diag(sign(sort(d)))

    Where d is the diagonal of D and sort is in descending order. U and V are also permuted to match the sorted order
    of the diagonal of D.
    """
    [X] = node.inputs

    if not check_assumption(fgraph, X, DIAGONAL):
        return None

    diag_vals = pt.diagonal(X, axis1=-2, axis2=-1)

    # Singular values are abs of diagonal, sorted descending
    abs_diag = pt.abs(diag_vals)
    idx = pt.argsort(-abs_diag)
    new_s = abs_diag[idx]

    if not node.op.core_op.compute_uv:
        [s] = node.outputs
        copy_stack_trace(s, new_s)
        return [new_s]

    n = X.shape[-1]
    new_U = pt.eye(n)[:, idx]
    # Vh = diag(sign(d_sorted)) @ P, where P = I[idx, :]
    sorted_signs = pt.sign(diag_vals[idx])
    new_Vh = alloc_diag(sorted_signs, axis1=-1, axis2=-2)[:, idx]
    u, s, vh = node.outputs

    copy_stack_trace(u, new_U)
    copy_stack_trace(s, new_s)
    copy_stack_trace(vh, new_Vh)

    return [new_U, new_s, new_Vh]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(Eig)])
def eig_to_eigh(fgraph, node):
    """Replace eig(X) with eigh(X) when X is symmetric.

    eigh is faster (~2x), returns real outputs, and supports gradients.
    """
    [X] = node.inputs
    # Eigh is a subclass of Eig, so this tracker also matches Eigh — skip it
    if isinstance(node.op.core_op, Eigh):
        return None
    if not check_assumption(fgraph, X, SYMMETRIC):
        return None

    w, v = eigh(X)
    # Eig returns complex, Eigh returns real — cast to match original types
    old_w, old_v = node.outputs
    w = w.astype(old_w.type.dtype)
    v = v.astype(old_v.type.dtype)
    copy_stack_trace(old_w, w)
    copy_stack_trace(old_v, v)
    return [w, v]


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


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(LU)])
def lu_of_diag(fgraph, node):
    """lu(D) -> (I, I, D) for diagonal D. P=I, L=I, U=D."""
    [X] = node.inputs

    if not check_assumption(fgraph, X, DIAGONAL):
        return None

    core_op = node.op.core_op
    out_dtype = node.outputs[-1].type.dtype
    n = X.shape[-1]

    new_U = X.astype(out_dtype) if X.type.dtype != out_dtype else X

    if core_op.permute_l:
        # Returns (PL, U) — PL = I
        new_PL = pt.eye(n, dtype=out_dtype)
        pl, u = node.outputs
        copy_stack_trace(pl, new_PL)
        copy_stack_trace(u, new_U)
        return [new_PL, new_U]

    if core_op.p_indices:
        # Returns (p_idx, L, U) — p_idx = arange(n), L = I
        new_p = pt.arange(n, dtype="int32")
        new_L = pt.eye(n, dtype=out_dtype)
        p, l, u = node.outputs
        copy_stack_trace(p, new_p)
        copy_stack_trace(l, new_L)
        copy_stack_trace(u, new_U)
        return [new_p, new_L, new_U]

    # Default: returns (P, L, U) — P = I, L = I
    p_dtype = node.outputs[0].type.dtype
    new_P = pt.eye(n, dtype=p_dtype)
    new_L = pt.eye(n, dtype=out_dtype)
    p, l, u = node.outputs
    copy_stack_trace(p, new_P)
    copy_stack_trace(l, new_L)
    copy_stack_trace(u, new_U)
    return [new_P, new_L, new_U]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(LUFactor)])
def lu_factor_of_diag(fgraph, node):
    """lu_factor(D) -> (D, arange(n)) for diagonal D."""
    [X] = node.inputs

    if not check_assumption(fgraph, X, DIAGONAL):
        return None

    n = X.shape[-1]
    new_LU = X
    new_pivots = pt.arange(n, dtype="int32")

    lu_out, piv_out = node.outputs
    copy_stack_trace(lu_out, new_LU)
    copy_stack_trace(piv_out, new_pivots)
    return [new_LU, new_pivots]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(QR)])
def qr_of_diag(fgraph, node):
    """qr(D) -> (diag(sign(d)), diag(|d|)) for diagonal D.

    Q = diag(sign(diagonal(D))), R = diag(abs(diagonal(D))).
    """
    [X] = node.inputs

    if not check_assumption(fgraph, X, DIAGONAL):
        return None

    core_op = node.op.core_op

    out_dtype = node.outputs[0].type.dtype
    diag_vals = pt.diagonal(X, axis1=-2, axis2=-1)

    if core_op.mode == "raw":
        # Raw returns (H, tau, R). For diagonal: H=D, tau=zeros(n), R=D
        new_H = X.astype(out_dtype) if X.type.dtype != out_dtype else X
        n = X.shape[-1]
        new_tau = pt.zeros(n, dtype=out_dtype)
        new_R = new_H
        results = [new_H, new_tau, new_R]
    elif core_op.mode == "r":
        new_R = alloc_diag(pt.abs(diag_vals).astype(out_dtype), axis1=-2, axis2=-1)
        results = [new_R]
    else:
        new_Q = alloc_diag(pt.sign(diag_vals).astype(out_dtype), axis1=-2, axis2=-1)
        new_R = alloc_diag(pt.abs(diag_vals).astype(out_dtype), axis1=-2, axis2=-1)
        results = [new_Q, new_R]

    if core_op.pivoting:
        n = X.shape[-1]
        new_p = pt.arange(n, dtype="int32")
        results.append(new_p)

    for old, new in zip(node.outputs, results, strict=True):
        copy_stack_trace(old, new)

    return results


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(Schur)])
def schur_of_diag(fgraph, node):
    """schur(D) -> (T, Z) for diagonal D.

    Without sort: T=D, Z=I.
    With sort: T=diag(d[perm]), Z=I[:, perm] where perm puts selected eigenvalues first.
    """
    [X] = node.inputs

    if not check_assumption(fgraph, X, DIAGONAL):
        return None

    out_dtype = node.outputs[0].type.dtype
    n = X.shape[-1]
    diag_vals = pt.diagonal(X, axis1=-2, axis2=-1).astype(out_dtype)

    match node.op.core_op.sort:
        case None:
            new_T = X.astype(out_dtype) if X.type.dtype != out_dtype else X
            new_Z = pt.eye(n, dtype=out_dtype)
        case "lhp":
            select = diag_vals < 0
        case "rhp":
            select = diag_vals >= 0
        case "iuc":
            select = pt.abs(diag_vals) <= 1
        case "ouc":
            select = pt.abs(diag_vals) > 1

    if node.op.core_op.sort is not None:
        sort_idx = pt.argsort(-select.astype("int64"))
        new_T = alloc_diag(diag_vals[sort_idx], axis1=-2, axis2=-1)
        new_Z = pt.eye(n, dtype=out_dtype)[:, sort_idx]

    T, Z = node.outputs
    copy_stack_trace(T, new_T)
    copy_stack_trace(Z, new_Z)
    return [new_T, new_Z]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(QZ)])
def qz_of_diag(fgraph, node):
    """qz(A, B) -> (AA, BB, Q, Z) for diagonal A, B.

    Without sort: AA=A, BB=B, Q=I, Z=I.
    With sort: diagonals of AA, BB are permuted so selected eigenvalues come first;
    Q=Z=permutation matrix.
    """
    A, B = node.inputs

    if not check_assumption(fgraph, A, DIAGONAL):
        return None
    if not check_assumption(fgraph, B, DIAGONAL):
        return None

    core_op = node.op.core_op
    out_dtype = node.outputs[0].type.dtype
    n = A.shape[-1]

    new_AA = A.astype(out_dtype) if A.type.dtype != out_dtype else A
    new_BB = B.astype(out_dtype) if B.type.dtype != out_dtype else B

    if core_op.sort is None:
        new_Q = pt.eye(n, dtype=out_dtype)
        new_Z = pt.eye(n, dtype=out_dtype)
        diag_a = pt.diagonal(A, axis1=-2, axis2=-1)
        diag_b = pt.diagonal(B, axis1=-2, axis2=-1)
    else:
        diag_a = pt.diagonal(A, axis1=-2, axis2=-1).astype(out_dtype)
        diag_b = pt.diagonal(B, axis1=-2, axis2=-1).astype(out_dtype)
        # Generalized eigenvalues λ_i = a_i / b_i (real diagonal case)
        # Use pt.neq/pt.eq for elementwise zero checks (== tests identity on TensorVariables)
        b_nonzero = pt.neq(diag_b, 0)
        match core_op.sort:
            case "lhp":
                select = b_nonzero & (diag_a * diag_b < 0)
            case "rhp":
                select = b_nonzero & (diag_a * diag_b >= 0)
            case "iuc":
                select = b_nonzero & (pt.abs(diag_a) <= pt.abs(diag_b))
            case "ouc":
                select = (pt.eq(diag_b, 0) & pt.neq(diag_a, 0)) | (
                    pt.abs(diag_a) > pt.abs(diag_b)
                )

        sort_idx = pt.argsort(-select.astype("int64"))
        new_AA = alloc_diag(diag_a[sort_idx], axis1=-2, axis2=-1)
        new_BB = alloc_diag(diag_b[sort_idx], axis1=-2, axis2=-1)
        # Q = Z = permutation matrix: Q.T @ diag(a) @ Z = diag(a[perm])
        perm = pt.eye(n, dtype=out_dtype)[:, sort_idx]
        new_Q = perm
        new_Z = perm
        diag_a = diag_a[sort_idx]
        diag_b = diag_b[sort_idx]

    if core_op.return_eigenvalues:
        alpha_dtype = node.outputs[2].type.dtype
        new_alpha = diag_a.astype(alpha_dtype)
        new_beta = diag_b.astype(out_dtype)
        results = [new_AA, new_BB, new_alpha, new_beta, new_Q, new_Z]
    else:
        results = [new_AA, new_BB, new_Q, new_Z]

    for old, new in zip(node.outputs, results, strict=True):
        copy_stack_trace(old, new)

    return results
