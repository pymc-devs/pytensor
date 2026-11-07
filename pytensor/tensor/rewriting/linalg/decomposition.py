from itertools import chain

from pytensor import tensor as pt
from pytensor.assumptions import (
    DIAGONAL,
    LOWER_TRIANGULAR,
    ORTHOGONAL,
    SYMMETRIC,
    UPPER_TRIANGULAR,
    check_assumption,
)
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.tensor.basic import alloc_diag, atleast_Nd
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.constructors import BlockDiagonal, block_diag
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.eigen import (
    Eig,
    Eigh,
    Eigvalsh,
    eigh,
    eigvalsh,
)
from pytensor.tensor.linalg.decomposition.lu import LU, LUFactor
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.schur import QZ, Schur
from pytensor.tensor.linalg.decomposition.svd import SVD, svd
from pytensor.tensor.linalg.products import KroneckerProduct, kron
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.rewriting.linalg.utils import (
    clients_through_expand_dims,
    strip_left_expand_dims_and_transpose,
)


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(Cholesky)])
def cholesky_ldotlt(fgraph, node):
    """
    rewrite cholesky(dot(L, L.T), lower=True) = L, where L is lower triangular,
    or cholesky(dot(U.T, U), upper=True) = U where U is upper triangular.

    Also works with matmul.
    """
    # The matched products are symmetric, so the transposed flag is irrelevant
    A, _ = strip_left_expand_dims_and_transpose(node.inputs[0])
    [out] = node.outputs
    lower = node.op.core_op.lower

    match A.owner_op_and_inputs:
        case (Blockwise(Dot()) | Dot(), l, r):
            if getattr(l.tag, "lower_triangular", False) or check_assumption(
                fgraph, l, LOWER_TRIANGULAR
            ):
                match r.owner_op_and_inputs:
                    # cholesky(dot(L,L.T)) case
                    case (
                        DimShuffle(is_left_expanded_matrix_transpose=True),
                        l_T,
                    ) if l_T == l:
                        new_out = atleast_Nd(l if lower else r, n=out.type.ndim)
                        return [new_out.astype(out.type.dtype)]

            if getattr(r.tag, "upper_triangular", False) or check_assumption(
                fgraph, r, UPPER_TRIANGULAR
            ):
                match l.owner_op_and_inputs:
                    # cholesky(dot(U.T,U)) case
                    case (
                        DimShuffle(is_left_expanded_matrix_transpose=True),
                        r_T,
                    ) if r_T == r:
                        new_out = atleast_Nd(l if lower else r, n=out.type.ndim)
                        return [new_out.astype(out.type.dtype)]


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

    # The sibling SVD may hang off an alias of x with dummy batch dims, or x may have
    # such dims. Scan for clients of both forms. Transposes are not aliases of
    # the same decomposition, so a transposed core is not scanned.
    x_core, transposed = strip_left_expand_dims_and_transpose(x)
    svd_clients = clients_through_expand_dims(fgraph, x)
    if x_core is not x and not transposed:
        svd_clients = chain(svd_clients, clients_through_expand_dims(fgraph, x_core))

    if node.op.core_op.compute_uv:
        # compute_uv=True returns [u, s, v].
        u, s, v = node.outputs

        # if at least u or v is used, no need to rewrite this node.
        if fgraph.clients[u] or fgraph.clients[v]:
            return None

        # Else, has to replace the s of this node with s of an SVD Op that compute_uv=False.
        # First, iterate to see if there is an SVD Op that can be reused.
        for cl, _ in svd_clients:
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
        if replacement_s.type.ndim > s.type.ndim:
            # A sibling SVD found on an expanded alias of x carries the
            # alias's dummy leading dims, which are broadcastable by
            # construction
            replacement_s = replacement_s.squeeze(
                axis=tuple(range(replacement_s.type.ndim - s.type.ndim))
            )
        return {s: atleast_Nd(replacement_s, n=s.type.ndim)}

    else:
        # compute_uv=False returns [s].
        # We want rewrite if there is another one with compute_uv=True.
        # For this case, just reuse the `s` from the one with compute_uv=True.
        for cl, _ in svd_clients:
            if cl is node:
                continue
            match (cl.op, *cl.outputs):
                case (Blockwise(SVD(compute_uv=True)), u, s, v) if (
                    fgraph.clients[u] or fgraph.clients[v]
                ):
                    [own_s] = node.outputs
                    if s.type.ndim > own_s.type.ndim:
                        # A sibling SVD found on an expanded alias of x
                        # carries the alias's dummy leading dims, which are
                        # broadcastable by construction
                        s = s.squeeze(axis=tuple(range(s.type.ndim - own_s.type.ndim)))
                    return [atleast_Nd(s, n=own_s.type.ndim)]


def _svd_of_orthogonal(X, op):
    """svd(Q) -> (Q, ones, I) for orthogonal Q (unitary, for complex Q)."""
    new_s = pt.ones(X.shape[:-1])
    if not op.compute_uv:
        return [new_s]

    new_Vh = pt.broadcast_to(pt.eye(X.shape[-1]), X.shape)
    return [X, new_s, new_Vh]


def _svd_of_diag(X, op):
    """Closed-form svd of an m x n diagonal matrix D with diagonal d.

    The singular values are abs(d) sorted descending. Column j of U is the standard
    basis vector e_{idx[j]}, where idx is that sort order, and row j of Vh is the same
    vector scaled by the phase d / abs(d) of the matching diagonal entry (1 where it is
    zero). With full_matrices, the remaining columns of U and rows of Vh are the unused
    basis vectors, in order.
    """
    diag_vals = pt.diagonal(X, axis1=-2, axis2=-1)
    abs_diag = pt.abs(diag_vals)

    if not op.compute_uv:
        return [pt.sort(abs_diag, axis=-1)[..., ::-1]]

    m, n = X.shape[-2], X.shape[-1]

    idx = pt.argsort(abs_diag, axis=-1)[..., ::-1]
    sorted_diag = pt.take_along_axis(diag_vals, idx, axis=-1)
    new_s = pt.take_along_axis(abs_diag, idx, axis=-1)
    phases = pt.where(pt.eq(sorted_diag, 0), 1, pt.sign(sorted_diag))

    if op.full_matrices:
        u_order = _complete_permutation(idx, size=m)
        vh_order = _complete_permutation(idx, size=n)
        padding = pt.ones(
            (*phases.shape[:-1], n - phases.shape[-1]), dtype=phases.dtype
        )
        phases = pt.concatenate([phases, padding], axis=-1)
    else:
        u_order = vh_order = idx

    new_U = pt.eye(m)[u_order].mT
    new_Vh = pt.eye(n)[vh_order] * pt.expand_dims(phases, -1)
    return [new_U, new_s, new_Vh]


def _svd_of_symmetric(X, op):
    """svd(X) from eigh(X) for real symmetric X.

    For X = V @ diag(w) @ V.T, the singular values are abs(w) and the singular vectors
    are U = V, Vh = (V * sign(w)).T, all sorted by descending abs(w).
    """
    if not op.compute_uv:
        return [pt.sort(pt.abs(eigvalsh(X)), axis=-1)[..., ::-1]]

    w, V = eigh(X)
    abs_w = pt.abs(w)
    idx = pt.argsort(abs_w, axis=-1)[..., ::-1]
    column_idx = pt.expand_dims(idx, -2)
    signs = pt.where(w < 0, -1, 1).astype(w.dtype)

    new_s = pt.take_along_axis(abs_w, idx, axis=-1)
    new_U = pt.take_along_axis(V, column_idx, axis=-1)
    new_Vh = pt.take_along_axis(V * pt.expand_dims(signs, -2), column_idx, axis=-1).mT
    return [new_U, new_s, new_Vh]


def _svd_of_orthogonal_product(product, Q, other, op, *, q_on_left):
    """svd(Q @ X) -> (Q @ U, s, Vh) and svd(X @ Q) -> (U, s, Vh @ Q), Q orthogonal.

    Q is unitary for complex Q. Each application peels one orthogonal factor, so
    Q1 @ X @ Q2 is handled in two.
    """

    def broadcast_batch(var, core_ndim):
        if var.type.broadcastable[:-core_ndim] == product.type.broadcastable[:-2]:
            return var
        return pt.broadcast_to(var, (*product.shape[:-2], *var.shape[-core_ndim:]))

    if not op.compute_uv:
        return [broadcast_batch(svd(other, compute_uv=False), core_ndim=1)]

    U, s, Vh = svd(other, full_matrices=op.full_matrices)
    if q_on_left:
        U, Vh = Q @ U, broadcast_batch(Vh, core_ndim=2)
    else:
        U, Vh = broadcast_batch(U, core_ndim=2), Vh @ Q
    return [U, broadcast_batch(s, core_ndim=1), Vh]


def _complete_permutation(first, size):
    """Order ``arange(size)`` as ``first``, then the unused positions ascending."""
    p = first.shape[-1]
    positions = pt.arange(size)
    is_at = pt.eq(positions[:, None], pt.expand_dims(first, -2))
    key = pt.where(pt.any(is_at, axis=-1), pt.argmax(is_at, axis=-1), p + positions)
    return pt.argsort(key, axis=-1)


def _svd_of_kron_or_block_diag(core, factors, op, *, transposed, is_kron):
    """svd(kron(A, B)) and svd(block_diag(A_1, ..., A_p)) from the svd of each factor.

    Both matrices are U_big @ S @ Vh_big, where U_big and Vh_big are the kron (or
    block_diag) of the factors' U and Vh. S holds the factors' singular values, combined
    (pairwise products for kron, concatenated for block_diag), at most one per row and
    column. Rectangular factors make S rectangular and add zero singular values.
    """
    factor_shapes = [factor.type.shape[-2:] for factor in factors]
    shapes_known = all(dim is not None for shape in factor_shapes for dim in shape)
    all_square = shapes_known and all(m == n for m, n in factor_shapes)
    same_orientation = shapes_known and (
        all(m >= n for m, n in factor_shapes) or all(m <= n for m, n in factor_shapes)
    )

    # A reduced svd from reduced factor svds is only diagonal when the factors share an
    # orientation. Otherwise the zero singular values need full factor bases.
    if op.compute_uv and not (op.full_matrices or same_orientation):
        return None

    # Positions (rows, cols) of the singular values in S, needed only when S is not the
    # square diagonal matrix diag(s_vals)
    needs_positions = op.compute_uv and op.full_matrices and not all_square
    if is_kron:
        a, b = factors
        if op.compute_uv:
            U_a, s_a, Vh_a = svd(a, full_matrices=op.full_matrices)
            U_b, s_b, Vh_b = svd(b, full_matrices=op.full_matrices)
            U_big, Vh_big = kron(U_a, U_b), kron(Vh_a, Vh_b)
        else:
            s_a, s_b = svd(a, compute_uv=False), svd(b, compute_uv=False)
        s_vals = pt.outer(s_a, s_b).ravel()
        if needs_positions:
            a_idx = pt.arange(s_a.shape[-1])[:, None]
            b_idx = pt.arange(s_b.shape[-1])[None, :]
            rows = (a_idx * b.shape[-2] + b_idx).ravel()
            cols = (a_idx * b.shape[-1] + b_idx).ravel()
    else:
        if op.compute_uv:
            block_svds = [
                svd(block, full_matrices=op.full_matrices) for block in factors
            ]
            U_big = block_diag(*[U for U, _, _ in block_svds])
            Vh_big = block_diag(*[Vh for _, _, Vh in block_svds])
            s_blocks = [s for _, s, _ in block_svds]
        else:
            s_blocks = [svd(block, compute_uv=False) for block in factors]
        batch_shape = tuple(core.shape[:-2])
        if batch_shape:
            s_blocks = [
                pt.broadcast_to(s, (*batch_shape, s.shape[-1])) for s in s_blocks
            ]
        s_vals = pt.concatenate(s_blocks, axis=-1)
        if needs_positions:
            row_offset = col_offset = 0
            block_rows, block_cols = [], []
            for block, s in zip(factors, s_blocks, strict=True):
                positions = pt.arange(s.shape[-1])
                block_rows.append(row_offset + positions)
                block_cols.append(col_offset + positions)
                row_offset = row_offset + block.shape[-2]
                col_offset = col_offset + block.shape[-1]
            rows = pt.concatenate(block_rows)
            cols = pt.concatenate(block_cols)

    M, N = core.shape[-2], core.shape[-1]

    def pad_with_zeros(values):
        n_zeros = pt.minimum(M, N) - values.shape[-1]
        zeros = pt.zeros((*values.shape[:-1], n_zeros), dtype=values.dtype)
        return pt.concatenate([values, zeros], axis=-1)

    if not op.compute_uv:
        if not all_square:
            s_vals = pad_with_zeros(s_vals)
        return [pt.sort(s_vals, axis=-1)[..., ::-1]]

    order = pt.argsort(s_vals, axis=-1)[..., ::-1]
    new_s = pt.take_along_axis(s_vals, order, axis=-1)
    if needs_positions:
        new_s = pad_with_zeros(new_s)
        row_perm = _complete_permutation(rows[order], size=M)
        col_perm = _complete_permutation(cols[order], size=N)
    else:
        row_perm = col_perm = order

    new_U = pt.take_along_axis(U_big, pt.expand_dims(row_perm, -2), axis=-1)
    new_Vh = pt.take_along_axis(Vh_big, pt.expand_dims(col_perm, -1), axis=-2)
    if transposed:
        new_U, new_Vh = new_Vh.mT, new_U.mT

    return [new_U, new_s, new_Vh]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(SVD)])
def svd_of_structured(fgraph, node):
    """Rewrite svd(X) of structured X as a closed form or as cheaper decompositions.

    X may be orthogonal, diagonal, real symmetric, a product with an orthogonal factor,
    a kron or a block_diag.
    """
    [X] = node.inputs
    op = node.op.core_op
    core, transposed = strip_left_expand_dims_and_transpose(X)

    # Assumption arms are ordered cheapest first: an orthogonal diagonal matrix needs no
    # sort, and a diagonal symmetric matrix needs no eigh
    match core.owner_op_and_inputs:
        case _ if check_assumption(fgraph, X, ORTHOGONAL):
            new_outputs = _svd_of_orthogonal(X, op)
        case _ if check_assumption(fgraph, X, DIAGONAL):
            new_outputs = _svd_of_diag(X, op)
        case _ if not X.type.dtype.startswith("complex") and check_assumption(
            fgraph, X, SYMMETRIC
        ):
            new_outputs = _svd_of_symmetric(X, op)
        case (Blockwise(Dot()) | Dot(), Q, rest) if core is X and check_assumption(
            fgraph, Q, ORTHOGONAL
        ):
            new_outputs = _svd_of_orthogonal_product(X, Q, rest, op, q_on_left=True)
        case (Blockwise(Dot()) | Dot(), rest, Q) if core is X and check_assumption(
            fgraph, Q, ORTHOGONAL
        ):
            new_outputs = _svd_of_orthogonal_product(X, Q, rest, op, q_on_left=False)
        case (KroneckerProduct(), a, b) if a.type.ndim == 2 and b.type.ndim == 2:
            new_outputs = _svd_of_kron_or_block_diag(
                core, [a, b], op, transposed=transposed, is_kron=True
            )
        case (Blockwise(BlockDiagonal()), *blocks):
            new_outputs = _svd_of_kron_or_block_diag(
                core, blocks, op, transposed=transposed, is_kron=False
            )
        case _:
            return None

    if new_outputs is None:
        return None

    new_outputs = [
        atleast_Nd(new, n=old.type.ndim).astype(old.type.dtype)
        for new, old in zip(new_outputs, node.outputs, strict=True)
    ]
    copy_stack_trace(node.outputs, new_outputs)
    return new_outputs


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
@node_rewriter([blockwise_of(Eigh)])
def eigh_of_diag(fgraph, node):
    """Closed-form eigh on diagonal inputs.

    - ``eigh(A)`` with ``A`` diagonal -> sort the diagonal; eigenvectors are the standard
      basis permuted by the sort.
    - ``eigh(A, B)`` with both diagonal -> eigenvalues are ``a / b``; eigenvectors are
      ``e_i / sqrt(b_i)`` so they satisfy ``V^T B V = I``.
    """
    w, v = node.outputs
    if len(node.inputs) == 1:
        [X] = node.inputs
        if not check_assumption(fgraph, X, DIAGONAL):
            return None
        diag_vals = pt.diagonal(X, axis1=-2, axis2=-1)
        sort_idx = pt.argsort(diag_vals, axis=-1)
        new_w = pt.take_along_axis(diag_vals, sort_idx, axis=-1)
        new_v = pt.eye(X.shape[-1], dtype=v.type.dtype)[sort_idx].mT
    elif len(node.inputs) == 2:
        A, B = node.inputs
        if not check_assumption(fgraph, A, DIAGONAL):
            return None
        if not check_assumption(fgraph, B, DIAGONAL):
            return None
        diag_a = pt.diagonal(A, axis1=-2, axis2=-1)
        diag_b = pt.diagonal(B, axis1=-2, axis2=-1)
        eigvals = diag_a / diag_b
        sort_idx = pt.argsort(eigvals, axis=-1)
        new_w = pt.take_along_axis(eigvals, sort_idx, axis=-1)
        # B-orthonormal eigenvectors: column j is e_{sort_idx[j]} / sqrt(b[sort_idx[j]]).
        sorted_b = pt.take_along_axis(diag_b, sort_idx, axis=-1)
        new_v = pt.eye(A.shape[-1], dtype=v.type.dtype)[sort_idx].mT / pt.expand_dims(
            pt.sqrt(sorted_b), -2
        )
    else:
        return None

    copy_stack_trace(w, new_w)
    copy_stack_trace(v, new_v)

    return [new_w, new_v]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(Eigvalsh)])
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
