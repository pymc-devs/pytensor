from collections.abc import Container

from pytensor import tensor as pt
from pytensor.assumptions import DIAGONAL, ORTHOGONAL, check_assumption
from pytensor.assumptions.positive_definite import POSITIVE_DEFINITE
from pytensor.compile import optdb
from pytensor.graph import Constant, graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    dfs_rewriter,
    node_rewriter,
)
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import scan_seqopt1
from pytensor.tensor.basic import atleast_Nd, split
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.decomposition.lu import lu_factor
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.solvers.core import SolveBase
from pytensor.tensor.linalg.solvers.general import Solve, lu_solve, solve
from pytensor.tensor.linalg.solvers.linear_control import (
    SolveBilinearDiscreteLyapunov,
    SolveSylvester,
    solve_discrete_lyapunov,
)
from pytensor.tensor.linalg.solvers.psd import CholeskySolve, cho_solve
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular, solve_triangular
from pytensor.tensor.linalg.solvers.tridiagonal import (
    tridiagonal_lu_factor,
    tridiagonal_lu_solve,
)
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.rewriting.linalg.utils import get_assume_a
from pytensor.tensor.variable import TensorVariable


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


@register_stabilize
@node_rewriter([blockwise_of(OpPattern(Solve, b_ndim=2))])
def psd_solve_to_chol_solve(fgraph, node):
    """Rewrite solve(A, b) → triangular solves via Cholesky when A is positive-definite."""
    assume_a = node.op.core_op.assume_a
    A, b = node.inputs
    if (
        assume_a == "pos"
        or getattr(A.tag, "psd", None) is True
        or check_assumption(fgraph, A, POSITIVE_DEFINITE)
    ):
        L = cholesky(A)
        Li_b = solve_triangular(L, b, lower=True, b_ndim=2)
        x = solve_triangular((L.mT), Li_b, lower=False, b_ndim=2)
        return [x]


@register_specialize
@node_rewriter([blockwise_of(SolveTriangular)])
def paired_triangular_solves_to_cho_solve(fgraph, node):
    """Fuse paired triangular solves from Cholesky into a single cho_solve.

    solve_triangular(L.T, solve_triangular(L, b), lower=False) -> cho_solve((L, True), b)
    """
    core_op = node.op.core_op

    # We're looking for the outer solve: solve_triangular(L.T, ..., lower=False)
    if core_op.lower:
        return None

    L_T, inner_result = node.inputs

    # Check L.T is a matrix transpose of a Cholesky factor
    match L_T.owner_op_and_inputs:
        case (DimShuffle(is_left_expanded_matrix_transpose=True), L):
            pass
        case _:
            return None

    # L must be output of a Cholesky(lower=True)
    match L.owner_op:
        case Blockwise(Cholesky(lower=True)):
            pass
        case _:
            return None

    # inner_result must be solve_triangular(L, b, lower=True)
    match inner_result.owner_op_and_inputs:
        case (Blockwise(SolveTriangular(lower=True)), inner_L, b):
            pass
        case _:
            return None

    # inner_L must be the same Cholesky output as L
    if inner_L is not L:
        return None

    b_ndim = core_op.b_ndim
    new_out = cho_solve((L, True), b, b_ndim=b_ndim)
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


@register_stabilize
@register_canonicalize
@node_rewriter([blockwise_of(OpPattern(Solve, assume_a="gen"))])
def generic_solve_to_structured_form(fgraph, node):
    """Upgrade solve(A, b, assume_a='gen') based on known structure of A.

    Priority order (most specialized first):
    - LOWER_TRIANGULAR -> solve_triangular(lower=True)
    - UPPER_TRIANGULAR -> solve_triangular(lower=False)
    - POSITIVE_DEFINITE -> solve(assume_a='pos')
    - SYMMETRIC → solve(assume_a='sym')
    """
    b_ndim = node.op.core_op.b_ndim
    A, b = node.inputs

    assume_a = get_assume_a(fgraph, A)
    if assume_a == "gen":
        return None

    return [solve(A, b, assume_a=assume_a, b_ndim=b_ndim)]


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


@register_stabilize
@node_rewriter([blockwise_of(SolveBase)])
def solve_of_inv_to_matmul(fgraph, node):
    """Replace solve(matrix_inverse(X), b) with X @ b.

    If A = inv(X), then solve(A, b) finds x such that A @ x = b,
    i.e., inv(X) @ x = b, so x = X @ b.
    """
    A, b = node.inputs

    match A.owner_op_and_inputs:
        case (Blockwise(MatrixInverse()), X):
            new_out = X @ b
            copy_stack_trace(node.outputs[0], new_out)
            return [new_out]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(SolveBase)])
def diagonal_solve_to_division(fgraph, node):
    """Replace solve(D, b) with b / diagonal(D) when D is known diagonal."""
    a, b = node.inputs

    # Scalar case already handled by scalar_solve_to_division
    if all(a.type.broadcastable[-2:]):
        return None

    if not check_assumption(fgraph, a, DIAGONAL):
        return None

    core_op = node.op.core_op
    b_ndim = core_op.b_ndim
    a_diag = pt.diagonal(a, axis1=-2, axis2=-1)

    match core_op:
        case SolveTriangular(unit_diagonal=True):
            # Unit diagonal means diag is all ones; solve is identity
            return [b]
        case Solve() | SolveTriangular():
            if b_ndim == 1:
                new_out = b / a_diag
            else:
                new_out = b / a_diag[..., :, None]
        case CholeskySolve():
            # D is the Cholesky factor; D @ D.T = diag(d^2) for diagonal D
            if b_ndim == 1:
                new_out = b / a_diag**2
            else:
                new_out = b / (a_diag**2)[..., :, None]
        case _:
            return None

    old_out = node.outputs[0]
    copy_stack_trace(old_out, new_out)
    return [new_out]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(SolveBase)])
def block_diag_solve_to_block_diag_solves(fgraph, node):
    """Push ``solve(block_diag(A_1, ..., A_n), b)`` into per-block solves.

    Two cases:

    - ``b`` is also ``block_diag(B_1, ..., B_n)`` with matching block sizes:
      decompose into ``block_diag(solve(A_i, B_i))`` (each per-block solve is
      ``(m_i, m_i)`` instead of ``(m_i, m_total)``).
    - Otherwise: split ``b`` into per-block row chunks and solve each block
      independently, then concatenate.

    Both paths reuse the user's solve op (preserving ``assume_a``, ``lower``,
    ``unit_diagonal``, etc.) — valid because for a block-diagonal matrix to be
    symmetric / diagonal / psd / triangular, each block must individually
    satisfy that property.

    See also :func:`local_block_diag_dot_to_dot_block_diag`.
    """
    A, b = node.inputs

    match A.owner_op_and_inputs:
        case (Blockwise(BlockDiagonal()), *blocks):
            pass
        case _:
            return None

    block_sizes = [block.type.shape[-1] for block in blocks]

    # Rewrite is conservative: we require all component matrices to be provably square.
    # It is possible to have a square matrix block_diagonal matrix comprised of non-square
    # components. In that case the original graph is valid, but it is not decomposable.
    if any(
        size is None or block.type.shape[-2] != size
        for size, block in zip(block_sizes, blocks)
    ):
        return None

    core_op = node.op.core_op
    per_block_op = Blockwise(core_op)

    # If b is also a block_diag with matching block sizes, solve per matching
    # pair and rebuild block_diag. Strictly better than the row-split path:
    # avoids solving against the zero off-diagonal columns of each row-chunk.
    match b.owner_op_and_inputs:
        case (Blockwise(BlockDiagonal()), *b_blocks) if (
            core_op.b_ndim == 2
            and len(b_blocks) == len(blocks)
            and all(
                B_i.type.shape[-2] == size and B_i.type.shape[-1] == size
                for B_i, size in zip(b_blocks, block_sizes)
            )
        ):
            per_block_solutions = [
                per_block_op(A_i, B_i) for A_i, B_i in zip(blocks, b_blocks)
            ]
            for sol in per_block_solutions:
                copy_stack_trace(node.outputs[0], sol)
            new_out = pt.linalg.block_diag(*per_block_solutions)
            copy_stack_trace(node.outputs[0], new_out)
            return [new_out]

    # For vector b (b_ndim=1) split along its only axis; for matrix b (b_ndim=2)
    # split along the row axis. Either way the per-block solve preserves b_ndim.
    split_axis = -1 if core_op.b_ndim == 1 else -2
    chunks = split(b, splits_size=block_sizes, axis=split_axis)

    per_block_solutions = []
    for block, chunk in zip(blocks, chunks):
        sol = per_block_op(block, chunk)
        copy_stack_trace(node.outputs[0], sol)
        per_block_solutions.append(sol)

    new_out = pt.concatenate(per_block_solutions, axis=split_axis)
    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(SolveBase)])
def orthogonal_solve_to_transpose_matmul(fgraph, node):
    """Replace solve(Q, b) with Q.T @ b when Q is orthogonal."""
    A, b = node.inputs

    if not check_assumption(fgraph, A, ORTHOGONAL):
        return None

    b_ndim = node.op.core_op.b_ndim
    if b_ndim == 1:
        new_out = (A.mT @ b[..., :, None])[..., 0]
    else:
        new_out = A.mT @ b

    old_out = node.outputs[0]
    copy_stack_trace(old_out, new_out)
    return [new_out]


@register_canonicalize
@register_stabilize
@node_rewriter([blockwise_of(SolveSylvester)])
def solve_sylvester_of_diag(fgraph, node):
    """Replace solve_sylvester(A, B, C) with C / (a[:, None] + b[None, :]) when both A, B are diagonal.

    The Sylvester equation A @ X + X @ B = C, when A = diag(a) and B = diag(b),
    reduces to X_ij = C_ij / (a_i + b_j).
    """
    A, B, C = node.inputs

    if not check_assumption(fgraph, A, DIAGONAL):
        return None
    if not check_assumption(fgraph, B, DIAGONAL):
        return None

    a = pt.diagonal(A, axis1=-2, axis2=-1)
    b = pt.diagonal(B, axis1=-2, axis2=-1)

    new_out = C / (a[..., :, None] + b[..., None, :])

    old_out = node.outputs[0]
    copy_stack_trace(old_out, new_out)
    return [new_out]


def decompose_A(A, assume_a, lower):
    if assume_a == "gen":
        return lu_factor(A)
    elif assume_a == "tridiagonal":
        return tridiagonal_lu_factor(A)
    elif assume_a == "pos":
        return cholesky(A, lower=lower)
    else:
        raise NotImplementedError


def solve_decomposed_system(
    A_decomp, b, transposed=False, lower=False, *, core_solve_op: Solve
):
    b_ndim = core_solve_op.b_ndim
    assume_a = core_solve_op.assume_a

    if assume_a == "gen":
        return lu_solve(
            A_decomp,
            b,
            b_ndim=b_ndim,
            trans=transposed,
        )
    elif assume_a == "tridiagonal":
        return tridiagonal_lu_solve(
            A_decomp,
            b,
            b_ndim=b_ndim,
            transposed=transposed,
        )
    elif assume_a == "pos":
        # We can ignore the transposed argument here because A is symmetric by assumption
        return cho_solve(
            (A_decomp, lower),
            b,
            b_ndim=b_ndim,
        )
    else:
        raise NotImplementedError


def _split_decomp_and_solve_steps(
    fgraph, node, *, eager: bool, allowed_assume_a: Container[str]
):
    if not isinstance(node.op.core_op, Solve):
        return None

    def get_root_A(a: TensorVariable) -> tuple[TensorVariable, bool]:
        # Find the root variable of the first input to Solve
        # If `a` is a left expand_dims or matrix transpose (DimShuffle variants),
        # the root variable is the pre-DimShuffled input.
        # Otherwise, `a` is considered the root variable.
        # We also return whether the root `a` is transposed.
        root_a = a
        transposed = False
        match a.owner_op_and_inputs:
            case (DimShuffle(is_left_expand_dims=True), root_a):  # type: ignore[misc]
                transposed = False
            case (DimShuffle(is_left_expanded_matrix_transpose=True), root_a):  # type: ignore[misc]
                transposed = True  # type: ignore[unreachable, unused-ignore]

        return root_a, transposed

    def find_solve_clients(var, assume_a):
        clients = []
        for cl, idx in fgraph.clients[var]:
            match (idx, cl.op, *cl.outputs):
                case (0, Blockwise(Solve(assume_a=assume_a_var)), *_) if (
                    assume_a_var == assume_a
                ):
                    clients.append(cl)
                case (0, DimShuffle(is_left_expand_dims=True), cl_out):
                    clients.extend(find_solve_clients(cl_out, assume_a))
        return clients

    assume_a = node.op.core_op.assume_a

    if assume_a not in allowed_assume_a:
        return None

    A, _ = get_root_A(node.inputs[0])

    # Find Solve using A (or left expand_dims of A)
    # TODO: We could handle arbitrary shuffle of the batch dimensions, just need to propagate
    #  that to the A_decomp outputs
    A_solve_clients_and_transpose = [
        (client, False) for client in find_solve_clients(A, assume_a)
    ]

    # Find Solves using A.T
    for cl, _ in fgraph.clients[A]:
        match (cl.op, *cl.outputs):
            case (DimShuffle(is_left_expanded_matrix_transpose=True), A_T):
                A_solve_clients_and_transpose.extend(
                    (client, True) for client in find_solve_clients(A_T, assume_a)
                )

    if not eager and len(A_solve_clients_and_transpose) == 1:
        # If theres' a single use don't do it... unless it's being broadcast in a Blockwise (or we're eager)
        # That's a "reuse" inside the inner vectorized loop
        batch_ndim = node.op.batch_ndim(node)
        (client, _) = A_solve_clients_and_transpose[0]
        original_A, b = client.inputs
        if not any(
            a_bcast and not b_bcast
            for a_bcast, b_bcast in zip(
                original_A.type.broadcastable[:batch_ndim],
                b.type.broadcastable[:batch_ndim],
                strict=True,
            )
        ):
            return None

    lower = node.op.core_op.lower
    A_decomp = decompose_A(A, assume_a=assume_a, lower=lower)

    replacements = {}
    for client, transposed in A_solve_clients_and_transpose:
        _, b = client.inputs
        new_x = solve_decomposed_system(
            A_decomp,
            b,
            transposed=transposed,
            lower=lower,
            core_solve_op=client.op.core_op,
        )
        [old_x] = client.outputs
        new_x = atleast_Nd(new_x, n=old_x.type.ndim).astype(old_x.type.dtype)
        copy_stack_trace(old_x, new_x)
        replacements[old_x] = new_x

    return replacements


def _scan_split_non_sequence_decomposition_and_solve(
    fgraph, node, *, allowed_assume_a: Container[str]
):
    """If the A of a Solve within a Scan is a function of non-sequences, split the LU decomposition step.

    The LU decomposition step can then be pushed out of the inner loop by the `scan_pushout_non_sequences` rewrite.
    """
    scan_op: Scan = node.op
    frozen_fgraph = scan_op.fgraph
    non_sequences = set(scan_op.inner_non_seqs(frozen_fgraph.inputs))
    new_scan_fgraph: FunctionGraph | None = None

    while True:
        for inner_node in (
            frozen_fgraph if new_scan_fgraph is None else new_scan_fgraph
        ).toposort():
            match (inner_node.op, *inner_node.inputs):
                case (Blockwise(Solve(assume_a=assume_a_var)), A, _b) if (
                    assume_a_var in allowed_assume_a
                ):
                    if all(
                        (isinstance(root_inp, Constant) or (root_inp in non_sequences))
                        for root_inp in graph_inputs([A])
                    ):
                        if new_scan_fgraph is None:
                            # Thaw the frozen graph into a mutable copy on the
                            # first match, carrying the tracked state over.
                            new_scan_fgraph, equiv = frozen_fgraph.unfreeze(
                                return_memo=True
                            )
                            non_sequences = {equiv[v] for v in non_sequences}
                            inner_node = equiv[inner_node]

                        replace_dict = _split_decomp_and_solve_steps(
                            new_scan_fgraph,
                            inner_node,
                            eager=True,
                            allowed_assume_a=allowed_assume_a,
                        )
                        assert (
                            isinstance(replace_dict, dict) and len(replace_dict) > 0
                        ), "Rewrite failed"
                        new_scan_fgraph.replace_all(replace_dict.items())
                        break  # Break to start over with a fresh toposort
        else:  # no_break
            break  # Nothing else changed

    if new_scan_fgraph is None:
        return

    new_scan_op = scan_op.clone_with_inner_graph(new_scan_fgraph)
    new_outs = new_scan_op.make_node(*node.inputs).outputs
    copy_stack_trace(node.outputs, new_outs)
    return new_outs


@register_specialize
@node_rewriter([blockwise_of(Solve)])
def reuse_decomposition_multiple_solves(fgraph, node):
    return _split_decomp_and_solve_steps(
        fgraph, node, eager=False, allowed_assume_a={"gen", "tridiagonal", "pos"}
    )


@node_rewriter([Scan])
def scan_split_non_sequence_decomposition_and_solve(fgraph, node):
    return _scan_split_non_sequence_decomposition_and_solve(
        fgraph, node, allowed_assume_a={"gen", "tridiagonal", "pos"}
    )


scan_seqopt1.register(
    scan_split_non_sequence_decomposition_and_solve.__name__,
    dfs_rewriter(scan_split_non_sequence_decomposition_and_solve, ignore_newtrees=True),
    "fast_run",
    "scan",
    "scan_pushout",
    position=2,
)


@node_rewriter([Blockwise])
def reuse_decomposition_multiple_solves_jax(fgraph, node):
    return _split_decomp_and_solve_steps(
        fgraph, node, eager=False, allowed_assume_a={"gen", "pos"}
    )


optdb["specialize"].register(
    reuse_decomposition_multiple_solves_jax.__name__,
    dfs_rewriter(reuse_decomposition_multiple_solves_jax, ignore_newtrees=True),
    "jax",
    use_db_name_as_tag=False,
)


@node_rewriter([Scan])
def scan_split_non_sequence_decomposition_and_solve_jax(fgraph, node):
    return _scan_split_non_sequence_decomposition_and_solve(
        fgraph, node, allowed_assume_a={"gen", "pos"}
    )


scan_seqopt1.register(
    scan_split_non_sequence_decomposition_and_solve_jax.__name__,
    dfs_rewriter(
        scan_split_non_sequence_decomposition_and_solve_jax, ignore_newtrees=True
    ),
    "jax",
    use_db_name_as_tag=False,
    position=2,
)


def _load_solve_sylvester():
    # Thin import wrapper to help with testing
    from jax.scipy.linalg import solve_sylvester

    return solve_sylvester


@node_rewriter([SolveBilinearDiscreteLyapunov])
def jax_bilinear_lyapunov_to_direct(fgraph, node):
    """
    Replace SolveBilinearDiscreteLyapunov with a direct computation that is supported by JAX < 0.8
    """
    try:
        _load_solve_sylvester()
        return None

    except ImportError:
        # solve_sylvester is only available in jax > 0.8, which is not available on conda-forge.
        # If it's not available, we can drop back to method="direct"
        A, B = node.inputs
        result = solve_discrete_lyapunov(A, B, method="direct")

        return [result]


optdb.register(
    "jax_bilinear_lyapunov_to_direct",
    dfs_rewriter(jax_bilinear_lyapunov_to_direct),
    "jax",
    position=0.9,  # Run before canonicalization
)
