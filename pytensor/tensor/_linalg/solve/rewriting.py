from collections.abc import Container
from copy import copy

from pytensor.compile import optdb
from pytensor.graph import Constant, graph_inputs
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter, node_rewriter
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import scan_seqopt1
from pytensor.tensor._linalg.solve.tridiagonal import (
    tridiagonal_lu_factor,
    tridiagonal_lu_solve,
)
from pytensor.tensor.basic import atleast_Nd
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.slinalg import Solve, cho_solve, cholesky, lu_factor, lu_solve
from pytensor.tensor.variable import TensorVariable


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
                transposed = True  # type: ignore[unreachable]

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
    non_sequences = set(scan_op.inner_non_seqs(scan_op.inner_inputs))
    new_scan_fgraph = scan_op.fgraph

    changed = False
    while True:
        for inner_node in new_scan_fgraph.toposort():
            match (inner_node.op, *inner_node.inputs):
                case (Blockwise(Solve(assume_a=assume_a_var)), A, _b) if (
                    assume_a_var in allowed_assume_a
                ):
                    if all(
                        (isinstance(root_inp, Constant) or (root_inp in non_sequences))
                        for root_inp in graph_inputs([A])
                    ):
                        if new_scan_fgraph is scan_op.fgraph:
                            # Clone the first time to avoid mutating the original fgraph
                            new_scan_fgraph, equiv = new_scan_fgraph.clone_get_equiv()
                            non_sequences = {
                                equiv[non_seq] for non_seq in non_sequences
                            }
                            inner_node = equiv[inner_node]  # type: ignore

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
                        changed = True
                        break  # Break to start over with a fresh toposort
        else:  # no_break
            break  # Nothing else changed

    if not changed:
        return

    # Return a new scan to indicate that a rewrite was done
    new_scan_op = copy(scan_op)
    new_scan_op.fgraph = new_scan_fgraph
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
