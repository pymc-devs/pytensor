from typing import TYPE_CHECKING

import numpy as np

from pytensor.graph.rewriting.basic import (
    GraphRewriter,
    copy_stack_trace,
    node_rewriter,
)
from pytensor.tensor.basic import ExtractDiag, concatenate, diag
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.products import (
    KroneckerProduct,
    MultiDot,
    matrix_dot,
)
from pytensor.tensor.linalg.summary import det
from pytensor.tensor.math import Dot, matmul, outer, prod
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)


if TYPE_CHECKING:
    from pytensor.tensor.type import TensorVariable


def _is_dot_node(node):
    """Check if a node is a Dot or Blockwise(Dot)"""
    if isinstance(node.op, Dot):
        return True
    if isinstance(node.op, Blockwise) and isinstance(node.op.core_op, Dot):
        return True
    return False


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


class MultiDotAbsorber(GraphRewriter):
    """Scan the entire fgraph for chains of 3+ matmul ops and wrap them in MultiDot.

    Matches both bare Dot (2-D) and Blockwise(Dot) (batched) nodes.
    Only absorbs nodes whose output has a single client (the next matmul in the chain).
    """

    def apply(self, fgraph):
        dot_nodes = [node for node in fgraph.toposort() if _is_dot_node(node)]
        absorbed = set()

        for node in dot_nodes:
            if node in absorbed:
                continue

            chain_nodes = [node]
            current = node

            # Walk backward: the left input might be another matmul
            while True:
                left_input = current.inputs[0]
                if (
                    left_input.owner
                    and _is_dot_node(left_input.owner)
                    and left_input.owner not in absorbed
                    and len(fgraph.clients[left_input]) == 1
                ):
                    current = left_input.owner
                    chain_nodes.insert(0, current)
                else:
                    break

            # Walk forward: the output might feed into another matmul as left input
            current = node
            while True:
                out = current.outputs[0]
                clients = fgraph.clients[out]
                if (
                    len(clients) == 1
                    and clients[0][0] != "output"
                    and _is_dot_node(clients[0][0])
                    and clients[0][0].inputs[0] is out
                    and clients[0][0] not in absorbed
                ):
                    current = clients[0][0]
                    chain_nodes.append(current)
                else:
                    break

            if len(chain_nodes) < 2:
                continue

            # Extract matrices: first node's left input, then each node's right input
            matrices = [
                chain_nodes[0].inputs[0],
                *(node.inputs[1] for node in chain_nodes),
            ]
            absorbed.update(chain_nodes)

            inner_inputs = [
                m.type.clone()(name=f"i{i}") for i, m in enumerate(matrices)
            ]

            # Naive chained matmuls -- optimization is done later by a separate rewrite
            inner_output = matrix_dot(*inner_inputs)
            multi_dot_op = MultiDot(inputs=inner_inputs, outputs=[inner_output])
            new_out = multi_dot_op(*matrices)
            copy_stack_trace(chain_nodes[-1].outputs[0], new_out)
            fgraph.replace(
                chain_nodes[-1].outputs[0],
                new_out,
                reason="dot_chain_to_multi_dot",
            )


multi_dot_absorber = MultiDotAbsorber()
multi_dot_absorber.name = "multi_dot_absorber"
register_canonicalize(multi_dot_absorber, "multi_dot", name="multi_dot_absorber")  # type: ignore[arg-type]


@register_canonicalize("multi_dot")
@node_rewriter([MultiDot])
def fuse_multi_dot_operands(fgraph, node):
    """Fuse matmul(MultiDot(...), X) or matmul(X, MultiDot(...)) into a larger MultiDot.

    Also handles matmul(MultiDot(...), MultiDot(...)) by merging both.
    """
    [multi_out] = node.outputs
    clients = fgraph.clients[multi_out]

    for client_node, _ in clients:
        if client_node == "output" or not _is_dot_node(client_node):
            continue

        left, right = client_node.inputs

        # Only absorb a MultiDot whose output has a single client (this Dot),
        # otherwise we'd duplicate the computation of its inner chain.
        left_is_multi = (
            left.owner
            and isinstance(left.owner.op, MultiDot)
            and len(fgraph.clients[left]) == 1
        )
        right_is_multi = (
            right.owner
            and isinstance(right.owner.op, MultiDot)
            and len(fgraph.clients[right]) == 1
        )

        if not left_is_multi and not right_is_multi:
            continue

        matrices = []
        if left_is_multi:
            matrices.extend(left.owner.inputs)
        else:
            matrices.append(left)
        if right_is_multi:
            matrices.extend(right.owner.inputs)
        else:
            matrices.append(right)

        inner_inputs = [m.type.clone()(name=f"i{i}") for i, m in enumerate(matrices)]
        inner_output = matrix_dot(*inner_inputs)
        new_out = MultiDot(inputs=inner_inputs, outputs=[inner_output])(*matrices)

        copy_stack_trace(client_node.outputs[0], new_out)

        return {client_node.outputs[0]: new_out}

    return None


@register_canonicalize("multi_dot")
@node_rewriter([MultiDot])
def flatten_nested_multi_dot(fgraph, node):
    """Flatten nested MultiDot inputs into a single MultiDot. Nested MultiDots arise from fuse_multi_dot_operands,
    when graphs like A @ B @ MultiDot(C, D, E, F) -> MultiDot(A, B, MultiDot(C, D, E, F)). This rewrite does a final
    clean up: MultiDot([A, B, MultiDot([C, D, E, F])]) -> MultiDot([A, B, C, D, E, F])
    """
    inputs = node.inputs
    has_nested = any(
        inp.owner
        and isinstance(inp.owner.op, MultiDot)
        and len(fgraph.clients[inp]) == 1
        for inp in inputs
    )

    if not has_nested:
        return None

    # Flatten inputs
    matrices = []
    for inp in inputs:
        if (
            inp.owner
            and isinstance(inp.owner.op, MultiDot)
            and len(fgraph.clients[inp]) == 1
        ):
            matrices.extend(inp.owner.inputs)
        else:
            matrices.append(inp)

    inner_inputs = []
    for m in matrices:
        if m in inner_inputs:
            # If a matrix is repeated, we need to copy it to avoid issues with the OpFromGraph
            inner_inputs.append(m.type(name=m.name))
        else:
            inner_inputs.append(m)

    inner_output = matrix_dot(*inner_inputs)
    new_out = MultiDot(inputs=inner_inputs, outputs=[inner_output])(*matrices)

    copy_stack_trace(node.outputs[0], new_out)
    return [new_out]


def matrix_chain_split_points(dimensions: list[int]) -> np.ndarray:
    """Return optimal split points for matrix-chain multiplication.

    Parameters
    ----------
    dimensions : list[int]
        Matrix dimensions encoded as a list of integers, where the i-th matrix has shape
        (dimensions[i], dimensions[i+1])

    Returns
    -------
    split_at : np.ndarray
        Integer array of shape (n, n), where n = len(dimensions) - 1.

        ``split_at[i, j]`` gives the index ``k`` such that the optimal way
        to parenthesize the subchain A_i ... A_j is:

            (A_i ... A_k) @ (A_{k+1} ... A_j)

        Entries on and below the diagonal are unused.
    """
    num_matrices = len(dimensions) - 1

    min_cost = np.zeros((num_matrices, num_matrices), dtype=np.float64)
    split_at = np.zeros((num_matrices, num_matrices), dtype=np.intp)

    # subchain_length is the number of matrices in the subchain minus 1:
    # 1 means pairs (Ai..A{i+1}), 2 means triples, etc.
    for subchain_length in range(1, num_matrices):
        for start in range(num_matrices - subchain_length):
            end = start + subchain_length

            best_cost = np.inf
            best_split = -1

            for split in range(start, end):
                candidate_cost = (
                    min_cost[start, split]
                    + min_cost[split + 1, end]
                    + dimensions[start] * dimensions[split + 1] * dimensions[end + 1]
                )

                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_split = split

            min_cost[start, end] = best_cost
            split_at[start, end] = best_split

    return split_at


def _build_optimal_matmul_tree(
    matrices: list[TensorVariable], split_at: np.ndarray, start: int, end: int
):
    """Return the optimal parenthesization of ``matrices[start:end+1]``.

    Parameters
    ----------
    matrices : list of TensorVariable
        Sequence of matrix expressions.
    split_at : np.ndarray
        DP split table where ``split_at[a, b]`` gives the index at which
        to split the subchain ``matrices[a:b+1]``.
    start, end : int
        Inclusive bounds of the subchain to reconstruct.
    """
    if start == end:
        return matrices[start]

    split = split_at[start, end]
    return matmul(
        _build_optimal_matmul_tree(matrices, split_at, start, split),
        _build_optimal_matmul_tree(matrices, split_at, split + 1, end),
    )


@register_specialize("multi_dot")
@node_rewriter([MultiDot])
def lower_multi_dot(fgraph, node):
    """Lower MultiDot to an optimally-ordered sequence of matmul ops.

    Core dimensions of each input are assumed to be the last two. All core dimensions must be statically known for
    the optimization to fire. If anything is unknown, fall back to left-to-right ordering.
    """
    matrices = node.inputs
    n = len(matrices)

    # Core shapes are encoded in matrix dims so that input_1.core_shape = matrix_dims[[0, 1]],
    # input_2.core_shape = matrix_dims[[1, 2]], and so on.
    matrix_dimensions = [
        matrices[0].type.shape[-2],
        *[a.type.shape[-1] for a in matrices],
    ]
    if any(d is None for d in matrix_dimensions):
        new_out = matrix_dot(*matrices)
        copy_stack_trace(node.outputs[0], new_out)
        return [new_out]

    split_points = matrix_chain_split_points(matrix_dimensions)
    new_out = _build_optimal_matmul_tree(
        list(matrices), split_points, start=0, end=n - 1
    )
    copy_stack_trace(node.outputs[0], new_out)

    return [new_out]
