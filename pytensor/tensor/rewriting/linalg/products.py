from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    node_rewriter,
)
from pytensor.tensor._linalg.constructors import BlockDiagonal
from pytensor.tensor._linalg.products import KroneckerProduct
from pytensor.tensor._linalg.summary import det
from pytensor.tensor.basic import ExtractDiag, concatenate, diag
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.math import outer, prod
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_stabilize,
)


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
