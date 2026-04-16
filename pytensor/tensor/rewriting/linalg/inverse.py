from pytensor import tensor as pt
from pytensor.graph import Apply, Constant, FunctionGraph
from pytensor.graph.rewriting.basic import (
    node_rewriter,
)
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.tensor.basic import AllocDiag, Eye
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.constructors import BlockDiagonal, block_diag
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse, MatrixPinv
from pytensor.tensor.linalg.products import KroneckerProduct, kron
from pytensor.tensor.linalg.solvers.general import solve
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.rewriting.linalg.utils import MATRIX_INVERSE_OPS, is_eye_mul


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
            return [kron(*(outer_op(m) for m in inner_matrices))]
