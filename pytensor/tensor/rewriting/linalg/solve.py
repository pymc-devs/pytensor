from pytensor import tensor as pt
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    node_rewriter,
)
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.tensor._linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor._linalg.solve.core import SolveBase
from pytensor.tensor._linalg.solve.general import Solve
from pytensor.tensor._linalg.solve.psd import CholeskySolve
from pytensor.tensor._linalg.solve.triangular import SolveTriangular, solve_triangular
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of


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
