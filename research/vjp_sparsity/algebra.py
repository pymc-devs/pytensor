"""Numerical identities applied before nonzero abstraction."""

import pytensor.tensor as pt
from pytensor import scalar as ps
from pytensor.assumptions import UPPER_TRIANGULAR, check_assumption
from pytensor.graph.basic import equal_computations
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.blockwise import local_useless_unbatched_blockwise


nonzero_algebra_db = EquilibriumDB(ignore_newtrees=False)


def _transpose_input(variable):
    match variable.owner_op_and_inputs:
        case DimShuffle(is_matrix_transpose=True), value:
            return value


def _half_lower(value):
    return pt.tril(value) - pt.diag(pt.diagonal(value) / 2.0)


@node_rewriter([SolveTriangular])
def local_projected_conjugate_solve(fgraph, node):
    """Expose solve(T, T @ V) behind a half-diagonal lower projection.

    For invertible upper-triangular T, Phi(T @ V) = Phi(T @ tril(V)).
    Writing Phi = Id - Psi then cancels the unprojected left solve:
    T^-1 Phi(T @ V) T^-T = (tril(V) - T^-1 Psi(T @ tril(V))) T^-T.
    Psi keeps the strict upper triangle and halves the diagonal.
    """
    op = node.op
    if op.lower or op.unit_diagonal or op.b_ndim != 2 or node.outputs[0].ndim != 2:
        return None
    matrix, rhs = node.inputs
    inner = _transpose_input(rhs)
    if inner is None or inner.owner_op != node.op:
        return None
    inner_matrix, inner_rhs = inner.owner.inputs
    if not equal_computations([matrix], [inner_matrix]):
        return None
    projected = _transpose_input(inner_rhs)
    if projected is None:
        return None
    match projected.owner_op_and_inputs:
        case Elemwise(scalar_op=ps.Sub()), lower, _:
            pass
        case _:
            return None
    match lower.owner_op_and_inputs:
        case Elemwise(scalar_op=ps.Mul()), *factors:
            pass
        case _:
            return None
    for product in factors:
        match product.owner_op_and_inputs:
            case Dot(), left, direction:
                pass
            case _:
                continue
        if not equal_computations([matrix], [left]):
            continue
        if not equal_computations([projected], [_half_lower(product)]):
            continue
        if not check_assumption(fgraph, matrix, UPPER_TRIANGULAR):
            return None
        direction = pt.tril(direction)
        product = matrix @ direction
        complement = pt.triu(product, 1) + pt.diag(pt.diagonal(product) / 2.0)
        left_solved = direction - node.op(matrix, complement)
        return [copy_stack_trace(node.outputs[0], node.op(matrix, left_solved.T).T)]


nonzero_algebra_db.register(
    "local_useless_unbatched_blockwise", local_useless_unbatched_blockwise, "basic"
)
nonzero_algebra_db.register(
    "local_projected_conjugate_solve", local_projected_conjugate_solve, "basic"
)
