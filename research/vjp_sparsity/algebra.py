"""Linear-algebra rewrites that expose exact cancellation in pullbacks.

Each rule is a general identity, not a pullback-specific pattern. Structural facts come from
the assumption system (triangular factors) and from constant masks, so the rules apply to
matrices with static shapes. ``AlgebraRewriter`` alternates them with canonicalization until
the graph stops changing.
"""

from collections import defaultdict
from functools import reduce
from itertools import pairwise

import numpy as np

import pytensor.tensor as pt
from pytensor import scalar as ps
from pytensor.assumptions import LOWER_TRIANGULAR, UPPER_TRIANGULAR, check_assumption
from pytensor.compile.mode import optdb
from pytensor.graph.basic import Constant
from pytensor.graph.rewriting.basic import GraphRewriter, in2out, node_rewriter
from pytensor.printing import debugprint
from pytensor.raise_op import Assert
from pytensor.tensor.basic import Alloc, Join
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.solvers.general import Solve
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import topo_unconditional_constant_folding
from pytensor.tensor.subtensor import Subtensor, get_idx_list
from research.vjp_sparsity.facts import COLUMNS, ROWS, orthonormal_sides


def is_matrix_transpose(variable):
    op = variable.owner_op
    return isinstance(op, DimShuffle) and op.new_order == (1, 0)


def scalar_op_of(variable):
    op = variable.owner_op
    return op.scalar_op if isinstance(op, Elemwise) else None


def static_matrix(variable):
    shape = getattr(variable.type, "shape", None)
    return shape is not None and len(shape) == 2 and None not in shape


def constant_array(variable):
    if isinstance(variable, Constant):
        return np.asarray(variable.data)
    return None


def is_true_constant(variable):
    data = constant_array(variable)
    return data is not None and bool(data.all())


def without_true_asserts(variable):
    while isinstance(variable.owner_op, Assert) and all(
        map(is_true_constant, variable.owner.inputs[1:])
    ):
        variable = variable.owner.inputs[0]
    return variable


def static_slices(variable):
    """The two constant slices of a 2D basic ``Subtensor``, normalized to ``range``s, or None."""
    node = variable.owner
    if not isinstance(node.op, Subtensor) or not static_matrix(node.inputs[0]):
        return None
    entries = get_idx_list(node.inputs, node.op.idx_list)
    entries = list(entries) + [slice(None)] * (2 - len(entries))
    if len(entries) != 2 or not all(isinstance(entry, slice) for entry in entries):
        return None
    ranges = []
    for entry, length in zip(entries, node.inputs[0].type.shape, strict=True):
        bounds = []
        for bound in (entry.start, entry.stop, entry.step):
            if bound is None:
                bounds.append(None)
            elif isinstance(bound, Constant):
                bounds.append(int(bound.data))
            else:
                return None
        ranges.append(range(length)[slice(*bounds)])
    return tuple(ranges)


def as_slice(indices):
    return slice(indices.start, indices.stop, indices.step)


def static_nonzero(fgraph, variable, memo=None):
    """Elements of a static-shaped matrix that can be nonzero for some input value."""
    memo = {} if memo is None else memo
    if variable in memo:
        return memo[variable]
    shape = variable.type.shape
    if (data := constant_array(variable)) is not None:
        result = np.broadcast_to(data != 0, shape)
    elif (
        isinstance(variable.owner_op, Alloc)
        and (value := constant_array(variable.owner.inputs[0])) is not None
    ):
        result = np.broadcast_to(value != 0, shape)
    elif check_assumption(fgraph, variable, UPPER_TRIANGULAR):
        result = np.triu(np.ones(shape, dtype=bool))
    elif check_assumption(fgraph, variable, LOWER_TRIANGULAR):
        result = np.tril(np.ones(shape, dtype=bool))
    elif is_matrix_transpose(variable):
        result = static_nonzero(fgraph, variable.owner.inputs[0], memo).T
    elif isinstance(variable.owner_op, Assert):
        result = static_nonzero(fgraph, variable.owner.inputs[0], memo)
    elif (
        isinstance(variable.owner_op, Subtensor)
        and (ranges := static_slices(variable)) is not None
    ):
        rows, columns = ranges
        result = static_nonzero(fgraph, variable.owner.inputs[0], memo)[
            as_slice(rows), as_slice(columns)
        ]
    elif isinstance(variable.owner_op, Dot):
        left, right = (
            static_nonzero(fgraph, value, memo) for value in variable.owner.inputs
        )
        result = (left.astype(int) @ right.astype(int)) > 0
    elif isinstance(scalar_op_of(variable), ps.Mul) and all(
        map(static_matrix, variable.owner.inputs)
    ):
        result = reduce(
            np.logical_and,
            [static_nonzero(fgraph, value, memo) for value in variable.owner.inputs],
        )
    elif isinstance(scalar_op_of(variable), ps.Add | ps.Sub) and all(
        map(static_matrix, variable.owner.inputs)
    ):
        result = reduce(
            np.logical_or,
            [static_nonzero(fgraph, value, memo) for value in variable.owner.inputs],
        )
    elif isinstance(scalar_op_of(variable), ps.Neg):
        result = static_nonzero(fgraph, variable.owner.inputs[0], memo)
    else:
        result = np.ones(shape, dtype=bool)
    memo[variable] = np.broadcast_to(result, shape)
    return memo[variable]


def signed_terms(variable):
    """Flatten nested Add, Sub, and Neg into (sign, term) pairs."""
    match scalar_op_of(variable):
        case ps.Add():
            return [
                term for value in variable.owner.inputs for term in signed_terms(value)
            ]
        case ps.Sub():
            positive, negative = variable.owner.inputs
            return signed_terms(positive) + [
                (-sign, term) for sign, term in signed_terms(negative)
            ]
        case ps.Neg():
            return [
                (-sign, term) for sign, term in signed_terms(variable.owner.inputs[0])
            ]
        case ps.Mul():
            base, coefficient = coefficient_and_base(variable)
            if (
                base is not variable
                and coefficient.size == 1
                and coefficient.item() == -1
            ):
                return [(-sign, term) for sign, term in signed_terms(base)]
    return [(1, variable)]


def signed_sum(terms):
    total = None
    for sign, term in terms:
        if total is None:
            total = term if sign > 0 else -term
        else:
            total = total + term if sign > 0 else total - term
    return total


def coefficient_and_base(variable):
    """Split ``base * constant`` into (base, constant), or treat ``variable`` as coefficient 1."""
    if isinstance(scalar_op_of(variable), ps.Mul) and len(variable.owner.inputs) == 2:
        first, second = variable.owner.inputs
        for base, coefficient in ((first, second), (second, first)):
            if (data := constant_array(coefficient)) is not None and constant_array(
                base
            ) is None:
                return base, data
    return variable, np.ones(())


def is_sum(variable):
    return isinstance(scalar_op_of(variable), ps.Add | ps.Sub)


def has_static_matrix_terms(node):
    return all(static_matrix(term) for _, term in signed_terms(node.outputs[0]))


@node_rewriter([Assert])
def remove_true_assert(fgraph, node):
    output = node.outputs[0]
    if (stripped := without_true_asserts(output)) is not output:
        return [stripped]


@node_rewriter([Dot])
def empty_contraction_dot(fgraph, node):
    left, _ = node.inputs
    if left.type.shape[-1] == 0 and static_matrix(node.outputs[0]):
        return [pt.zeros(node.outputs[0].type.shape, dtype=node.outputs[0].dtype)]


@node_rewriter([DimShuffle])
def push_transpose_down(fgraph, node):
    [output] = node.outputs
    if not is_matrix_transpose(output):
        return None
    [value] = node.inputs
    if is_matrix_transpose(value):
        return [value.owner.inputs[0]]
    if (data := constant_array(value)) is not None:
        return [pt.constant(data.T)]
    if isinstance(value.owner_op, Dot):
        left, right = value.owner.inputs
        return [pt.dot(right.mT, left.mT)]
    if isinstance(scalar_op_of(value), ps.Add | ps.Sub | ps.Mul | ps.Neg) and all(
        operand.type.ndim == 2 for operand in value.owner.inputs
    ):
        return [value.owner.op(*(operand.mT for operand in value.owner.inputs))]


def product_chain(variable):
    """Flatten nested matrix products into their factors."""
    if isinstance(variable.owner_op, Dot):
        return [
            factor for value in variable.owner.inputs for factor in product_chain(value)
        ]
    return [without_true_asserts(variable)]


def chain_product(factors):
    product = factors[0]
    for factor in factors[1:]:
        product = pt.dot(product, factor)
    return product


def transposed_operand(variable):
    if is_matrix_transpose(variable):
        return without_true_asserts(variable.owner.inputs[0])
    return None


@node_rewriter([Subtensor])
def slice_through_elemwise(fgraph, node):
    """``f(x, y)[s] -> f(x[s], y[s])`` for an elementwise ``f`` on same-shaped matrices."""
    [output] = node.outputs
    value = node.inputs[0]
    if (ranges := static_slices(output)) is None or not isinstance(
        value.owner_op, Elemwise
    ):
        return None
    rows, columns = (as_slice(indices) for indices in ranges)
    operands = []
    for operand in value.owner.inputs:
        if (data := constant_array(operand)) is not None:
            operands.append(
                pt.constant(np.broadcast_to(data, value.type.shape)[rows, columns])
            )
        elif operand.type.shape == value.type.shape:
            operands.append(operand[rows, columns])
        else:
            return None
    return [value.owner.op(*operands)]


@node_rewriter([Subtensor])
def slice_through_dot(fgraph, node):
    """``(x @ y)[r, c] -> x[r, :] @ y[:, c]``."""
    [output] = node.outputs
    value = node.inputs[0]
    if (ranges := static_slices(output)) is None or not isinstance(value.owner_op, Dot):
        return None
    rows, columns = (as_slice(indices) for indices in ranges)
    left, right = value.owner.inputs
    return [pt.dot(left[rows, :], right[:, columns])]


def qr_factors_of(fgraph, variable):
    """``(Q, R)`` of a non-pivoting QR of ``variable`` computed elsewhere in the graph, or None."""
    for client, _ in fgraph.clients.get(variable, ()):
        op = client.op.core_op if isinstance(client.op, Blockwise) else client.op
        if isinstance(op, QR) and op.mode in ("economic", "full") and not op.pivoting:
            return client.outputs[0], client.outputs[1]
    return None


@node_rewriter([Subtensor])
def slice_of_factorized_matrix(fgraph, node):
    """``a[r, c] -> q[r, :] @ r_factor[:, c]`` where ``q, r_factor = qr(a)`` is in the graph.

    Only slices are rewritten, so the QR node's own input is left alone and no cycle forms.
    """
    [output] = node.outputs
    if (ranges := static_slices(output)) is None:
        return None
    factors = qr_factors_of(fgraph, node.inputs[0])
    if factors is None:
        return None
    q, r_factor = factors
    rows, columns = (as_slice(indices) for indices in ranges)
    return [pt.dot(q[rows, :], r_factor[:, columns])]


def slice_points(fgraph, variable, axis):
    """Interior boundaries of constant slices that clients take of ``variable`` along ``axis``."""
    points = set()
    for client, _ in fgraph.clients.get(variable, ()):
        ranges = (
            static_slices(client.outputs[0])
            if isinstance(client.op, Subtensor)
            else None
        )
        if ranges is None or ranges[1 - axis] != range(variable.type.shape[1 - axis]):
            continue
        if ranges[axis].step == 1:
            points.update({ranges[axis].start, ranges[axis].stop})
    return {point for point in points if 0 < point < variable.type.shape[axis]}


def contraction_blocks(variable, axis, start, stop):
    """The ``[start:stop]`` block of ``variable`` along ``axis``, sliced at the transpose's base if any."""
    if is_matrix_transpose(variable):
        base = variable.owner.inputs[0]
        block = base[:, start:stop] if axis == 0 else base[start:stop, :]
        return block.mT
    return variable[start:stop, :] if axis == 0 else variable[:, start:stop]


@node_rewriter([Dot])
def split_contraction_at_slices(fgraph, node):
    """``x @ y -> sum_k x[:, k] @ y[k, :]`` over blocks at which ``x`` or ``y`` is sliced elsewhere.

    Exposes the blocks a pullback works with, so that terms written on blocks of a matrix can
    meet terms written on the whole matrix.
    """
    left, right = node.inputs
    if not (static_matrix(left) and static_matrix(right)):
        return None
    length = left.type.shape[1]
    points = set()
    for variable, axis in ((left, 1), (right, 0)):
        if is_matrix_transpose(variable):
            points |= slice_points(fgraph, variable.owner.inputs[0], 1 - axis)
        else:
            points |= slice_points(fgraph, variable, axis)
    if not points:
        return None
    bounds = [0, *sorted(points), length]
    blocks = [
        pt.dot(
            contraction_blocks(left, 1, start, stop),
            contraction_blocks(right, 0, start, stop),
        )
        for start, stop in pairwise(bounds)
    ]
    return [signed_sum([(1, block) for block in blocks])]


@node_rewriter([Join])
def join_of_adjacent_slices(fgraph, node):
    """``join(axis, x[..., :k], x[..., k:]) -> x``."""
    axis = node.op.axis
    parts = node.inputs
    if not parts or not all(isinstance(part.owner_op, Subtensor) for part in parts):
        return None
    base = parts[0].owner.inputs[0]
    if not static_matrix(base) or any(
        part.owner.inputs[0] is not base for part in parts
    ):
        return None
    position = 0
    for part in parts:
        ranges = static_slices(part)
        if ranges is None or ranges[1 - axis] != range(base.type.shape[1 - axis]):
            return None
        if ranges[axis].step != 1 or ranges[axis].start != position:
            return None
        position = ranges[axis].stop
    if position != base.type.shape[axis]:
        return None
    return [base]


def cancelled_identity_size(fgraph, first, second, memo):
    """Side of the identity ``first @ second`` equals, or None when the pair does not cancel.

    ``X^T @ X`` cancels when ``X`` has orthonormal columns, ``X @ X^T`` when it has orthonormal rows.
    """
    if transposed_operand(first) is second and COLUMNS in orthonormal_sides(
        fgraph, second, memo
    ):
        return second.type.shape[-1]
    if transposed_operand(second) is first and ROWS in orthonormal_sides(
        fgraph, first, memo
    ):
        return first.type.shape[-2]
    return None


def product_or_identity(factors, identity_size, dtype):
    if factors:
        return chain_product(factors)
    if identity_size is None:
        return None
    return pt.eye(identity_size, dtype=dtype)


@node_rewriter([Dot])
def cancel_orthonormal_pairs(fgraph, node):
    """Drop a cancelling ``X^T @ X`` or ``X @ X^T`` pair from a product chain."""
    chain = product_chain(node.outputs[0])
    memo = {}
    for position in range(len(chain) - 1):
        size = cancelled_identity_size(
            fgraph, chain[position], chain[position + 1], memo
        )
        if size is not None:
            rest = chain[:position] + chain[position + 2 :]
            result = product_or_identity(rest, size, node.outputs[0].dtype)
            return None if result is None else [result]
    return None


@node_rewriter([Dot])
def cancel_orthonormal_pairs_into_sum(fgraph, node):
    """``X^T @ (X @ y + z) -> y + X^T @ z`` for orthonormal columns, and the mirror for rows.

    Only the cancelling term leaves the product, so factoring cannot pull it back in.
    """
    left, right = node.inputs
    dtype = node.outputs[0].dtype
    memo = {}
    for sum_side in (1, 0):
        summed, other = (right, left) if sum_side == 1 else (left, right)
        if not is_sum(summed):
            continue
        other_chain = product_chain(other)
        boundary = other_chain[-1] if sum_side == 1 else other_chain[0]
        terms = signed_terms(summed)
        for position, (sign, term) in enumerate(terms):
            term_chain = product_chain(term)
            if sum_side == 1:
                size = cancelled_identity_size(fgraph, boundary, term_chain[0], memo)
                factors = other_chain[:-1] + term_chain[1:]
            else:
                size = cancelled_identity_size(fgraph, term_chain[-1], boundary, memo)
                factors = term_chain[:-1] + other_chain[1:]
            if size is None:
                continue
            cancelled = product_or_identity(factors, size, dtype)
            if cancelled is None:
                continue
            remaining = terms[:position] + terms[position + 1 :]
            if not remaining:
                return [cancelled if sign > 0 else -cancelled]
            rest = signed_sum(remaining)
            rest_product = pt.dot(other, rest) if sum_side == 1 else pt.dot(rest, other)
            return [signed_sum([(1, rest_product), (sign, cancelled)])]
    return None


@node_rewriter([Elemwise])
def factor_common_operand(fgraph, node):
    """``a @ x +- a @ y -> a @ (x +- y)``, and likewise for a shared right factor."""
    if not is_sum(node.outputs[0]) or not has_static_matrix_terms(node):
        return None
    terms = signed_terms(node.outputs[0])
    chains = [(sign, product_chain(term)) for sign, term in terms]
    for side in (0, -1):
        groups = defaultdict(list)
        for position, (_, chain) in enumerate(chains):
            if len(chain) > 1:
                groups[chain[side]].append(position)
        shared = next(
            (positions for positions in groups.values() if len(positions) > 1), None
        )
        if shared is None:
            continue
        common = chains[shared[0]][1][side]
        remainders = []
        for position in shared:
            sign, chain = chains[position]
            remainders.append(
                (sign, chain_product(chain[1:] if side == 0 else chain[:-1]))
            )
        rest = signed_sum(remainders)
        factored = pt.dot(common, rest) if side == 0 else pt.dot(rest, common)
        others = [
            terms[position] for position in range(len(terms)) if position not in shared
        ]
        return [signed_sum([(1, factored), *others])]
    return None


@node_rewriter([Elemwise])
def collect_constant_coefficients(fgraph, node):
    """``x * c1 +- x * c2 -> x * (c1 +- c2)`` for constant arrays ``c1`` and ``c2``."""
    [output] = node.outputs
    if not is_sum(output) or not has_static_matrix_terms(node):
        return None
    groups = defaultdict(list)
    dropped = False
    memo = {}
    for sign, term in signed_terms(output):
        if not static_nonzero(fgraph, term, memo).any():
            dropped = True
            continue
        base, coefficient = coefficient_and_base(term)
        groups[base].append(sign * coefficient)
    if not dropped and all(len(coefficients) < 2 for coefficients in groups.values()):
        return None
    terms = []
    for base, coefficients in groups.items():
        coefficient = np.broadcast_to(sum(coefficients), output.type.shape)
        if not coefficient.any():
            continue
        terms.append(
            (1, base if (coefficient == 1).all() else base * pt.constant(coefficient))
        )
    if not terms:
        return [pt.zeros(output.type.shape, dtype=output.dtype)]
    return [signed_sum(terms)]


@node_rewriter([Dot])
def pull_scalar_out_of_dot(fgraph, node):
    """``x @ (c * y) -> c * (x @ y)`` for a constant ``c`` with a single element."""
    left, right = node.inputs
    for position, operand in enumerate((left, right)):
        base, coefficient = coefficient_and_base(operand)
        if base is not operand and coefficient.size == 1:
            operands = [left, right]
            operands[position] = base
            return [
                pt.constant(coefficient.item(), dtype=node.outputs[0].dtype)
                * pt.dot(*operands)
            ]
    return None


@node_rewriter([Elemwise])
def factor_common_coefficient(fgraph, node):
    """``x * c +- y * c -> (x +- y) * c`` for the same constant array ``c``."""
    [output] = node.outputs
    if not is_sum(output) or not has_static_matrix_terms(node):
        return None
    groups = []
    others = []
    for sign, term in signed_terms(output):
        base, coefficient = coefficient_and_base(term)
        if base is term or coefficient.ndim == 0:
            others.append((sign, term))
            continue
        for group_coefficient, members in groups:
            if (
                group_coefficient.shape == coefficient.shape
                and (group_coefficient == coefficient).all()
            ):
                members.append((sign, base))
                break
        else:
            groups.append((coefficient, [(sign, base)]))
    if all(len(members) < 2 for _, members in groups):
        return None
    terms = list(others)
    for coefficient, members in groups:
        if len(members) > 1:
            terms.append((1, signed_sum(members) * pt.constant(coefficient)))
        else:
            sign, base = members[0]
            terms.append((sign, base * pt.constant(coefficient)))
    return [signed_sum(terms)]


@node_rewriter([Elemwise])
def mask_from_structure(fgraph, node):
    """Apply a constant mask ``c`` to each term of ``base * c`` that the structure decides.

    A term whose possible nonzeros all sit where ``c`` is 1 passes through unmasked, and one whose
    possible nonzeros all sit where ``c`` is 0 is dropped. The rewrite applies only if some term is
    decided.
    """
    [output] = node.outputs
    base, coefficient = coefficient_and_base(output)
    if coefficient.ndim == 0 or base is output or not static_matrix(output):
        return None
    coefficient = np.broadcast_to(coefficient, output.type.shape)
    mask = pt.constant(coefficient)
    memo = {}
    terms, decided = [], False
    for sign, term in signed_terms(base):
        if not static_matrix(term) or term.type.shape != output.type.shape:
            return None
        nonzero = static_nonzero(fgraph, term, memo)
        if (coefficient[nonzero] == 1).all():
            terms.append((sign, term))
            decided = True
        elif (coefficient[nonzero] == 0).all():
            decided = True
        else:
            terms.append((sign, term * mask))
    if not decided:
        return None
    if not terms:
        return [pt.zeros(output.type.shape, dtype=output.dtype)]
    return [signed_sum(terms)]


def solve_parts(node):
    op = node.op.core_op if isinstance(node.op, Blockwise) else node.op
    if not isinstance(op, Solve | SolveTriangular) or op.b_ndim != 2:
        return None
    return op, *node.inputs


def solver_reads_whole_matrix(fgraph, op, matrix):
    """Whether ``op`` uses every entry of ``matrix`` that can be nonzero.

    A triangular solve ignores the other triangle, and the diagonal too when it assumes a unit
    diagonal. ``Solve`` reads only part of the matrix unless it assumes a general one.
    """
    if isinstance(op, Solve):
        return op.assume_a == "gen"
    if op.unit_diagonal or not static_matrix(matrix):
        return False
    read = np.tri(*matrix.type.shape, dtype=bool)
    if not op.lower:
        read = ~np.tri(*matrix.type.shape, k=-1, dtype=bool)
    return not (static_nonzero(fgraph, matrix) & ~read).any()


@node_rewriter([Blockwise, Solve, SolveTriangular])
def solve_of_product(fgraph, node):
    """``solve(a, a @ x) -> x`` when the solver reads all of ``a``, and ``solve(a, -b) -> -solve(a, b)``."""
    if (parts := solve_parts(node)) is None:
        return None
    op, matrix, rhs = parts
    if (
        isinstance(rhs.owner_op, Dot)
        and without_true_asserts(rhs.owner.inputs[0]) is without_true_asserts(matrix)
        and solver_reads_whole_matrix(fgraph, op, matrix)
    ):
        return [rhs.owner.inputs[1]]
    if isinstance(scalar_op_of(rhs), ps.Neg):
        return [-node.op(matrix, rhs.owner.inputs[0])]
    base, coefficient = coefficient_and_base(rhs)
    if base is not rhs and coefficient.size == 1:
        return [pt.constant(coefficient.item()) * node.op(matrix, base)]


ALGEBRA_REWRITES = [
    remove_true_assert,
    empty_contraction_dot,
    push_transpose_down,
    factor_common_operand,
    collect_constant_coefficients,
    mask_from_structure,
    solve_of_product,
    cancel_orthonormal_pairs,
    cancel_orthonormal_pairs_into_sum,
    slice_through_elemwise,
    slice_through_dot,
    slice_of_factorized_matrix,
    split_contraction_at_slices,
    join_of_adjacent_slices,
    pull_scalar_out_of_dot,
    factor_common_coefficient,
]


class AlgebraRewriter(GraphRewriter):
    """Alternate the algebra rewrites with canonicalization until the graph is unchanged."""

    def __init__(self, max_rounds=20):
        self.max_rounds = max_rounds

    def apply(self, fgraph):
        canonicalize = optdb.query("+canonicalize")
        algebra = in2out(*ALGEBRA_REWRITES, ignore_newtrees=False)
        topo_unconditional_constant_folding.rewrite(fgraph)
        previous = None
        for _ in range(self.max_rounds):
            canonicalize.rewrite(fgraph)
            algebra.rewrite(fgraph)
            signature = debugprint(fgraph, file="str", print_type=False)
            if signature == previous:
                return
            previous = signature
