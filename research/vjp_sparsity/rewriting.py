"""Infer VJP nonzeros with rewrites dispatched on the producing Ops."""

from functools import reduce, wraps

import numpy as np
import scipy.sparse

import pytensor.tensor as pt
from pytensor import scalar as ps
from pytensor.assumptions import LOWER_TRIANGULAR, UPPER_TRIANGULAR, check_assumption
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import optdb
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import in2out, node_rewriter
from pytensor.graph.rewriting.db import EquilibriumDB, RewriteDatabaseQuery, SequenceDB
from pytensor.graph.rewriting.unify import OpPattern
from pytensor.scan.basic import scan
from pytensor.sparse.basic import DenseFromSparse, SparseFromDense
from pytensor.sparse.math import StructuredDot
from pytensor.tensor.basic import (
    Alloc,
    AllocDiag,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    Split,
    alloc_diag,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.constructors import BlockDiagonal
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, CholeskyGrad
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.solvers.general import Solve
from pytensor.tensor.linalg.solvers.psd import CholeskySolve
from pytensor.tensor.linalg.solvers.triangular import SolveTriangular
from pytensor.tensor.math import Dot, Prod, Sum
from pytensor.tensor.rewriting.basic import elemwise_of
from pytensor.tensor.rewriting.blockwise import local_useless_unbatched_blockwise
from pytensor.tensor.rewriting.shape import ShapeOptimizer
from pytensor.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedSubtensor,
    IncSubtensor,
    Subtensor,
)
from pytensor.tensor.type import TensorType


class MayBeNonzero(Op):
    """Structural query, distinct from an existing runtime nonzero predicate."""

    __props__ = ()

    def make_node(self, value):
        return Apply(self, [value], [TensorType("bool", shape=value.type.shape)()])

    def perform(self, node, inputs, outputs):
        raise RuntimeError("MayBeNonzero must be rewritten before evaluation")

    def infer_shape(self, node, shapes):
        return shapes

    def do_constant_folding(self, fgraph, node):
        return False


may_be_nonzero = MayBeNonzero()
nonzero_lift_db = EquilibriumDB(ignore_newtrees=False)
nonzero_rewrites_db = SequenceDB()
nonzero_basic_query = RewriteDatabaseQuery(include=["basic"])


def register_nonzero(tracks):
    """Register a producer rule returning a mask for that Op's output."""

    def register(fn):
        @node_rewriter(tracks)
        @wraps(fn)
        def rewrite(fgraph, node):
            queries = [
                [
                    client.outputs[0]
                    for client, _ in fgraph.clients[output]
                    if isinstance(client, Apply) and isinstance(client.op, MayBeNonzero)
                ]
                for output in node.outputs
            ]
            if not any(queries):
                return None
            masks = fn(fgraph, node)
            if masks is not None:
                masks = [masks] if isinstance(masks, Variable) else masks
                return {
                    query: mask
                    for clients, mask in zip(queries, masks, strict=True)
                    for query in clients
                }

        nonzero_lift_db.register(rewrite.__name__, rewrite, "basic")
        return rewrite

    return register


def _shape_only(variables, *, allow_integer_inputs=False):
    """Whether values are fixed by constants/shapes, optionally integer indices.

    A sort permutation depends on floating-point data and must stay conservative.
    Shape-derived slice bounds and held-fixed integer indices can be used directly.
    """
    pending = list(variables)
    seen = set()
    while pending:
        variable = pending.pop()
        if variable in seen:
            continue
        seen.add(variable)
        if isinstance(variable, Constant) or isinstance(
            variable.owner_op, Shape | Shape_i
        ):
            continue
        if variable.owner_op is None:
            dtype = getattr(variable.type, "dtype", "")
            if allow_integer_inputs and (
                dtype.startswith(("int", "uint")) or dtype == "bool"
            ):
                continue
            return False
        pending.extend(variable.owner.inputs)
    return True


@register_nonzero([elemwise_of(ps.Mul)])
def local_nonzero_mul(fgraph, node):
    return reduce(pt.and_, map(may_be_nonzero, node.inputs))


@register_nonzero([elemwise_of(ps.Add), elemwise_of(ps.Sub)])
def local_nonzero_add_sub(fgraph, node):
    if _shape_only(node.inputs):
        return pt.neq(node.outputs[0], 0)
    return reduce(pt.or_, map(may_be_nonzero, node.inputs))


@register_nonzero([elemwise_of(ps.TrueDiv)])
def local_nonzero_div(fgraph, node):
    numerator, denominator = node.inputs
    return pt.second(may_be_nonzero(denominator), may_be_nonzero(numerator))


@register_nonzero([elemwise_of(ps.Second)])
def local_nonzero_second(fgraph, node):
    shape_source, value = node.inputs
    return pt.second(may_be_nonzero(shape_source), may_be_nonzero(value))


@register_nonzero([elemwise_of(ps.Switch)])
def local_nonzero_switch(fgraph, node):
    condition, yes, no = node.inputs
    if _shape_only([condition]):
        return pt.where(condition, may_be_nonzero(yes), may_be_nonzero(no))
    return pt.second(
        may_be_nonzero(condition), may_be_nonzero(yes) | may_be_nonzero(no)
    )


@register_nonzero([elemwise_of(ps.LogicalComparison)])
def local_nonzero_shape_comparison(fgraph, node):
    if _shape_only(node.inputs):
        return node.outputs[0]


@register_nonzero(
    [
        elemwise_of(ps.Cast),
        elemwise_of(OpPattern(ps.UnaryScalarOp, preserves_zero=True)),
    ]
)
def local_nonzero_zero_preserving(fgraph, node):
    return may_be_nonzero(node.inputs[0])


@register_nonzero([DimShuffle])
def local_nonzero_dimshuffle(fgraph, node):
    return node.op(may_be_nonzero(node.inputs[0]))


@register_nonzero([Reshape, SpecifyShape])
def local_nonzero_reshape(fgraph, node):
    value, *shape = node.inputs
    return node.op(may_be_nonzero(value), *shape)


@register_nonzero([Alloc])
def local_nonzero_alloc(fgraph, node):
    value, *shape = node.inputs
    return pt.broadcast_to(may_be_nonzero(value), tuple(shape))


@register_nonzero([Sum, Prod])
def local_nonzero_reduce(fgraph, node):
    reduction = pt.any if isinstance(node.op, Sum) else pt.all
    return reduction(may_be_nonzero(node.inputs[0]), axis=node.op.axis)


@register_nonzero([MakeVector, Join])
def local_nonzero_join(fgraph, node):
    if isinstance(node.op, MakeVector):
        return pt.stack([may_be_nonzero(value) for value in node.inputs])
    return pt.concatenate(
        [may_be_nonzero(value) for value in node.inputs], axis=node.op.axis
    )


@register_nonzero([Split])
def local_nonzero_split(fgraph, node):
    value, *parameters = node.inputs
    return node.op.make_node(may_be_nonzero(value), *parameters).outputs


@register_nonzero([Eye])
def local_nonzero_eye(fgraph, node):
    rows, columns, offset = node.inputs
    return pt.eye(rows, columns, k=offset, dtype="bool")


@register_nonzero([ExtractDiag])
def local_nonzero_diagonal(fgraph, node):
    return node.op(may_be_nonzero(node.inputs[0]))


@register_nonzero([AllocDiag, OpPattern(Blockwise, core_op=OpPattern(AllocDiag))])
def local_nonzero_alloc_diagonal(fgraph, node):
    op = node.op.core_op if isinstance(node.op, Blockwise) else node.op
    mask = may_be_nonzero(node.inputs[0])
    batch_dims = mask.ndim - 1 if isinstance(node.op, Blockwise) else 0
    return alloc_diag(
        mask,
        offset=op.offset,
        axis1=op.axis1 + batch_dims,
        axis2=op.axis2 + batch_dims,
    )


@register_nonzero(
    [BlockDiagonal, OpPattern(Blockwise, core_op=OpPattern(BlockDiagonal))]
)
def local_nonzero_block_diagonal(fgraph, node):
    return node.op(*map(may_be_nonzero, node.inputs))


@register_nonzero([Subtensor, AdvancedSubtensor])
def local_nonzero_subtensor(fgraph, node):
    value, *indices = node.inputs
    if _shape_only(indices, allow_integer_inputs=True):
        return node.op(may_be_nonzero(value), *indices)
    return pt.broadcast_to(pt.any(may_be_nonzero(value)), node.outputs[0].shape)


@register_nonzero([IncSubtensor, AdvancedIncSubtensor])
def local_nonzero_scatter(fgraph, node):
    base, update, *indices = node.inputs
    if _shape_only(indices, allow_integer_inputs=True):
        return node.op(may_be_nonzero(base), may_be_nonzero(update), *indices)
    possible = pt.any(may_be_nonzero(base)) | pt.any(may_be_nonzero(update))
    return pt.broadcast_to(possible, node.outputs[0].shape)


@register_nonzero([DenseFromSparse, SparseFromDense])
def local_nonzero_sparse_conversion(fgraph, node):
    return may_be_nonzero(node.inputs[0])


@register_nonzero([Dot, OpPattern(Blockwise, core_op=OpPattern(Dot))])
def local_nonzero_dot(fgraph, node):
    left, right = node.inputs
    return pt.matmul(may_be_nonzero(left), may_be_nonzero(right))


@register_nonzero([StructuredDot])
def local_nonzero_sparse_dot(fgraph, node):
    possible = reduce(pt.and_, (pt.any(may_be_nonzero(value)) for value in node.inputs))
    return pt.broadcast_to(possible, node.outputs[0].shape)


@register_nonzero([OpFromGraph])
def local_nonzero_inner_graph(fgraph, node):
    inner = node.op.fgraph.unfreeze()
    outputs = clone_replace(
        inner.outputs, replace=dict(zip(inner.inputs, node.inputs, strict=True))
    )
    return [may_be_nonzero(output) for output in outputs]


def _reachability(structure):
    """Conservative inverse structure on the domain of invertible matrices."""
    size = structure.shape[-1]
    initial = structure | pt.eye(size, dtype="bool")
    steps = pt.cast(pt.ceil(pt.log2(pt.maximum(size, 1))), "int64")
    powers = scan(
        lambda _, previous: pt.matmul(previous, previous),
        sequences=pt.arange(steps),
        outputs_info=initial,
        return_updates=False,
    )
    return pt.concatenate([initial[None], powers], axis=0)[-1]


def _triangle_mask(structure, lower):
    rows, columns = structure.shape[-2:]
    keep = (
        pt.tri(rows, columns, dtype="bool")
        if lower
        else ~pt.tri(rows, columns, k=-1, dtype="bool")
    )
    return structure & keep


@register_nonzero([Cholesky, OpPattern(Blockwise, core_op=OpPattern(Cholesky))])
def local_nonzero_cholesky(fgraph, node):
    """Boolean elimination fill-in for the factor's values, not its derivative."""
    op = node.op.core_op if isinstance(node.op, Blockwise) else node.op
    structure = _triangle_mask(may_be_nonzero(node.inputs[0]), op.lower)
    initial = structure | structure.swapaxes(-1, -2)
    size = structure.shape[-1]
    initial |= pt.eye(size, dtype="bool")
    indices = pt.arange(size)

    def eliminate(column, previous):
        neighbors = previous[..., :, column] & (indices > column)
        return previous | (neighbors[..., :, None] & neighbors[..., None, :])

    filled = scan(
        eliminate, sequences=indices, outputs_info=initial, return_updates=False
    )
    result = pt.concatenate([initial[None], filled], axis=0)[-1]
    return _triangle_mask(result, op.lower)


@register_nonzero([CholeskyGrad])
def local_nonzero_cholesky_grad(fgraph, node):
    """Reverse the column recurrence using factor and cotangent value masks."""
    triangle = LOWER_TRIANGULAR if node.op.lower else UPPER_TRIANGULAR
    if not check_assumption(fgraph, node.inputs[0], triangle):
        return None
    factor, direction = map(may_be_nonzero, node.inputs)
    if not node.op.lower:
        factor, direction = factor.T, direction.T
    size = factor.shape[0]
    rows = pt.arange(size)
    initial = _triangle_mask(direction, lower=True)

    def column(j, adjoint, factor):
        before, after = rows < j, rows > j
        off_diagonal = adjoint[:, j] & after
        diagonal = adjoint[j, j] | pt.any(off_diagonal & factor[:, j])
        adjoint |= off_diagonal[:, None] & factor[j][None, :] & before[None, :]
        row_update = pt.any(off_diagonal[:, None] & factor, axis=0)
        row_update |= diagonal & factor[j]
        adjoint = adjoint[j].set(adjoint[j] | (row_update & before))
        return adjoint[:, j].set(off_diagonal | (pt.eq(rows, j) & diagonal))

    adjoints = scan(
        column,
        sequences=rows[::-1],
        outputs_info=initial,
        non_sequences=[factor],
        return_updates=False,
    )
    result = pt.concatenate([initial[None], adjoints], axis=0)[-1]
    return result if node.op.lower else result.T


@register_nonzero([QR, OpPattern(Blockwise, core_op=OpPattern(QR))])
def local_nonzero_qr(fgraph, node):
    """Factor value masks from boolean Gram-Schmidt on the valid QR domain."""
    op = node.op.core_op if isinstance(node.op, Blockwise) else node.op
    if op.mode != "economic" or op.pivoting:
        return None
    structure = may_be_nonzero(node.inputs[0])
    rows, columns = structure.shape[-2:]
    size = pt.minimum(rows, columns)
    initial = pt.zeros((*structure.shape[:-2], rows, size), dtype="bool")

    def orthogonalize(column, previous, structure):
        value = structure[..., :, column]
        overlaps = pt.any(previous & value[..., :, None], axis=-2)
        possible = value | pt.any(previous & overlaps[..., None, :], axis=-1)
        return previous[..., :, column].set(possible)

    columns_mask = scan(
        orthogonalize,
        sequences=pt.arange(size),
        outputs_info=initial,
        non_sequences=[structure],
        return_updates=False,
    )
    q = pt.concatenate([initial[None], columns_mask], axis=0)[-1]
    r = _triangle_mask(pt.matmul(q.swapaxes(-1, -2), structure), lower=False)
    return [q, r]


@register_nonzero(
    [
        Solve,
        SolveTriangular,
        CholeskySolve,
        OpPattern(
            Blockwise, core_op=OpPattern(Solve | SolveTriangular | CholeskySolve)
        ),
    ]
)
def local_nonzero_solve(fgraph, node):
    op = node.op.core_op if isinstance(node.op, Blockwise) else node.op
    matrix, rhs = node.inputs
    structure = may_be_nonzero(matrix)
    if isinstance(op, SolveTriangular | CholeskySolve):
        structure = _triangle_mask(structure, op.lower)
    elif op.assume_a in {"sym", "her", "pos"}:
        triangle = _triangle_mask(structure, op.lower)
        structure = triangle | triangle.swapaxes(-1, -2)
    elif op.assume_a == "tridiagonal":
        index = pt.arange(matrix.shape[-1])
        structure &= pt.abs(index[:, None] - index[None, :]) <= 1
    inverse = _reachability(structure)
    if isinstance(op, CholeskySolve):
        transpose = inverse.swapaxes(-1, -2)
        inverse = (
            pt.matmul(transpose, inverse) if op.lower else pt.matmul(inverse, transpose)
        )
    rhs_mask = may_be_nonzero(rhs)
    if op.b_ndim == 1:
        return pt.matmul(inverse, rhs_mask[..., None])[..., 0]
    return pt.matmul(inverse, rhs_mask)


@node_rewriter([MayBeNonzero])
def local_nonzero_constants(fgraph, node):
    [value] = node.inputs
    if isinstance(value, Constant):
        data = value.data.toarray() if scipy.sparse.issparse(value.data) else value.data
        return [pt.constant(np.asarray(data) != 0)]


nonzero_lift_db.register("local_nonzero_constants", local_nonzero_constants, "basic")
nonzero_rewrites_db.register(
    "unbatch", in2out(local_useless_unbatched_blockwise), "basic", position=-1
)
nonzero_rewrites_db.register("shape", ShapeOptimizer(), "basic", position=0)
nonzero_rewrites_db.register(
    "pre-canonicalize", optdb.query("+canonicalize"), "basic", position=1
)
nonzero_rewrites_db.register("lift", nonzero_lift_db, "basic", position=2)
nonzero_rewrites_db.register(
    "post-canonicalize", optdb.query("+canonicalize"), "basic", position=4
)


def infer_nonzero(vjp, cotangents, *, wrt=(), query=nonzero_basic_query):
    """Infer structural nonzeros from an ordinary VJP on a private graph clone.

    ``cotangents`` identifies the symbolic direction inputs: their nonzero tests
    remain in the result so callers can supply basis directions. Other runtime
    inputs are unknown primal values; a connection cannot be excluded just
    because a coefficient happens to vanish at one particular input value.

    Lifting leaves root queries unresolved. A final resolution pass turns queries
    on cotangent roots into ordinary ``value != 0`` expressions, and all other
    unresolved queries into all-true masks.

    Requested ``wrt`` values are opaque during inference, including values whose
    producers normally force some elements to zero. They are directly
    perturbable when differentiating with respect to them.

    All canonicalization, lifting, and cleanup operate on cloned nodes and
    finish before the original primal bindings are restored. No rewrites run
    after restoration, and no inference annotations remain in the result.
    """
    single = isinstance(vjp, Variable)
    vjp = [vjp] if single else list(vjp)
    cotangents = [cotangents] if isinstance(cotangents, Variable) else list(cotangents)
    wrt = [wrt] if isinstance(wrt, Variable) else list(wrt)
    protected = {variable: variable.type() for variable in [*wrt, *cotangents]}
    cotangent_roots = {protected[variable] for variable in cotangents}
    cloned = clone_replace(vjp, replace=protected)
    fgraph = FunctionGraph(
        outputs=[may_be_nonzero(gradient) for gradient in cloned], clone=False
    )

    @node_rewriter([MayBeNonzero])
    def resolve_nonzero_queries(fgraph, node):
        [value] = node.inputs
        if value in cotangent_roots:
            return [pt.neq(value, 0)]
        return [pt.ones(value.shape, dtype="bool")]

    query = query.register((in2out(resolve_nonzero_queries, ignore_newtrees=False), 3))
    nonzero_rewrites_db.query(query).rewrite(fgraph)
    assert not any(isinstance(node.op, MayBeNonzero) for node in fgraph.toposort())
    # Bind only after the complete rewrite pipeline; never rewrite these nodes.
    fgraph.replace_all(
        ((placeholder, original) for original, placeholder in protected.items()),
        import_missing=True,
    )
    return fgraph.outputs[0] if single else fgraph.outputs
