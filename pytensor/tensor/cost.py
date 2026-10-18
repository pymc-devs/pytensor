from collections.abc import Sequence
from functools import reduce, singledispatch
from operator import add
from weakref import WeakKeyDictionary

import numpy as np

from pytensor.compile.ops import TypeCastingOp
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.cost import (
    FLOPS_PER_MULTIPLY_ADD,
    Bound,
    Cost,
    CostExpr,
    DimEntry,
    op_cost,
)
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.traversal import walk_toposort
from pytensor.graph.type import HasShape
from pytensor.scalar.basic import Add, Composite, Mul, ScalarOp
from pytensor.tensor.basic import (
    MakeVector,
    ScalarFromTensor,
    TensorFromScalar,
    as_tensor_variable,
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.shape import Shape, Shape_i, SpecifyShape
from pytensor.tensor.subtensor import Subtensor
from pytensor.tensor.type import TensorType
from pytensor.tensor.variable import TensorVariable


class ShapeSymbols:
    """Canonical shape symbols, shared by every cost read through one instance.

    A dimension is a static int when its value is a constant. Otherwise it is a symbol,
    shared by every structurally equal expression this instance has seen. Costs are
    comparable only when their symbols come from the same instance: the prover treats
    symbols from anywhere else as unrelated, so it reports "not provable". The instance
    keeps every expression it has seen alive, so it should live no longer than the
    comparisons it serves.
    """

    def __init__(self) -> None:
        self._id_of: dict[Variable, int] = {}
        self._ids: dict[object, int] = {}
        self._symbol_of: dict[int, Variable] = {}
        self._dim_of: dict[Variable, DimEntry] = {}

    @property
    def symbols(self) -> frozenset[Variable]:
        """Every symbol minted so far, the ``zero_able`` set when any dim may be 0."""
        return frozenset(self._symbol_of.values())

    def dim(self, expr: Variable) -> DimEntry:
        if expr in self._dim_of:
            return self._dim_of[expr]
        try:
            dim: DimEntry = int(get_underlying_scalar_constant_value(expr))
        except NotScalarConstantError:
            dim = self._symbol_of.setdefault(self._canonical_id(expr), expr)
        self._dim_of[expr] = dim
        return dim

    def _canonical_id(self, var: Variable) -> int:
        """An id shared by structurally equal expressions, found without recursion."""

        def deps(v: Variable) -> tuple:
            if v in self._id_of or v.owner is None:
                return ()
            return tuple(v.owner.inputs)

        for v in walk_toposort([var], deps):
            if v in self._id_of:
                continue
            if isinstance(v, Constant):
                key: object = ("constant", v.signature())
            elif v.owner is None:
                key = ("leaf", v)
            else:
                node = v.owner
                key = (
                    node.op,
                    tuple(self._id_of[inp] for inp in node.inputs),
                    node.outputs.index(v),
                )
            self._id_of[v] = self._ids.setdefault(key, len(self._ids))
        return self._id_of[var]


def dims_of(
    var: Variable, shape_feature: ShapeFeature, symbols: ShapeSymbols
) -> tuple[DimEntry, ...]:
    """The dimensions of `var`: static ints, and canonical symbols for the rest."""
    static_shape = var.type.shape if isinstance(var.type, HasShape) else ()
    return tuple(
        static if static is not None else symbols.dim(shape_feature.get_shape(var, i))
        for i, static in enumerate(static_shape)
    )


def node_cost(node: Apply, shape_feature: ShapeFeature, symbols: ShapeSymbols) -> Cost:
    """Cost of `node`, with dimensions read from `shape_feature`."""
    return op_cost(
        node.op,
        node,
        [dims_of(inp, shape_feature, symbols) for inp in node.inputs],
        [dims_of(out, shape_feature, symbols) for out in node.outputs],
    )


def graph_cost(
    graph: FunctionGraph | Sequence[Variable], symbols: ShapeSymbols | None = None
) -> Cost:
    """Cost of `graph`, the sum over its Apply nodes with each node counted once.

    Parameters
    ----------
    graph : FunctionGraph or sequence of Variable
        The graph, or its outputs. Outputs get a new FunctionGraph and shape inference
        on every call. A FunctionGraph reuses the shapes its ShapeFeature holds, or gets
        a ShapeFeature for the duration of the call.
    symbols : ShapeSymbols, optional
        The symbol table for symbolic dimensions. Pass the same instance to every call
        whose results are compared. Defaults to a fresh table.
    """
    fgraph = (
        graph
        if isinstance(graph, FunctionGraph)
        else FunctionGraph(outputs=list(graph), clone=False)
    )
    if symbols is None:
        symbols = ShapeSymbols()
    shape_feature = getattr(fgraph, "shape_feature", None)
    attached = shape_feature is None
    if shape_feature is None:
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
    try:
        return Cost.sum(
            node_cost(node, shape_feature, symbols) for node in fgraph.apply_nodes
        )
    finally:
        if attached:
            fgraph.remove_feature(shape_feature)


def to_graph(expr: CostExpr) -> TensorVariable:
    """A scalar ``float64`` expression equal to `expr`, over its shape symbols.

    A cost too large for ``int64`` loses precision instead of wrapping around.
    Evaluating the expression computes whatever its symbols depend on, so a dimension
    that shape inference can only read off an Op's output runs that Op.
    """
    terms = []
    for monomial, coef in expr.monomials.items():
        term = as_tensor_variable(np.float64(coef))
        for symbol, exponent in monomial:
            term = term * as_tensor_variable(symbol).astype("float64") ** exponent
        terms.append(term)
    if not terms:
        return as_tensor_variable(np.float64(0))
    return reduce(add, terms)


def _does_float_work(node: Apply) -> bool:
    """Whether `node` reads or writes a float or complex value, and so costs FLOPs."""
    return any(
        np.dtype(var.type.dtype).kind in "fc"
        for var in (*node.inputs, *node.outputs)
        if hasattr(var.type, "dtype")
    )


def _bytes_read_and_written(
    node: Apply,
    input_dims: Sequence[tuple[DimEntry, ...]],
    output_dims: Sequence[tuple[DimEntry, ...]],
) -> CostExpr:
    """Each input position counts once, so an input passed twice is read twice."""
    return CostExpr.sum(
        CostExpr.from_dim_product([np.dtype(var.type.dtype).itemsize, *dims])
        for var, dims in zip(
            (*node.inputs, *node.outputs), (*input_dims, *output_dims), strict=True
        )
        if hasattr(var.type, "dtype")
    )


@singledispatch
def scalar_flops(scalar_op: ScalarOp, n_inputs: int) -> int:
    """FLOPs per element of `scalar_op` applied to `n_inputs` inputs.

    One per operation unless a rule says otherwise. The weights are a convention, not a
    measurement of any backend. Callers charge them only for operations that read or
    write a float or complex value.
    """
    return 1


@scalar_flops.register(Add)
@scalar_flops.register(Mul)
def scalar_flops_variadic(scalar_op: ScalarOp, n_inputs: int) -> int:
    return max(1, n_inputs - 1)


# A Composite's inner graph never changes, so its count is computed once. A
# scalar_flops rule registered after a Composite was first costed does not reach it.
_COMPOSITE_FLOPS: WeakKeyDictionary[Composite, int] = WeakKeyDictionary()


@scalar_flops.register(Composite)
def scalar_flops_Composite(scalar_op: Composite, n_inputs: int) -> int:
    flops = _COMPOSITE_FLOPS.get(scalar_op)
    if flops is None:
        flops = _COMPOSITE_FLOPS[scalar_op] = sum(
            scalar_flops(node.op, len(node.inputs))
            for node in scalar_op.fgraph.apply_nodes
            if _does_float_work(node)
        )
    return flops


@op_cost.register(Elemwise)
def op_cost_Elemwise(op: Elemwise, node: Apply, input_dims, output_dims) -> Cost:
    flops = CostExpr.zero()
    if _does_float_work(node):
        flops = CostExpr.from_dim_product(output_dims[0]) * scalar_flops(
            op.scalar_op, len(node.inputs)
        )
    return Cost(
        Bound.exact(flops),
        Bound.exact(_bytes_read_and_written(node, input_dims, output_dims)),
    )


@op_cost.register(Dot)
def op_cost_Dot(op: Dot, node: Apply, input_dims, output_dims) -> Cost:
    """Two FLOPs per multiply-add: ``2 * m * n * p`` for ``(m, p) @ (p, n)``."""
    flops = CostExpr.zero()
    if _does_float_work(node):
        flops = CostExpr.from_dim_product(
            [FLOPS_PER_MULTIPLY_ADD, *output_dims[0], input_dims[0][-1]]
        )
    return Cost(
        Bound.exact(flops),
        Bound.exact(_bytes_read_and_written(node, input_dims, output_dims)),
    )


_CORE_NODES: WeakKeyDictionary[Blockwise, dict[tuple[TensorType, ...], Apply]] = (
    WeakKeyDictionary()
)


def _core_node(op: Blockwise, input_types: tuple[TensorType, ...]) -> Apply:
    """A core node of `op` for inputs of `input_types`, built once per pair."""
    nodes = _CORE_NODES.setdefault(op, {})
    if input_types not in nodes:
        nodes[input_types] = op._create_dummy_core_node([t() for t in input_types])
    return nodes[input_types]


@op_cost.register(Blockwise)
def op_cost_Blockwise(op: Blockwise, node: Apply, input_dims, output_dims) -> Cost:
    """The core Op's FLOPs once per batch element; bytes of the batched values."""
    bytes_ = Bound.exact(_bytes_read_and_written(node, input_dims, output_dims))
    if op_cost.dispatch(type(op.core_op)) is op_cost.dispatch(object):
        return Cost(Bound.unknown(), bytes_)

    core_node = _core_node(op, tuple(inp.type for inp in node.inputs))
    core_cost = op_cost(
        core_node.op,
        core_node,
        [
            dims[len(dims) - len(sig) :]
            for dims, sig in zip(input_dims, op.inputs_sig, strict=True)
        ],
        [
            dims[len(dims) - len(sig) :]
            for dims, sig in zip(output_dims, op.outputs_sig, strict=True)
        ],
    )
    out_dims = output_dims[0]
    batch = CostExpr.from_dim_product(
        out_dims[: len(out_dims) - len(op.outputs_sig[0])]
    )
    return Cost(core_cost.flops * batch, bytes_)


@op_cost.register(DimShuffle)
@op_cost.register(Subtensor)
@op_cost.register(TypeCastingOp)
@op_cost.register(SpecifyShape)
@op_cost.register(Shape)
@op_cost.register(Shape_i)
@op_cost.register(MakeVector)
@op_cost.register(ScalarFromTensor)
@op_cost.register(TensorFromScalar)
def op_cost_free(op: Op, node: Apply, input_dims, output_dims) -> Cost:
    """Views, markers, and shape bookkeeping touch no tensor data."""
    return Cost.zero()
