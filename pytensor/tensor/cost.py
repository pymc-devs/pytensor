from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.cost import Cost, DimEntry, op_cost
from pytensor.graph.traversal import walk_toposort
from pytensor.graph.type import HasShape
from pytensor.tensor.basic import (
    get_underlying_scalar_constant_value,
)
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.rewriting.shape import ShapeFeature


class ShapeSymbols:
    """Canonical shape symbols for one costing pass.

    A dimension is a static int when its value is a constant. Otherwise it is a symbol,
    shared by every structurally equal expression seen in the same pass. The instance
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
