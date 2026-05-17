from collections.abc import Sequence

from pytensor.assumptions.core import ALL_KEYS, FactState, register_assumption
from pytensor.compile.ops import TypeCastingOp
from pytensor.graph.basic import Apply, Variable
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import as_tensor_variable


class SpecifyAssumptions(TypeCastingOp):
    """No-op that declares structural assumptions on a tensor for use by graph rewrites.

    ``assumptions`` is a tuple of ``(name, FactState)`` pairs sorted by ``name``, where
    ``name`` matches the name of an :class:`AssumptionKey`. Two instances with the same
    fact set compare equal via ``__props__``, so PyTensor's graph merge collapses
    duplicates.
    """

    __props__ = ("assumptions",)

    assumptions: tuple[tuple[str, FactState], ...]

    def __init__(self, assumptions: dict[str, FactState]):
        super().__init__()
        self.assumptions = tuple(
            (name, FactState(state)) for name, state in sorted(assumptions.items())
        )

    def __str__(self):
        facts = ", ".join(
            name if state is FactState.TRUE else f"!{name}"
            for name, state in self.assumptions
        )
        return f"{type(self).__name__}{{{facts}}}"

    def make_node(self, x):
        if not isinstance(x, Variable):
            x = as_tensor_variable(x)
        out = x.type()
        return Apply(self, [x], [out])

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes

    def pullback(
        self, inputs, outputs, output_cotangents: Sequence[Variable]
    ) -> list[Variable]:
        return list(output_cotangents)


def specify_assumption_rule(key, op, feature, fgraph, node, input_states):
    """Report the declared state for ``key`` joined with whatever inference derived
    from the input. The join surfaces ``ConflictingAssumptionsError`` when the user
    asserts a state that contradicts what the system can prove (e.g. asserting
    ``diagonal=False`` on something proved diagonal)."""
    for name, state in op.assumptions:
        if name == key.name:
            return [FactState.join(state, input_states[0])]
    return [input_states[0]]


def assume(
    x: TensorLike,
    diagonal: bool | None = None,
    lower_triangular: bool | None = None,
    upper_triangular: bool | None = None,
    symmetric: bool | None = None,
    positive_definite: bool | None = None,
    orthogonal: bool | None = None,
):
    """Attach structural assumptions to a symbolic tensor.

    Returns a tensor identical to *x* at runtime but carrying the declared assumptions so that
    graph rewrites can exploit them. Each keyword may be ``True`` (assert the property holds),
    ``False`` (assert it does not hold), or ``None`` (no assertion).

    Parameters
    ----------
    x : tensor-like
        The input to annotate.
    diagonal : bool, optional
        Assert that *x* is (or is not) a diagonal matrix.
    lower_triangular : bool, optional
        Assert that *x* is (or is not) lower-triangular.
    upper_triangular : bool, optional
        Assert that *x* is (or is not) upper-triangular.
    symmetric : bool, optional
        Assert that *x* is (or is not) symmetric.
    positive_definite : bool, optional
        Assert that *x* is (or is not) positive-definite.
    orthogonal : bool, optional
        Assert that *x* is (or is not) orthogonal.

    Returns
    -------
    out : TensorVariable
        A view of *x* with the assumptions attached.

    Examples
    --------
    >>> import pytensor.tensor as pt
    >>> x = pt.dmatrix("x")
    >>> x_diag = assume(x, diagonal=True)
    >>> x_not_sym = assume(x, symmetric=False)
    """
    if not isinstance(x, Variable):
        x = as_tensor_variable(x)

    values = {
        "diagonal": diagonal,
        "lower_triangular": lower_triangular,
        "upper_triangular": upper_triangular,
        "symmetric": symmetric,
        "positive_definite": positive_definite,
        "orthogonal": orthogonal,
    }
    assumptions = {
        name: FactState.TRUE if value else FactState.FALSE
        for name, value in values.items()
        if value is not None
    }

    if not assumptions:
        return x

    return SpecifyAssumptions(assumptions)(x)


for _key in ALL_KEYS:
    register_assumption(_key, SpecifyAssumptions)(specify_assumption_rule)
