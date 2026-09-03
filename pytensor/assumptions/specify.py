from collections.abc import Sequence

from pytensor.assumptions.core import (
    KEY_REGISTRY,
    AssumptionKey,
    FactState,
    register_universal_assumption,
)
from pytensor.compile.ops import TypeCastingOp
from pytensor.graph.basic import Apply, Variable
from pytensor.tensor import TensorLike
from pytensor.tensor.basic import as_tensor_variable


class SpecifyAssumptions(TypeCastingOp):
    """No-op that declares structural assumptions on a tensor for use by graph rewrites.

    ``assumptions`` is a tuple of ``(AssumptionKey, FactState)`` pairs sorted by key
    name. Declaring a fact therefore requires holding the key itself, and constructing
    a key registers it, so a graph cannot carry an assumption the system has never
    heard of. Two instances with the same fact set compare equal via ``__props__``, so
    PyTensor's graph merge collapses duplicates.

    Parameters
    ----------
    assumptions : dict mapping AssumptionKey to FactState
        The facts to declare.
    """

    __props__ = ("assumptions",)

    assumptions: tuple[tuple[AssumptionKey, FactState], ...]

    def __init__(self, assumptions: dict[AssumptionKey, FactState]):
        super().__init__()
        passed_by_name = [
            key for key in assumptions if not isinstance(key, AssumptionKey)
        ]
        if passed_by_name:
            raise TypeError(
                f"SpecifyAssumptions is keyed by AssumptionKey, not by name: "
                f"{passed_by_name!r}. Pass the key objects, or declare by name "
                f"with assume()."
            )
        self.assumptions = tuple(
            (key, FactState(state))
            for key, state in sorted(assumptions.items(), key=lambda kv: kv[0].name)
        )

    def __str__(self):
        facts = ", ".join(
            key.name if state is FactState.TRUE else f"!{key.name}"
            for key, state in self.assumptions
        )
        return f"{type(self).__name__}{{{facts}}}"

    def make_node(self, x):
        if not isinstance(x, Variable):
            x = as_tensor_variable(x)
        out = x.type()
        return Apply(self, [x], [out])

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def pullback(
        self, inputs, outputs, output_cotangents: Sequence[Variable]
    ) -> list[Variable]:
        return list(output_cotangents)


@register_universal_assumption(SpecifyAssumptions)
def specify_assumption_rule(key, op, feature, fgraph, node, input_states):
    """Report the declared state for ``key`` joined with whatever inference derived
    from the input. The join surfaces ``ConflictingAssumptionsError`` when the user
    asserts a state that contradicts what the system can prove (e.g. asserting
    ``diagonal=False`` on something proved diagonal)."""
    for declared_key, state in op.assumptions:
        if declared_key == key:
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
    selection: bool | None = None,
    permutation: bool | None = None,
    unique_indices: bool | None = None,
    **assumptions: bool | None,
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
    selection : bool, optional
        Assert that *x* is (or is not) a selection matrix
    permutation : bool, optional
        Assert that *x* is (or is not) a permutation matrix
    unique_indices : bool, optional
        Assert that *x*'s entries address distinct positions when used as an
        index (or that they do not): no value repeats, and no negative entry
        aliases a non-negative one (e.g. ``-1`` and ``n-1``). Such an index can
        never enlarge the axis it indexes, so it can be lifted earlier through
        operations without risk of duplicating computation.
    **assumptions : bool, optional
        Assumptions registered by downstream libraries, passed by key name, e.g.
        ``time_varying=True``.

    Returns
    -------
    out : TensorVariable
        A view of *x* with the assumptions attached.

    Examples
    --------
    .. code-block:: python

        import pytensor.tensor as pt

        x = pt.dmatrix("x")
        x_diag = assume(x, diagonal=True)
        x_not_sym = assume(x, symmetric=False)
    """
    if not isinstance(x, Variable):
        x = as_tensor_variable(x)

    core_values = {
        "diagonal": diagonal,
        "lower_triangular": lower_triangular,
        "upper_triangular": upper_triangular,
        "symmetric": symmetric,
        "positive_definite": positive_definite,
        "orthogonal": orthogonal,
        "selection": selection,
        "permutation": permutation,
        "unique_indices": unique_indices,
    }

    unknown = [name for name in assumptions if name not in KEY_REGISTRY]
    if unknown:
        extensions = sorted(KEY_REGISTRY.keys() - core_values.keys())
        raise ValueError(
            f"Unknown assumption(s): {', '.join(unknown)}. Registered extension "
            f"assumptions are: {', '.join(extensions) if extensions else '(none)'}. "
            f"Register a new one by constructing an AssumptionKey."
        )

    declared = {
        KEY_REGISTRY[name]: FactState.TRUE if value else FactState.FALSE
        for name, value in (core_values | assumptions).items()
        if value is not None
    }

    if not declared:
        return x

    return SpecifyAssumptions(declared)(x)
