from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Any

from pytensor.graph import Apply, FunctionGraph, Op
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.traversal import walk_toposort
from pytensor.tensor.variable import TensorConstant


class FactState(IntFlag):
    """Three-valued logic for assumption inference.

    The three fact states are TRUE, FALSE and UNKNOWN.

    UNKNOWN is the default condition, in which we cannot confirm or deny the fact.  TRUE and FALSE are definitive
    states.  If we have evidence that a fact is both TRUE and FALSE, we get CONFLICT. In general, a CONFLICT state
    should not be possible.
    """

    UNKNOWN = 0
    TRUE = auto()
    FALSE = auto()
    CONFLICT = TRUE | FALSE

    def __bool__(self) -> bool:
        raise TypeError(
            "FactState has no boolean value; compare against FactState.TRUE, "
            "FactState.UNKNOWN, etc. explicitly, or use AssumptionFeature.check()."
        )

    @classmethod
    def join(cls, left: "FactState", right: "FactState") -> "FactState":
        """Combine two pieces of evidence about the *same* (variable, key)."""
        return cls(left | right)


@dataclass(frozen=True)
class AssumptionKey:
    """Identifies a named structural property (e.g. "diagonal" or "triangular").

    ``short_name`` is an abbreviated label used by ``debugprint(print_assumptions=True)``;
    it falls back to ``name`` when empty.
    """

    name: str
    short_name: str = ""

    def __repr__(self) -> str:
        return self.name


class ConflictingAssumptionsError(ValueError):
    """Raised when joining evidence about a (variable, key) produces ``FactState.CONFLICT``.

    Surfaces contradictions between user-asserted facts and inferred facts (or between
    multiple sources of inferred facts) before they can silently corrupt downstream rewrites.
    """


# An inference function takes the active AssumptionKey, the Op, the current AssumptionFeature, the FunctionGraph,
# the Apply node being analyzed, and the states of the input variables for the current key. It returns a list of
# FactState with one entry per node output.
InferFactFn = Callable[
    [AssumptionKey, Op, "AssumptionFeature", FunctionGraph, Apply, list[FactState]],
    list[FactState],
]

# The global inference registry maps (AssumptionKey, Op type) pairs to lists of inference functions.
# Rules are tried in registration order; the first to return TRUE wins.
ASSUMPTION_INFER_REGISTRY: dict[tuple[AssumptionKey, type], list[InferFactFn]] = {}

# Registry mapping assumptions to other assumptions they imply.  For example, a "diagonal" matrix is also "symmetric"
# and "triangular".  This is consulted after all other inference rules to derive additional facts.
IMPLIES: dict[AssumptionKey, list[AssumptionKey]] = {}


def register_implies(stronger: AssumptionKey, *weaker: AssumptionKey) -> None:
    """Declare that *stronger* being TRUE implies each *weaker* key is also TRUE."""
    IMPLIES.setdefault(stronger, []).extend(weaker)


# Maps a key to a function that infers it directly from a literal TensorConstant's data.
# AssumptionFeature consults this for variables with no owner, letting a property be
# derived from raw values (e.g. recognising a constant diagonal matrix).
ConstantInferFn = Callable[[TensorConstant], FactState]
CONSTANT_INFER_REGISTRY: dict[AssumptionKey, ConstantInferFn] = {}


def register_constant_inference(key: AssumptionKey, fn: ConstantInferFn) -> None:
    """Register *fn* as the literal-constant inference for *key*."""
    CONSTANT_INFER_REGISTRY[key] = fn


# The canonical structural-property keys
DIAGONAL = AssumptionKey("diagonal", short_name="diag")
LOWER_TRIANGULAR = AssumptionKey("lower_triangular", short_name="tril")
UPPER_TRIANGULAR = AssumptionKey("upper_triangular", short_name="triu")
SYMMETRIC = AssumptionKey("symmetric", short_name="sym")
POSITIVE_DEFINITE = AssumptionKey("positive_definite", short_name="pd")
ORTHOGONAL = AssumptionKey("orthogonal", short_name="orth")
SELECTION = AssumptionKey("selection", short_name="sel")
PERMUTATION = AssumptionKey("permutation", short_name="perm")

ALL_KEYS = (
    DIAGONAL,
    LOWER_TRIANGULAR,
    UPPER_TRIANGULAR,
    SYMMETRIC,
    POSITIVE_DEFINITE,
    ORTHOGONAL,
    SELECTION,
    PERMUTATION,
)

# Implications about structural properties derivably from other structural properties
register_implies(DIAGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR, SYMMETRIC)
register_implies(POSITIVE_DEFINITE, SYMMETRIC)
register_implies(PERMUTATION, SELECTION, ORTHOGONAL)


def register_assumption(
    key: AssumptionKey, *op_types: type
) -> Callable[[InferFactFn], InferFactFn]:
    """Decorator that registers an inference rule for ``(key, op_type)`` pairs.

    The decorated function is called as ``fn(key, op, feature, fgraph, node, input_states)``
    and must return a list of :class:`FactState` with one entry per node output.
    """

    def decorator(fn: InferFactFn) -> InferFactFn:
        for op_type in op_types:
            ASSUMPTION_INFER_REGISTRY.setdefault((key, op_type), []).append(fn)
        return fn

    return decorator


def infer_assumption_for_node(
    key: AssumptionKey,
    op: Op,
    feature: "AssumptionFeature",
    fgraph: FunctionGraph,
    node: Apply,
    input_states: list[FactState],
) -> list[FactState]:
    """Determine the *key* fact for every output of *node*.

    Walks the Op's MRO for the most specific :func:`register_assumption` rule and
    returns the first non-UNKNOWN result. Falls back to ``UNKNOWN`` per output.
    """
    rules: list[InferFactFn] = []
    for cls in type(op).__mro__:
        fns = ASSUMPTION_INFER_REGISTRY.get((key, cls))
        if fns is not None:
            rules = fns
            break

    n_outputs = len(node.outputs)
    for fn in rules:
        output_states = fn(key, op, feature, fgraph, node, input_states)
        if len(output_states) != n_outputs:
            raise ValueError(
                f"an assumption rule returned {len(output_states)} states for "
                f"{n_outputs} outputs on node {node!r}"
            )
        if any(s is not FactState.UNKNOWN for s in output_states):
            return output_states

    return [FactState.UNKNOWN] * n_outputs


class AssumptionFeature(Feature):
    """``FunctionGraph`` feature that tracks symbolic assumptions about variables.

    Assumptions (e.g. "this matrix is diagonal") are represented as ``(variable,
    AssumptionKey) -> FactState`` mappings. Facts are inferred lazily via per-Op
    rules registered with :func:`register_assumption`.

    Attached lazily by :func:`check_assumption` on first use.
    Results are cached and automatically refreshed when the graph changes.
    """

    __slots__ = ("_stale_vars", "_var_to_keys", "cache", "fgraph")

    def on_attach(self, fgraph: Any) -> None:
        if hasattr(fgraph, "assumption_feature"):
            raise AlreadyThere("AssumptionFeature is already attached")
        self.fgraph = fgraph
        self.cache: dict[tuple[Any, AssumptionKey], FactState] = {}
        self._var_to_keys: dict[Any, set[AssumptionKey]] = {}
        self._stale_vars: set = set()
        fgraph.assumption_feature = self

    def on_detach(self, fgraph: Any) -> None:
        self.cache = {}
        self._var_to_keys = {}
        self._stale_vars = set()
        self.fgraph = None
        del fgraph.assumption_feature

    def on_change_input(self, fgraph, node, i, old_var, new_var, reason=None) -> None:
        # Carry non-UNKNOWN facts from old_var to new_var, then defer downstream
        # UNKNOWN invalidation to the next get()/check() call.
        # CONFLICT is cached, not raised: the graph mutation has already landed,
        # so raising here corrupts the caller's view.
        for key in self._var_to_keys.get(old_var, ()):
            old_state = self.cache.get((old_var, key))
            if old_state is None or old_state is FactState.UNKNOWN:
                continue
            new_state = self.cache.get((new_var, key))
            if new_state is None or new_state is FactState.UNKNOWN:
                self.cache[(new_var, key)] = old_state
            else:
                self.cache[(new_var, key)] = FactState.join(old_state, new_state)
            self._var_to_keys.setdefault(new_var, set()).add(key)
            # new_var gained a fact whose implications were cached UNKNOWN before it was
            # known; flag it so the next get() sweep re-derives those stale entries.
            self._stale_vars.add(new_var)

        self._stale_vars.update(node.outputs)

    def on_prune(self, fgraph, node, reason=None) -> None:
        self._stale_vars.difference_update(node.outputs)

    def clone(self) -> "AssumptionFeature":
        # Cache entries are keyed on variable identity, so the cloned feature starts empty
        # and re-derives facts from the cloned graph (assumptions attached via ``assume()``
        # follow their variables through the clone since they live as graph nodes).
        return AssumptionFeature()

    def get(self, var: Any, key: AssumptionKey) -> FactState:
        """Return the inferred :class:`FactState` for ``(var, key)``.

        Raise :class:`ConflictingAssumptionsError` when the cached state is
        :attr:`FactState.CONFLICT` — surfaces conflicts from substitutions or
        owner-inferred rules at the point of query, where the graph is stable
        and the caller can react.
        """
        cache_key = (var, key)
        state = self.cache.get(cache_key)
        if state is None or state is FactState.UNKNOWN:
            if self._stale_vars:
                self._drop_unknown_downstream(self._stale_vars)
                self._stale_vars.clear()
                # Re-check: the UNKNOWN will still be valid
                # if the stale vars were not ancestors of var.
                state = self.cache.get(cache_key)
            if state is None:
                state = self._compute(var, key)
        if state is FactState.CONFLICT:
            raise ConflictingAssumptionsError(
                f"Conflicting evidence for key {key!r} on {var!r}."
            )
        return state

    def check(self, var: Any, key: AssumptionKey) -> bool:
        """Return ``True`` iff the assumption is definitively TRUE for ``var``."""
        return self.get(var, key) is FactState.TRUE

    def _compute(self, var: Any, key: AssumptionKey) -> FactState:
        """Infer the fact state for ``(var, key)``, caching ancestors inputs-first.

        After owner-based inference, walks the ``IMPLIES`` graph in both directions:
          - Forward: a stronger key being TRUE makes this (weaker) key TRUE.
          - Contrapositive: a weaker key being FALSE makes this (stronger) key FALSE.

        Raise :class:`ConflictingAssumptionsError` if owner-inferred rules produce
        ``FactState.CONFLICT``.
        """

        def deps(v: Any) -> tuple:
            # Stop descending at cached variables — their facts (and their ancestors')
            # are already settled for this key.
            if (v, key) in self.cache:
                return ()
            owner = getattr(v, "owner", None)
            return tuple(owner.inputs) if owner is not None else ()

        for v in walk_toposort([var], deps):
            if (v, key) in self.cache:
                continue

            # Cache UNKNOWN up front so any recursive ``self.get(v, other_key)`` triggered
            # by the IMPLIES checks sees a placeholder instead of recursing back here.
            self.cache[(v, key)] = FactState.UNKNOWN
            self._var_to_keys.setdefault(v, set()).add(key)

            owner = getattr(v, "owner", None)
            if owner is not None:
                input_states = [self.get(inp, key) for inp in owner.inputs]
                output_states = infer_assumption_for_node(
                    key, owner.op, self, self.fgraph, owner, input_states
                )
                state = output_states[owner.outputs.index(v)]
            elif isinstance(v, TensorConstant) and key in CONSTANT_INFER_REGISTRY:
                state = CONSTANT_INFER_REGISTRY[key](v)
            else:
                state = FactState.UNKNOWN

            if state is FactState.UNKNOWN:
                for stronger, weaker_list in IMPLIES.items():
                    if key in weaker_list and self.get(v, stronger) is FactState.TRUE:
                        state = FactState.TRUE
                        break

            if state is FactState.UNKNOWN:
                for weaker in IMPLIES.get(key, ()):
                    if self.get(v, weaker) is FactState.FALSE:
                        state = FactState.FALSE
                        break

            if state is FactState.CONFLICT:
                raise ConflictingAssumptionsError(
                    f"Conflicting evidence for {key} on {v!r} from owner-inferred rules."
                )

            self.cache[(v, key)] = state

        return self.cache[(var, key)]

    def _drop_unknown_downstream(self, start_vars) -> None:
        """Drop UNKNOWN entries for *start_vars* and downstream so they re-probe."""
        cache = self.cache
        var_to_keys = self._var_to_keys
        clients = self.fgraph.clients
        queue = deque(start_vars)
        seen = set(start_vars)
        while queue:
            var = queue.popleft()
            keys = var_to_keys.get(var)
            if keys:
                for key in [
                    k for k in keys if cache.get((var, k)) is FactState.UNKNOWN
                ]:
                    cache.pop((var, key), None)
                    keys.discard(key)
                if not keys:
                    var_to_keys.pop(var, None)
            for client_node, _ in clients.get(var, ()):
                for out in client_node.outputs:
                    if out not in seen:
                        seen.add(out)
                        queue.append(out)


def true_if(cond: bool, else_false: bool = False) -> list[FactState]:
    """``[TRUE]`` when *cond* holds, ``[UNKNOWN]`` (or ``[FALSE]``) otherwise."""
    if cond:
        return [FactState.TRUE]
    return [FactState.FALSE] if else_false else [FactState.UNKNOWN]


def propagate_first(key, op, feature, fgraph, node, input_states) -> list[FactState]:
    """Output inherits the assumption iff the first input has it."""
    return true_if(input_states[0] is FactState.TRUE)


def all_inputs_have_key(
    key, op, feature, fgraph, node, input_states
) -> list[FactState]:
    """Output inherits the assumption iff *every* input has it."""
    return true_if(all(s is FactState.TRUE for s in input_states))


def check_assumption(
    fgraph: FunctionGraph | None, var: Any, key: AssumptionKey
) -> bool:
    """Return True iff *key* is definitively TRUE for *var* in *fgraph*.

    Lazily attaches :class:`AssumptionFeature` to *fgraph* if it is not already present.
    """
    if fgraph is None:
        return False
    feature = getattr(fgraph, "assumption_feature", None)
    if feature is None:
        feature = AssumptionFeature()
        fgraph.attach_feature(feature)
    return feature.check(var, key)
