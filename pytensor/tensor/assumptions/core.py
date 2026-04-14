from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Any

from pytensor.graph import Apply, FunctionGraph, Op
from pytensor.graph.features import AlreadyThere, Feature


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
        return self is FactState.TRUE

    @classmethod
    def join(cls, left: "FactState", right: "FactState") -> "FactState":
        """Combine two pieces of evidence about the *same* (variable, key)."""
        return cls(left | right)


@dataclass(frozen=True)
class AssumptionKey:
    """Identifies a named structural property (e.g. "diagonal" or "triangular")."""

    name: str

    def __repr__(self) -> str:
        return self.name


# An inference function takes an Op, the current AssumptionFeature, the FunctionGraph, the Apply node being analyzed,
# the states of the input variables for the current key, and returns a list of FactState (one per output).
InferFactFn = Callable[
    [Op, "AssumptionFeature", FunctionGraph, Apply, list[FactState]],
    list[FactState],
]

# The global inference registry maps (AssumptionKey, Op type) pairs to inference functions.  The most specific
# applicable rule is used for each node.
ASSUMPTION_INFER_REGISTRY: dict[tuple[AssumptionKey, type], InferFactFn] = {}

# Registry mapping assumptions to other assumptions they imply.  For example, a "diagonal" matrix is also "symmetric"
# and "triangular".  This is consulted after all other inference rules to derive additional facts.
IMPLIES: dict[AssumptionKey, list[AssumptionKey]] = {}


def register_implies(stronger: AssumptionKey, *weaker: AssumptionKey) -> None:
    """Declare that *stronger* being TRUE implies each *weaker* key is also TRUE."""
    IMPLIES.setdefault(stronger, []).extend(weaker)


def register_assumption(
    key: AssumptionKey, *op_types: type
) -> Callable[[InferFactFn], InferFactFn]:
    """Decorator that registers an inference rule for ``(key, op_type)`` pairs.

    The decorated function is called as ``fn(op, feature, fgraph, node, input_states)``
    and must return a list of :class:`FactState` with one entry per node output.
    """

    def decorator(fn: InferFactFn) -> InferFactFn:
        for op_type in op_types:
            ASSUMPTION_INFER_REGISTRY[(key, op_type)] = fn
        return fn

    return decorator


def lookup_assumption_rule(key: AssumptionKey, op: Any) -> InferFactFn | None:
    """Find the most specific registered rule for *(key, type(op))*, walking the MRO."""
    for cls in type(op).__mro__:
        fn = ASSUMPTION_INFER_REGISTRY.get((key, cls))
        if fn is not None:
            return fn
    return None


def _default_infer_assumption(node: Any) -> list[FactState]:
    """Absent evidence, all facts are assumed to be UNKNOWN for all outputs of all Ops."""
    return [FactState.UNKNOWN] * len(node.outputs)


def _validate_output_states(
    node: Any, output_states: list[FactState]
) -> list[FactState]:
    if len(output_states) != len(node.outputs):
        raise ValueError(
            f"infer_assumption returned {len(output_states)} states for "
            f"{len(node.outputs)} outputs on node {node!r}"
        )
    return [FactState(s) for s in output_states]


def infer_assumption_for_node(
    op: Op,
    key: AssumptionKey,
    feature: "AssumptionFeature",
    fgraph: FunctionGraph,
    node: Apply,
    input_states: list[FactState],
) -> list[FactState]:
    """Determine the *key* fact for every output of *node*.

    Resolution order:
      1. ``op.infer_assumption(key, feature, fgraph, node, input_states)``
      2. Registered rule via :func:`register_assumption`
      3. Conservative ``UNKNOWN`` for every output.
    """
    meth = getattr(op, "infer_assumption", None)
    if meth is not None:
        output_states = meth(key, feature, fgraph, node, input_states)
        if output_states is not NotImplemented:
            return _validate_output_states(node, output_states)

    fn = lookup_assumption_rule(key, op)
    if fn is not None:
        output_states = fn(op, feature, fgraph, node, input_states)
        return _validate_output_states(node, output_states)

    return _default_infer_assumption(node)


class AssumptionFeature(Feature):
    """``FunctionGraph`` feature that tracks symbolic assumptions about variables.

    Assumptions (e.g. "this matrix is diagonal") are represented as ``(variable, AssumptionKey) -> FactState``
    mappings.  Facts are inferred lazily via per-Op rules registered with :func:`register_assumption` or via
    an ``infer_assumption`` method on the Op itself.

    Results are cached and automatically invalidated when the graph changes.
    """

    __slots__ = ("cache", "fgraph", "user_facts")

    def on_attach(self, fgraph: Any) -> None:
        if hasattr(fgraph, "assumption_feature"):
            raise AlreadyThere("AssumptionFeature is already attached")
        self.fgraph = fgraph
        self.cache: dict[tuple[Any, AssumptionKey], FactState] = {}
        self.user_facts: dict[tuple[Any, AssumptionKey], FactState] = {}
        fgraph.assumption_feature = self

    def on_detach(self, fgraph: Any) -> None:
        self.cache = {}
        self.user_facts = {}
        self.fgraph = None
        del fgraph.assumption_feature

    def on_import(self, fgraph, node, reason) -> None:
        self.invalidate_from_vars(node.outputs)

    def on_change_input(self, fgraph, node, i, old_var, new_var, reason=None) -> None:
        if node is not None:
            self.invalidate_from_vars(node.outputs)

    def on_prune(self, fgraph, node, reason) -> None:
        self.invalidate_from_vars(node.outputs)

    def clone(self) -> "AssumptionFeature":
        return AssumptionFeature()

    def get(self, var: Any, key: AssumptionKey) -> FactState:
        """Return the inferred :class:`FactState` for ``(var, key)``"""
        cache_key = (var, key)
        if cache_key not in self.cache:
            self.cache[cache_key] = self._compute(var, key)
        return self.cache[cache_key]

    def check(self, var: Any, key: AssumptionKey) -> bool:
        """Return ``True`` iff the assumption is definitively TRUE for ``var``."""
        return bool(self.get(var, key))

    def set_user_fact(self, var: Any, key: AssumptionKey, state: FactState) -> None:
        """Join *state* with any existing user evidence for ``(var, key)``."""
        state = FactState(state)
        cache_key = (var, key)
        old = self.user_facts.get(cache_key, FactState.UNKNOWN)
        new = FactState.join(old, state)
        if new != old:
            self.user_facts[cache_key] = new
            self.invalidate_from_vars([var])

    def replace_user_fact(self, var: Any, key: AssumptionKey, state: FactState) -> None:
        """Overwrite user evidence for ``(var, key)``."""
        self.user_facts[(var, key)] = FactState(state)
        self.invalidate_from_vars([var])

    def clear_user_fact(self, var: Any, key: AssumptionKey) -> None:
        cache_key = (var, key)
        if cache_key in self.user_facts:
            del self.user_facts[cache_key]
            self.invalidate_from_vars([var])

    def _compute(self, var: Any, key: AssumptionKey) -> FactState:
        """Infer the fact state for ``(var, key)`` by walking ancestors bottom-up.

        Collects uncached ancestors of *var* via DFS, then evaluates them
        inputs-first so each node's inputs are already cached.
        """
        # Phase 1 — collect uncached ancestors via iterative DFS
        stack = [var]
        order: list[Any] = []  # will be reversed to get bottom-up
        visited: set[int] = set()

        while stack:
            v = stack.pop()
            vid = id(v)
            if vid in visited or (v, key) in self.cache:
                continue
            visited.add(vid)
            order.append(v)
            owner = getattr(v, "owner", None)
            if owner is not None:
                stack.extend(owner.inputs)

        # Phase 2 — evaluate bottom-up (inputs before outputs)
        prev_key = getattr(self, "_current_key", None)
        self._current_key = key
        try:
            for v in reversed(order):
                if (v, key) in self.cache:
                    continue
                self.cache[(v, key)] = self._compute_one(v, key)
        finally:
            self._current_key = prev_key

        return self.cache[(var, key)]

    def _compute_one(self, var: Any, key: AssumptionKey) -> FactState:
        """Evaluate a single variable whose inputs are already cached for *key*."""
        state = FactState.UNKNOWN
        state = FactState.join(state, self.static_fact(var, key))
        state = FactState.join(
            state, self.user_facts.get((var, key), FactState.UNKNOWN)
        )

        owner = getattr(var, "owner", None)
        if owner is not None:
            input_states = [self.get(inp, key) for inp in owner.inputs]
            output_states = infer_assumption_for_node(
                owner.op, key, self, self.fgraph, owner, input_states
            )
            out_idx = owner.outputs.index(var)
            state = FactState.join(state, output_states[out_idx])

        if not state:
            for stronger, weaker_list in IMPLIES.items():
                if key in weaker_list and self.get(var, stronger):
                    state = FactState.join(state, FactState.TRUE)
                    break

        return state

    def static_fact(self, var: Any, key: AssumptionKey) -> FactState:
        """Hook for non-Op fact sources.  Returns UNKNOWN by default."""
        return FactState.UNKNOWN

    def invalidate_from_vars(self, start_vars: Iterable[Any]) -> None:
        """Clear cached facts for *start_vars* and everything downstream."""
        queue = deque(start_vars)
        seen = {id(v) for v in start_vars}
        while queue:
            var = queue.popleft()
            self._clear_cached_var(var)
            for client_node, _ in self.fgraph.clients.get(var, ()):
                for out in client_node.outputs:
                    if id(out) not in seen:
                        seen.add(id(out))
                        queue.append(out)

    def _clear_cached_var(self, var: Any) -> None:
        stale = [k for k in self.cache if k[0] is var]
        for k in stale:
            del self.cache[k]
