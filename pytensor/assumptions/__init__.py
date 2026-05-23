# Import every rule module so its registrations land on the global registry that
# ``check_assumption`` consults. The property modules pull in the shared helper
# modules (alloc, dimshuffle, subtensor, dot, elemwise) transitively.
import pytensor.assumptions.alloc
import pytensor.assumptions.blockwise
import pytensor.assumptions.diagonal
import pytensor.assumptions.dimshuffle
import pytensor.assumptions.orthogonal
import pytensor.assumptions.permutation
import pytensor.assumptions.positive_definite
import pytensor.assumptions.reshape
import pytensor.assumptions.selection
import pytensor.assumptions.shape
import pytensor.assumptions.subtensor
import pytensor.assumptions.symmetric
import pytensor.assumptions.triangular
from pytensor.assumptions.core import (
    ALL_KEYS,
    DIAGONAL,
    IMPLIES,
    LOWER_TRIANGULAR,
    ORTHOGONAL,
    PERMUTATION,
    POSITIVE_DEFINITE,
    SELECTION,
    SYMMETRIC,
    UPPER_TRIANGULAR,
    AssumptionFeature,
    AssumptionKey,
    ConflictingAssumptionsError,
    FactState,
    check_assumption,
    register_assumption,
    register_constant_inference,
    register_implies,
)
from pytensor.assumptions.specify import (
    SpecifyAssumptions,
    assume,
    specify_assumption_rule,
)
from pytensor.graph.fg import FunctionGraph


def summarize_assumptions(feature: AssumptionFeature, var) -> str:
    """Return a compact ``{...}`` tag of the facts known about *var*.

    Each TRUE fact appears as its ``short_name``; each FALSE fact as ``!short_name``.
    Implied facts are pruned: a TRUE fact is dropped when a stronger fact (one that
    implies it) is also TRUE, and a FALSE fact is dropped when a weaker fact (one it
    implies) is also FALSE. Return ``""`` when nothing is known.
    """
    states = {}
    for key in ALL_KEYS:
        try:
            states[key] = feature.get(var, key)
        except ConflictingAssumptionsError:
            states[key] = FactState.CONFLICT

    tokens = []
    for key in ALL_KEYS:
        state = states[key]
        label = key.short_name or key.name
        if state is FactState.TRUE:
            if any(
                key in weaker and states[stronger] is FactState.TRUE
                for stronger, weaker in IMPLIES.items()
            ):
                continue
            tokens.append(label)
        elif state is FactState.FALSE:
            if any(states[w] is FactState.FALSE for w in IMPLIES.get(key, ())):
                continue
            tokens.append(f"!{label}")
        elif state is FactState.CONFLICT:
            tokens.append(f"{label}=CONFLICT")

    return "{" + ", ".join(tokens) + "}" if tokens else ""


def assumption_tags(outputs) -> dict:
    """Map each variable reachable from *outputs* to its :func:`summarize_assumptions`
    tag, omitting variables with no known facts."""
    fgraph = FunctionGraph(outputs=list(outputs), clone=False)
    feature = AssumptionFeature()
    fgraph.attach_feature(feature)
    tags = {}
    for var in fgraph.variables:
        tag = summarize_assumptions(feature, var)
        if tag:
            tags[var] = tag
    return tags
