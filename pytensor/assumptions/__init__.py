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
import pytensor.assumptions.tridiagonal
import pytensor.assumptions.unit_magnitude
from pytensor.assumptions.bundles import register_matrix_property_rules
from pytensor.assumptions.core import (
    ALL_KEYS,
    DIAGONAL,
    IMPLIES,
    KEY_REGISTRY,
    LOWER_TRIANGULAR,
    MATRIX_KEYS,
    ORTHOGONAL,
    PERMUTATION,
    POSITIVE_DEFINITE,
    SELECTION,
    SYMMETRIC,
    TRIDIAGONAL,
    UNIQUE_INDICES,
    UNIT_MAGNITUDE,
    UPPER_TRIANGULAR,
    AssumptionFeature,
    AssumptionKey,
    ConflictingAssumptionsError,
    FactState,
    check_assumption,
    get_assumption_value,
    register_assumption,
    register_constant_inference,
    register_implies,
    register_universal_assumption,
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
    implies) is also FALSE. Any other value that differs from ``key.unknown`` appears
    as :meth:`AssumptionKey.format_fact` prints it. Return ``""`` when nothing is
    known.
    """
    states = {}
    conflicted = set()
    for key in ALL_KEYS:
        try:
            states[key] = feature.get(var, key)
        except ConflictingAssumptionsError:
            states[key] = key.unknown
            conflicted.add(key)

    tokens = []
    for key in ALL_KEYS:
        state = states[key]
        label = key.short_name or key.name
        if key in conflicted:
            tokens.append(f"{label}=CONFLICT")
            continue
        if key.is_true(state):
            if any(
                key in weaker and stronger.is_true(states[stronger])
                for stronger, weaker in IMPLIES.items()
            ):
                continue
        elif key.is_false(state):
            if any(w.is_false(states[w]) for w in IMPLIES.get(key, ())):
                continue
        elif state == key.unknown:
            continue
        tokens.append(key.format_fact(label, state))

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
