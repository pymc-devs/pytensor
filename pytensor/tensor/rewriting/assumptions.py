from pytensor.assumptions import ALL_KEYS, AssumptionFeature
from pytensor.assumptions.specify import SpecifyAssumptions
from pytensor.compile.mode import optdb
from pytensor.graph.basic import Variable
from pytensor.graph.rewriting.basic import GraphRewriter, node_rewriter
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)


_KEY_BY_NAME = {key.name: key for key in ALL_KEYS}


def _assumption_feature(fgraph) -> AssumptionFeature:
    feature = getattr(fgraph, "assumption_feature", None)
    if feature is None:
        feature = AssumptionFeature()
        fgraph.attach_feature(feature)
    return feature


def _drain_marker(feature: AssumptionFeature, node) -> Variable:
    """Resolve one marker's declarations, returning the input to redirect consumers to.

    Nested markers are peeled so that ``assume(assume(...))`` collapses in one step.
    """
    [out] = node.outputs
    for name, _ in node.op.assumptions:
        feature.get(out, _KEY_BY_NAME[name])

    inp: Variable = node.inputs[0]
    while inp.owner is not None and isinstance(inp.owner.op, SpecifyAssumptions):
        inp = inp.owner.inputs[0]
    return inp


class DrainSpecifyAssumptions(GraphRewriter):
    """Drain ``SpecifyAssumptions`` declarations into the ``AssumptionFeature`` and
    remove the marker nodes.

    A ``SpecifyAssumptions`` node is an opaque view of its input, so it blocks any
    rewrite that pattern-matches across it. Running before canonicalization, this
    rewriter resolves every declared fact into the feature cache (where the
    ``check_assumption`` consumers read it) and then drops the node.
    """

    def apply(self, fgraph):
        if not any(
            isinstance(node.op, SpecifyAssumptions) for node in fgraph.apply_nodes
        ):
            return None  # Fast bail out

        nodes = [
            node
            for node in fgraph.toposort()
            if isinstance(node.op, SpecifyAssumptions)
        ]

        feature = _assumption_feature(fgraph)
        replacements = {node.outputs[0]: _drain_marker(feature, node) for node in nodes}

        fgraph.replace_all(
            tuple(replacements.items()), reason="drain_specify_assumptions"
        )


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([SpecifyAssumptions])
def drain_specify_assumptions_node(fgraph, node):
    """Drain a marker that appears after the whole-graph pass has already run.

    A rewrite can then declare an assumption the same way construction does.
    """
    return [_drain_marker(_assumption_feature(fgraph), node)]


optdb.register(
    "drain_specify_assumptions",
    DrainSpecifyAssumptions(),
    "fast_run",
    "fast_compile",
    position=0.8,
)
