from pytensor.assumptions import AssumptionFeature
from pytensor.assumptions.specify import SpecifyAssumptions
from pytensor.compile.mode import optdb
from pytensor.graph.rewriting.basic import GraphRewriter, node_rewriter
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)


@register_canonicalize
@register_stabilize("fast_compile")
@register_specialize("fast_compile")
@node_rewriter([SpecifyAssumptions])
def local_drain_specify_assumptions(fgraph, node):
    """Record a marker's declarations before replacing it with its input.

    Register in each pass so markers introduced by rewrites are drained too.
    The feature carries the resolved facts to the input during replacement.
    """
    feature = getattr(fgraph, "assumption_feature", None)
    if feature is None:
        feature = AssumptionFeature()
        fgraph.attach_feature(feature)

    for key, _ in node.op.assumptions:
        feature.get(node.out, key)
    return node.inputs


class DrainSpecifyAssumptions(GraphRewriter):
    """Drain initial markers before they obstruct canonicalization."""

    def apply(self, fgraph):
        if not any(
            isinstance(node.op, SpecifyAssumptions) for node in fgraph.apply_nodes
        ):
            return

        replacements = {}
        for node in fgraph.toposort():
            if isinstance(node.op, SpecifyAssumptions):
                [inp] = local_drain_specify_assumptions.transform(fgraph, node)
                # Topological order lets nested markers collapse in one replacement.
                replacements[node.out] = replacements.get(inp, inp)

        fgraph.replace_all(
            tuple(replacements.items()), reason="drain_specify_assumptions"
        )


optdb.register(
    "drain_specify_assumptions",
    DrainSpecifyAssumptions(),
    "fast_run",
    "fast_compile",
    position=0.8,
)
