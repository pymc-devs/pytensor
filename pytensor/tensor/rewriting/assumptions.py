from pytensor.assumptions import AssumptionFeature
from pytensor.assumptions.specify import SpecifyAssumptions
from pytensor.compile.mode import optdb
from pytensor.graph.rewriting.basic import GraphRewriter


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

        assumption_feature = getattr(fgraph, "assumption_feature", None)
        if assumption_feature is None:
            assumption_feature = AssumptionFeature()
            fgraph.attach_feature(assumption_feature)

        replacements = {}
        for node in nodes:
            [out] = node.outputs
            # Resolve the asserted facts into the cache.
            for key, _ in node.op.assumptions:
                assumption_feature.get(out, key)
            # Drain the marker: redirect its consumers to the raw input, peeling
            # already-drained nested SpecifyAssumptions -- ``nodes`` is in toposort
            # order, so a single replace_all collapses ``assume(assume(...))`` chains
            # all the way down.
            inp = node.inputs[0]
            while inp in replacements:
                inp = replacements[inp]
            replacements[out] = inp

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
