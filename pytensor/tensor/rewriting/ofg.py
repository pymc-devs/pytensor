from pytensor.compile import optdb
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import Apply, Variable, node_rewriter
from pytensor.graph.fg import FrozenFunctionGraph
from pytensor.graph.rewriting.basic import copy_stack_trace, dfs_rewriter
from pytensor.tensor.basic import AllocDiag
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.special import XLog1PY, XLogY


def inline_ofg_node(node: Apply) -> list[Variable]:
    frozen_fg: FrozenFunctionGraph = node.op._frozen_fgraph
    replacements = dict(zip(frozen_fg.inputs, node.inputs))
    inlined_outs = frozen_fg.bind(replacements)
    copy_stack_trace(frozen_fg.outputs, inlined_outs)
    return inlined_outs


@node_rewriter([OpFromGraph])
def inline_ofg_expansion(fgraph, node):
    """
    This optimization expands internal graph of OpFromGraph.
    Only performed if node.op.is_inline == True
    Doing so can improve optimization at the cost of compilation speed.
    """
    op = node.op
    if not op.is_inline:
        return False

    return inline_ofg_node(node)


# We want to run this before the first merge optimizer
# and before the first scan optimizer.
optdb.register(
    "inline_ofg_expansion",
    dfs_rewriter(inline_ofg_expansion),
    "fast_compile",
    "fast_run",
    position=-0.01,
)


@register_specialize("inline_ofg")
@node_rewriter([AllocDiag, XLogY, XLog1PY])
def late_inline_OpFromGraph(fgraph, node):
    """
    Inline `OpFromGraph` nodes.

    OpFromGraph nodes are used to compactly represent the output of a function graph. Certain `Ops`, like, einsum,
    diag, and kron, are implemented using pytensor `Op`s. As a result, their outputs are not a single `Op`, but a
    graph. To allow rewrites to easily spot and manipulate these "composite functions", we use the `OpFromGraph` node.
    This node is a thin wrapper around the output graph. It is not, however, meant to be included in the final
    program, because it hides the inner graph from certain optimizations.

    This rewrite specifies that all `OpFromGraph` nodes should be replaced by their inner graphs by setting the
    `inplace=True` flag.

    Parameters
    ----------
    fgraph: FunctionGraph
        The function graph being rewritten
    node: Apply
        Node of the function graph to be optimized

    Returns
    -------

    """
    return inline_ofg_node(node)
