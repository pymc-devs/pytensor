from pytensor import clone_replace
from pytensor.graph import node_rewriter
from pytensor.tensor.basic import AllocDiag
from pytensor.tensor.rewriting.basic import register_specialize


@register_specialize("inline_ofg")
@node_rewriter([AllocDiag])
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
    op = node.op
    return clone_replace(op.inner_outputs, dict(zip(op.inner_inputs, node.inputs)))
