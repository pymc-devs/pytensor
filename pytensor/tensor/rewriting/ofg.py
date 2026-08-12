from pytensor.compile.rewriting import inline_ofg_node
from pytensor.graph import node_rewriter
from pytensor.graph.rewriting.basic import MergeOptimizer, dfs_rewriter
from pytensor.tensor.basic import AllocDiag
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.rewriting.elemwise import fuse_seqopt
from pytensor.tensor.special import (
    LogAddExp,
    LogSoftmax,
    LogSumExp,
    Softmax,
    XLog1PY,
    XLogY,
)


@register_specialize("inline_ofg")
@node_rewriter([AllocDiag, XLogY, XLog1PY, LogSumExp, LogAddExp])
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


@node_rewriter([Softmax, LogSoftmax, LogSumExp, LogAddExp, XLogY, XLog1PY])
def inline_symbolic_for_fusion(fgraph, node):
    """Inline `SymbolicOp`s so their bodies can fuse.

    The inner graph is otherwise compiled on its own, making the op a fusion barrier
    both inside and across its boundary. Ops are listed explicitly, as some are worth
    keeping opaque anyway (`KroneckerProduct` is: its body is one broadcasting `mul`
    behind a `reshape`). Backends that dispatch these ops directly (JAX, PyTorch and MLX
    have their own `Softmax`) exclude the whole fusion pass, so they keep them whole.

    Ops that `late_inline_OpFromGraph` already inlines at specialize are listed here
    too, since fusion also runs in modes that skip specialize.
    """
    return inline_ofg_node(node)


# Fusion is the only thing that needs the body, so inline as its first step: every
# Op-level rewrite still matches on the op.
fuse_seqopt.register(
    "inline_symbolic_for_fusion",
    dfs_rewriter(inline_symbolic_for_fusion),
    "fast_run",
    "fusion",
    position=0,
)
# Each op is inlined from its own inner graph, so two ops over the same inputs (a
# `Softmax` and a `LogSoftmax` over the same logits) yield structurally identical but
# distinct copies of everything they share. The `merge2` pass cannot clean this up: it
# sits at the same optdb position as the fusion pass, where ties are broken by name, so
# it only runs once fusion has pulled both copies into the `Composite`.
fuse_seqopt.register(
    "merge_inlined_symbolic",
    MergeOptimizer(),
    "fast_run",
    "fusion",
    "merge",
    position=0.5,
)
