"""Fold constant Scan computations together with their temporary input buffers."""

from pytensor.compile.mode import Mode, get_mode
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import HasMultipleInnerGraph
from pytensor.graph.rewriting.basic import get_active_mode, in2out, node_rewriter
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.traversal import graph_inputs
from pytensor.link.vm import VMLinker
from pytensor.scan.op import Scan
from pytensor.tensor.basic import Alloc, AllocEmpty
from pytensor.tensor.rewriting.basic import (
    constant_folding,
    local_remove_useless_assert,
    unconditional_constant_folding,
)


@node_rewriter([Alloc, AllocEmpty])
def fold_buffer(fgraph, node):
    return unconditional_constant_folding.transform(fgraph, node)


scan_inputs_constant_folding = in2out(
    local_remove_useless_assert,
    fold_buffer,
    constant_folding,
    ignore_newtrees=True,
)


def _folding_mode(fgraph: FunctionGraph, op: Scan) -> Mode:
    """Build a pure-VM, non-inplace compile mode matching the active optimizer.

    Constant folding evaluates the Scan via its Python/VM thunk before the late
    ``scan_inner_graph`` pass runs. Rewriting a transient copy of the inner
    graph with the active mode's algebraic rewrites ensures folded Scan outputs
    agree with the outer graph (e.g., canonicalization rewrites such as
    ``eigh_of_diag``) without introducing backend-only or inplace ops that the
    folding thunk cannot execute.
    """
    mode = get_active_mode(fgraph)
    if (custom_mode := getattr(op, "mode", None)) is not None:
        mode = get_mode(custom_mode)
    _, opt = mode.get_linker_optimizer(mode.provided_linker, mode.provided_optimizer)
    if isinstance(opt, RewriteDatabaseQuery):
        opt = opt.excluding("inplace", "jax", "numba", "pytorch", "mlx", "cxx_only")
    return Mode(
        linker=VMLinker(use_cloop=False, c_thunks=False),
        optimizer=opt,
        db=mode.optdb,
    )


@node_rewriter([Scan])
def scan_constant_folding(fgraph, node):
    """Fold a closed constant Scan, allowing temporary Alloc/AllocEmpty inputs.

    Buffer construction is evaluated on a copy. Only the completed Scan outputs
    replace the original graph, so intermediate buffers never become persistent
    constants when the Scan cannot be folded.
    """
    if not all(isinstance(inp, Constant) for inp in graph_inputs(node.inputs)):
        return None

    # Allow temporary allocations and nested Scans, while honoring folding
    # opt-outs across all inner graphs.
    pending = [node.op.fgraph]
    seen = set()
    while pending:
        graph = pending.pop()
        if id(graph) in seen:
            continue
        seen.add(id(graph))
        for inner_node in graph.apply_nodes:
            if isinstance(inner_node.op, Alloc | AllocEmpty):
                continue
            if isinstance(inner_node.op, Scan):
                pending.append(inner_node.op.fgraph)
                continue
            if not inner_node.op.do_constant_folding(graph, inner_node):
                return None
            if isinstance(inner_node.op, HasMultipleInnerGraph):
                pending.extend(inner_node.op.inner_graphs.values())

    inputs_fg = FunctionGraph(outputs=node.inputs, clone=True)
    scan_inputs_constant_folding.rewrite(inputs_fg)
    if not all(isinstance(inp, Constant) for inp in inputs_fg.outputs):
        return None

    inner = node.op.fgraph.unfreeze()
    _folding_mode(fgraph, node.op).optimizer.rewrite(inner)
    rewritten_op = node.op.clone_with_inner_graph(inner)
    constant_node = rewritten_op.make_node(*inputs_fg.outputs)
    return unconditional_constant_folding.transform(fgraph, constant_node)
