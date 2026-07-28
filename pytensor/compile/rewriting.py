"""Backend inner-graph rewriting: the generic baking helper and the ``OpFromGraph`` registrations and inlining."""

from collections import defaultdict
from functools import singledispatch

from pytensor.compile.aliasing import (
    add_supervisor_to_fgraph,
    insert_deepcopy,
)
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.io import In, Out
from pytensor.compile.mode import optdb
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FrozenFunctionGraph
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    dfs_rewriter,
    get_active_mode,
    graph_rewriter,
    node_rewriter,
)
from pytensor.link.basic import PerformLinker
from pytensor.link.c.basic import CLinker, OpWiseCLinker
from pytensor.link.jax.linker import JAXLinker
from pytensor.link.mlx.linker import MLXLinker
from pytensor.link.numba.linker import NumbaLinker
from pytensor.link.pytorch.linker import PytorchLinker
from pytensor.link.vm import VMLinker


def rewrite_inner_graph(fgraph, match, rewrite):
    """Bake the inner graphs of the ``match``-ing nodes for the active backend.

    An inner-graph op is matched directly or as the ``core_op`` of a `Blockwise`
    (so an `OpFromGraph`/`Scan`/`Minimize` wrapped in a `Blockwise` still gets its
    inner graph optimized for the backend). Nodes are grouped by ``(inner op, core
    input types)`` -- the inplace/aliasing contract a ``rewrite`` bakes depends
    only on the (un-batched, core-level) buffer shapes, which those types capture
    -- so each distinct inner graph is prepared once and shared. For each group
    ``rewrite(linker, op, node, inner, mode=...)`` mutates the unfrozen ``inner``
    graph in place (optimize + features + boundary deepcopies), deriving its own
    optimizer from ``mode`` -- ``mode.optimizer`` to bake inplace, or
    ``mode.excluding("inplace").optimizer`` to leave the graph functional; the new
    op (re-wrapped in its `Blockwise` if needed) then replaces the nodes.
    """
    from pytensor.tensor.blockwise import Blockwise

    mode = get_active_mode(fgraph)
    linker = mode.linker

    def unwrap(node):
        """Return ``(inner_op, inner_node, rewrap)`` for a matching node, else ``None``.

        ``inner_node`` is the node whose input types the ``rewrite`` sees; for a
        `Blockwise` it is the *core* (un-batched) node, so per-node shape logic
        (e.g. `Scan`'s destroyability) reasons about the core buffers. ``rewrap``
        rebuilds the outer op from a new inner op.
        """
        op = node.op
        if isinstance(op, Blockwise) and match(op.core_op):
            core_node = op._create_dummy_core_node(node.inputs)

            def rewrap(new_core_op, op=op):
                return type(op)(
                    new_core_op,
                    signature=op.signature,
                    name=op.name,
                    gufunc_spec=op.gufunc_spec,
                    destroy_map=op.destroy_map,
                )

            return op.core_op, core_node, rewrap
        if match(op):
            return op, node, lambda new_op: new_op
        return None

    groups: dict = defaultdict(list)
    node_meta: dict = {}
    for node in fgraph.apply_nodes:
        if (meta := unwrap(node)) is not None:
            inner_op, inner_node, _ = meta
            # Ops sharing a frozen inner graph but with different destroy/view maps
            # bake differently (the maps decide which taps may be destroyed and which
            # boundary deepcopies are needed), so they must group separately.
            # The node input types are not redundant with the op's hash/eq either:
            # they can be more specific than the op's nominal types (e.g. static
            # shapes), and per-node contracts like Scan's destroyability depend on them.
            key = (
                inner_op,
                tuple(i.type for i in inner_node.inputs),
                tuple((o, tuple(v)) for o, v in sorted(inner_op.destroy_map.items())),
                tuple((o, tuple(v)) for o, v in sorted(inner_op.view_map.items())),
            )
            groups[key].append(node)
            node_meta[node] = meta
    if not groups:
        return

    node_to_new_op: dict = {}
    for nodes in groups.values():
        rep_node = nodes[0]
        inner_op, inner_node, _ = node_meta[rep_node]
        inner = inner_op.fgraph.unfreeze()
        # Expose the compile mode to nested inner-graph rewrites (mirrors ``FunctionMaker``)
        inner._compile_mode = mode
        try:
            rewrite(linker, inner_op, inner_node, inner, mode=mode)
        finally:
            del inner._compile_mode
        new_inner_op = inner_op.clone_with_inner_graph(inner)
        if new_inner_op != inner_op:
            for node in nodes:
                node_to_new_op[node] = node_meta[node][2](new_inner_op)

    if not node_to_new_op:
        return

    for node in fgraph.toposort():
        new_op = node_to_new_op.get(node)
        if new_op is not None:
            new_node = new_op.make_node(*node.inputs)
            fgraph.replace_all(
                list(zip(node.outputs, new_node.outputs, strict=True)),
                reason="rewrite_inner_graph",
            )


@singledispatch
def rewrite_ofg_inner_graph(linker, op, node, inner, *, mode):
    """Rewrite an ``OpFromGraph`` inner graph (in place) for ``linker``'s backend."""
    raise NotImplementedError(
        f"Linker {type(linker).__name__} has not registered an OpFromGraph "
        "inner-graph rewrite"
    )


def _ofg_inner_optimizer(mode):
    # Recognition rewrites fold a pattern into an inner-graph op (e.g.
    # ``exp(x) / sum(exp(x))`` -> ``Softmax``, itself an ``OpFromGraph``). Running
    # them on an ``OpFromGraph`` inner graph -- which may *be* that pattern --
    # would re-create the op inside itself and recurse without end.
    return mode.excluding("symbolic_op_recognition").optimizer


@rewrite_ofg_inner_graph.register(VMLinker)
@rewrite_ofg_inner_graph.register(PerformLinker)
@rewrite_ofg_inner_graph.register(CLinker)
@rewrite_ofg_inner_graph.register(OpWiseCLinker)
@rewrite_ofg_inner_graph.register(NumbaLinker)
def destructive_rewrite_ofg_inner_graph(linker, op, node, inner, *, mode):
    # ``OpFromGraph`` must not mutate its inputs, so all are protected; inplace may
    # still be baked between purely internal buffers.
    input_specs = [In(x, borrow=True, mutable=False) for x in inner.inputs]
    add_supervisor_to_fgraph(fgraph=inner, input_specs=input_specs, accept_inplace=True)
    _ofg_inner_optimizer(mode).rewrite(inner)
    # The op's outputs must not alias its inputs or each other (it declares no
    # view_map, so the outer graph cannot see such aliases); deepcopies break any
    # boundary alias the optimized graph ends up with.
    output_specs = [Out(o, borrow=False) for o in inner.outputs]
    insert_deepcopy(inner, wrapped_inputs=input_specs, wrapped_outputs=output_specs)


@rewrite_ofg_inner_graph.register(JAXLinker)
@rewrite_ofg_inner_graph.register(PytorchLinker)
@rewrite_ofg_inner_graph.register(MLXLinker)
def functional_rewrite_ofg_inner_graph(linker, op, node, inner, *, mode):
    """Structurally optimize the inner graph for the functional JIT backends."""
    _ofg_inner_optimizer(mode).rewrite(inner)


@graph_rewriter
def ofg_inner_graph(fgraph):
    # ``OpWithCoreShape`` is imported lazily: at module import time
    # ``pytensor.tensor.random`` is only partially initialized. ``*WithCoreShape``
    # are leaf backend ops with dedicated dispatch; re-optimizing them would loop.
    from pytensor.tensor.random.op import OpWithCoreShape

    rewrite_inner_graph(
        fgraph,
        lambda op: isinstance(op, OpFromGraph) and not isinstance(op, OpWithCoreShape),
        rewrite_ofg_inner_graph,
    )


optdb.register(
    "ofg_inner_graph",
    ofg_inner_graph,
    "minimum_compile",
    "compile_inner_graph",
    position=49.6,
)


def inline_ofg_node(node: Apply) -> list[Variable]:
    frozen_fg: FrozenFunctionGraph = node.op.fgraph
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
