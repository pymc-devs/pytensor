"""The layout of every value of a graph (`graph_layouts`, `LayoutFeature`).

A `Layout` (layout.py) is a claim the generated code relies on. Layouts come from how a
value is produced:

* memory the planner provides (frames, fresh outputs) and buffers that generated code,
  Rust kernels and Scan allocate are C-contiguous;
* views get theirs from their input's (as the op declares, with the rules below);
* numba kernels allocate their outputs packed, in the order asked for (default C);
* graph inputs are DYNAMIC unless checked.

The layout of each output follows from what its op says: `llvm_layout`
(registry.py), and its `view_map`. It is binding: the
`LayoutFeature` of the graph holds the layout of every value (`node_outputs`), and the
planner makes whatever runs a node give its outputs those layouts.
"""

from __future__ import annotations

import numpy as np

from pytensor.graph.basic import Constant, Variable
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.rewriting.basic import GraphRewriter
from pytensor.link.llvm.layout import Layout
from pytensor.link.llvm.types import is_absent


def _is_tensor(var) -> bool:
    return hasattr(var.type, "ndim") and not is_absent(var.type)


def _default_layout(var, inputs_c=False):
    if not _is_tensor(var):
        return None
    # A sparse constant's data isn't an ndarray, and the backend rejects it later.
    if isinstance(var, Constant) and isinstance(var.data, np.ndarray):
        return Layout.of_const(var.data)
    if inputs_c:
        return Layout.c(var.type.ndim)
    return Layout.dynamic(var.type.ndim)


def graph_layouts(inputs, outputs, input_layouts=None, inlined=False) -> dict:
    """{variable: Layout} for the variables of the graph from the variables `inputs`
    to `outputs` (`node_outputs`; None for those that aren't tensors).
    `input_layouts`: per input, default nothing known.

    With `inlined`, every OpFromGraph is treated as inlined, which is what generated
    code does with them: its outputs get the layouts its inner graph gives them, even
    if its `execution` says it is compiled on its own. (Except a FusedElemwise
    generated as one loop, and a Blockwise or random variable with its core shapes,
    nodes of their own.)"""
    from pytensor.compile.builders import OpFromGraph
    from pytensor.graph.traversal import io_toposort
    from pytensor.link.llvm.dispatch.fused import fusable
    from pytensor.link.llvm.planning.signature import node_outputs
    from pytensor.tensor.blockwise import OpWithCoreShape

    layouts = {}
    for k, v in enumerate(inputs):
        layout = input_layouts[k] if input_layouts is not None else None
        layouts[v] = layout if layout is not None else _default_layout(v)
    for node in io_toposort(inputs, outputs):
        ins = [layouts[i] if i in layouts else _default_layout(i) for i in node.inputs]
        if (
            inlined
            and isinstance(node.op, OpFromGraph)
            and not isinstance(node.op, OpWithCoreShape)
            and not fusable(node.op, node)
        ):
            inner = node.op.fgraph
            inner_lays = graph_layouts(inner.inputs, inner.outputs, ins, inlined=True)
            layouts.update(
                zip(node.outputs, (inner_lays[var] for var in inner.outputs))
            )
            continue
        outs = node_outputs(node, ins)
        layouts.update(
            (v, var.layout) for v, var in zip(node.outputs, outs, strict=True)
        )
    for var in outputs:
        if var not in layouts:
            layouts[var] = _default_layout(var)
    return layouts


class LayoutFeature(Feature):
    """The layout of every variable of a FunctionGraph, as the LLVM backend gives it:
    `fgraph.llvm_layout(var)` (None for values that aren't tensors).

    This is where the planner gets the layouts of the graph's values from: whatever
    runs a node gives its outputs these layouts (`node_outputs`), and code is
    generated for them. `input_layout`: what is known about the graph's inputs, "any"
    strides or "c" (the linker's option of that name).

    Layouts are computed lazily and cached. Replacing an input of a node invalidates
    everything downstream: a replacement can have a weaker layout, and then facts
    derived from the old one would be wrong, not just stale.
    """

    provides = ("llvm_layout",)

    cache: dict[Variable, Layout | None]

    def __init__(self, input_layout="any"):
        self.input_layout = input_layout
        self.cache = {}

    def clone(self):
        return type(self)(self.input_layout)

    def on_attach(self, fgraph):
        if "llvm_layout" in fgraph._feature_methods:
            raise AlreadyThere("a LayoutFeature is already attached")

    def on_detach(self, fgraph):
        self.cache.clear()

    def on_prune(self, fgraph, node, reason):
        for var in node.outputs:
            self.cache.pop(var, None)

    def on_change_input(self, fgraph, node, i, var, new_var, reason=None):
        todo = list(node.outputs)
        while todo:
            v = todo.pop()
            if self.cache.pop(v, None) is None:
                continue  # never computed: neither is anything downstream
            for client, _ in fgraph.clients.get(v, ()):
                todo.extend(client.outputs)

    def set_input_layout(self, input_layout):
        if input_layout != self.input_layout:
            self.input_layout = input_layout
            self.cache.clear()

    def llvm_layout(self, fgraph, var) -> Layout | None:
        """The layout of `var`."""
        from pytensor.link.llvm.planning.signature import node_outputs

        cache = self.cache
        if var in cache:
            return cache[var]
        stack = [var]
        while stack:
            v = stack[-1]
            if v in cache:
                stack.pop()
                continue
            node = v.owner
            if node is None:
                cache[v] = _default_layout(
                    v, self.input_layout == "c" and v in fgraph.inputs
                )
                stack.pop()
                continue
            missing = [i for i in node.inputs if i not in cache]
            if missing:
                stack.extend(missing)
                continue
            outs = node_outputs(node, [cache[i] for i in node.inputs])
            cache.update(
                (v, output.layout) for v, output in zip(node.outputs, outs, strict=True)
            )
            stack.pop()
        return cache[var]


def layout_feature(fgraph) -> LayoutFeature:
    """The `LayoutFeature` of `fgraph`, attached if it has none."""
    feature: LayoutFeature | None = fgraph._feature_methods.get("llvm_layout")
    if feature is None:
        feature = LayoutFeature()
        fgraph.attach_feature(feature)
    return feature


class AttachLayoutFeature(GraphRewriter):
    """Attaches a `LayoutFeature` to the graph, for the rewrites after it (the LLVM
    linker requires it)."""

    def add_requirements(self, fgraph):
        layout_feature(fgraph)

    def apply(self, fgraph):
        pass


__all__ = [
    "AttachLayoutFeature",
    "LayoutFeature",
    "graph_layouts",
    "layout_feature",
]
