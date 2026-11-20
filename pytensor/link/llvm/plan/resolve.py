"""Stage 2: how each node is run.

A node is run by generated code (an emitter, `llvm_emit`), by a Rust kernel, or by a
numba kernel. `Resolver` decides, and says what the choice means for the node's outputs
(how they come to be, and their layouts): a `CalledKernel` describes a kernel, `None`
stands for generated code.

The layouts of the outputs are not a choice: they are the binding ones (what
`llvm_layout` declares, `node_outputs`), and what runs a node must give its outputs
layouts that imply them (`conforms`). A Rust kernel that doesn't is not used.

The items of a program (`program.py`) and nodes inside generated code (a unit, a Scan's
loop body) choose differently:

* An item of a program prefers a Rust kernel: there is nothing to compile.
* Inside generated code, generated code is preferred: it is inlined into its
  surroundings, a kernel is a call.

`resolve_items` resolves the items of a unit or a loop body in order, following the
binding layouts from one item to the next. Code generation consumes the result
(`Resolved`) and makes no choices of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from pytensor.graph.basic import Constant
from pytensor.graph.traversal import io_toposort
from pytensor.link.llvm.codegen.blockwise import (
    blockwise_core,
    blockwise_core_layouts,
)
from pytensor.link.llvm.codegen.call import CalledKernel
from pytensor.link.llvm.codegen.interface import llvm_emit
from pytensor.link.llvm.codegen.node import build_node_function
from pytensor.link.llvm.codegen.signature import (
    Output,
    OutputMode,
    Signature,
    node_outputs,
    node_signature,
)
from pytensor.link.llvm.codegen.types import ir, is_absent, repr_of
from pytensor.link.llvm.layout import Layout, graph_layouts, itemsize_of, shape_of
from pytensor.link.llvm.plan.flatten import flatten
from pytensor.link.llvm.plan.values import NodeItem, Values
from pytensor.link.llvm.rust_kernels import interp, rust_kernel, rust_kernel_outputs
from pytensor.scan.op import Scan
from pytensor.tensor.blockwise import Blockwise


@dataclass
class Call:
    """An item that is a kernel called with values, not a node: what the planner
    compiled on its own (a node, a compiled unit)."""

    #: the values passed, one per argument of the kernel's signature
    inputs: list
    #: the values produced, one per output of the signature
    outputs: list
    description: str


@dataclass
class Resolved:
    """An item of generated code with everything decided."""

    item: NodeItem | Call
    #: the layouts of its inputs (of a `Call`: of its arguments)
    in_layouts: list
    #: the kernel it calls; None: its code is generated (and inlined)
    kernel: CalledKernel | None
    #: the signature of its function: representations of inputs and outputs
    signature: Signature

    @property
    def arguments(self) -> list:
        """The values passed, one per input of the signature."""
        if isinstance(self.item, Call):
            return self.item.inputs
        return [
            vid
            for vid, var in zip(self.item.inputs, self.item.node.inputs, strict=True)
            if not is_absent(var.type)
        ]

    @property
    def description(self) -> str:
        if isinstance(self.item, Call):
            return self.item.description
        if self.kernel is not None:
            return self.kernel.description
        return str(self.item.node.op)


@dataclass
class Body:
    """A graph with every decision made about how its nodes run (`Resolver.graph`):
    a Scan's loop body, a Blockwise's core, a unit's inner graph, or the program.

    Its items are in the order they run, and refer to values by their vids."""

    #: the values (vids) of the graph's inputs and outputs
    in_ids: list
    out_ids: list
    items: list  # of Resolved
    #: {vid: (type, data)} of the constants the items read
    constants: dict


class Resolver:
    def __init__(self, options, stats, values: Values):
        self.opts = options
        self.stats = stats
        self.values = values
        #: (inputs, outputs, layouts) -> Body (`graph`)
        self._graphs: dict[tuple, Body] = {}

    # ------------------------------------------------------------ the kinds of kernels

    def has_inner_graph(self, node) -> bool:
        """Does the node's generated code run an inner graph (a Scan runs its loop body,
        a Blockwise its core graph)?

        For such nodes the planner tries generated code before a Rust kernel, because
        the generated loop has the inner graph inlined into it."""
        return isinstance(node.op, Scan | Blockwise)

    def graph(self, inputs, outputs, in_layouts) -> Body:
        """Decide how each node of a graph runs, when the graph runs inside generated
        code (`resolve_items`).

        The graph goes from the variables `inputs` to the variables `outputs`.
        `in_layouts` gives the layout of each input (None for inputs that aren't
        tensors). Inputs that are constants stay constants of the graph.

        The layouts of the graph's variables come from `graph_layouts`, with every
        OpFromGraph inlined (that is what generated code does with them). The values
        inside an inlined OpFromGraph aren't variables of the graph: they get their
        layouts from `node_outputs` while resolving, as in the planner.

        The result is cached: the same graph with the same input layouts is resolved
        only once."""
        key = (tuple(inputs), tuple(outputs), tuple(in_layouts))
        if key in self._graphs:
            return self._graphs[key]
        values = self.values
        scope: dict = {}
        in_ids = [
            values.of(v, scope) if isinstance(v, Constant) else values.new(v.type)
            for v in inputs
        ]
        scope.update(zip(inputs, in_ids, strict=True))
        items = flatten(values, io_toposort(inputs, outputs), scope, inside_unit=True)
        out_ids = [values.of(o, scope) for o in outputs]
        known = graph_layouts(inputs, outputs, in_layouts, inlined=True)
        layouts: dict = {}
        for v, k in scope.items():
            lay = Layout.of_const(v.data) if k in values.consts else known.get(v)
            if lay is not None:
                layouts.setdefault(k, lay)
        resolved = self.resolve_items(items, layouts)
        body = Body(in_ids, out_ids, resolved, values.constants_of(items, out_ids))
        self._graphs[key] = body
        return body

    def rust(self, node, in_layouts, binding) -> CalledKernel | None:
        """The node's Rust kernel, if it has one that gives its outputs the `binding`
        layouts."""
        k = rust_kernel(node.op, node, in_layouts, self.opts.linalg_parallel)
        if k is None:
            return None
        outputs = rust_kernel_outputs(k, node, in_layouts)
        if not conforms(node, outputs, binding):
            return None
        self.stats.n_rust_kernels += 1
        return CalledKernel(k.address, k.config, outputs, repr(k), k)

    def numba(self, node, binding) -> CalledKernel:
        """The node's numba implementation as a kernel. It allocates its outputs, packed
        in the `binding` layouts (C-contiguous where they aren't packed: they must be
        implied by it)."""
        from pytensor.link.llvm.numba_kernels import numba_kernel
        from pytensor.tensor.random.type import RandomType

        if any(isinstance(v.type, RandomType) for v in (*node.inputs, *node.outputs)):
            raise NotImplementedError(
                f"LLVM backend: no implementation of the random variable {node.op}"
            )
        layouts = [
            None if lay is None else _allocated_layout(node.op, o, lay)
            for o, lay in zip(node.outputs, binding, strict=True)
        ]
        orders = [
            None if lay is None or lay.is_c(shape_of(o)) else tuple(lay.order)
            for o, lay in zip(node.outputs, layouts, strict=True)
        ]
        kernel = numba_kernel(node, None if not any(orders) else orders)
        self.stats.n_numba += 1
        self.stats.numba.append(str(node.op))
        outs = [
            Output(repr_of(o.type), OutputMode.ABSENT)
            if is_absent(o.type)
            else Output(repr_of(o.type, lay), OutputMode.RETURNED)
            for o, lay in zip(node.outputs, layouts, strict=True)
        ]
        destroyed = getattr(node.op, "destroy_map", None) or {}
        return CalledKernel(
            interp().numba_entry(),
            kernel.address,
            outs,
            f"numba: {node.op}",
            kernel,
            frozenset(i for ins in destroyed.values() for i in ins),
        )

    def fallback(self, node, in_layouts, binding) -> CalledKernel:
        """The kernel for a node we don't generate: Rust, else numba."""
        kernel = self.rust(node, in_layouts, binding)
        if kernel is not None:
            return kernel
        if not self.opts.numba_fallback:
            raise NotImplementedError(f"No LLVM backend implementation for {node.op}")
        return self.numba(node, binding)

    # ------------------------------------------------------------ inside generated code

    def can_generate(self, node, in_layouts) -> bool:
        """Is there an emitter that handles this node? (Tried in a scratch module: an
        emitter says no by raising, possibly halfway through.)"""
        if llvm_emit.dispatch(type(node.op)) is llvm_emit.dispatch(object):
            return False
        try:
            build_node_function(
                ir.Module("probe"), node, "probe", in_layouts=in_layouts, parallel=False
            )
            return True
        except NotImplementedError:
            return False

    def inside(self, node, in_layouts, binding) -> CalledKernel | None:
        """How a node inside generated code runs: None if its code is generated (and
        inlined), else the kernel that runs it."""
        if isinstance(node.op, Scan):
            return None
        if isinstance(node.op, Blockwise):
            # generated if its core graph can be resolved (the emitter does it again,
            # from the cache)
            core_layouts = blockwise_core_layouts(node, in_layouts)
            try:
                self.graph(*blockwise_core(node), core_layouts)
                return None
            except NotImplementedError:
                return self.fallback(node, in_layouts, binding)
        if self.can_generate(node, in_layouts):
            return None
        return self.fallback(node, in_layouts, binding)

    def resolve_items(self, items, layouts: dict) -> list:
        """Decide how each of `items` runs, in order.

        `layouts` (vid -> Layout) must hold the layouts of the values the items read
        from outside. It may also hold layouts already known for values the items
        produce; those are binding. The layouts of the other values the items produce
        are derived with `node_outputs` and added to it."""
        resolved = []
        for item in items:
            node = item.node
            in_layouts = [layouts.get(vid) for vid in item.inputs]
            binding = [
                layouts.get(vid, o.layout)
                for vid, o in zip(
                    item.outputs, node_outputs(node, in_layouts), strict=True
                )
            ]
            kernel = self.inside(node, in_layouts, binding)
            signature = node_signature(
                node,
                frozenset(),
                in_layouts,
                declared=None if kernel is None else kernel.outputs,
            )
            check_conforms(node, signature.outputs, binding)
            for vid, lay in zip(item.outputs, binding, strict=True):
                if lay is not None:
                    layouts[vid] = lay
            resolved.append(Resolved(item, in_layouts, kernel, signature))
        return resolved

def _allocated_layout(op, var, lay) -> Layout:
    """The layout a kernel allocates `var` in to give it the layout `lay`: `lay` if it
    is packed, else C-contiguous (which must imply it)."""
    shape, size = shape_of(var), itemsize_of(var)
    if lay.fully_packed(shape):
        return lay
    if not Layout.c(var.type.ndim).implies(lay, shape, size):
        raise NotImplementedError(
            f"{op}: new memory can't have the layout {lay.describe(shape)}"
        )
    return Layout.c(var.type.ndim)


def conforms(node, outputs, binding) -> bool:
    """Do the `outputs` (`Output`s) of something running `node` have layouts that
    imply the `binding` ones?"""
    return all(
        lay is None or o.layout.implies(lay, shape_of(var), itemsize_of(var))
        for var, o, lay in zip(node.outputs, outputs, binding, strict=True)
    )


def check_conforms(node, outputs, binding):
    """`conforms`, or an error: what runs `node` was chosen to give its outputs the
    binding layouts."""
    if not conforms(node, outputs, binding):
        actual = [None if o.layout is None else str(o.layout) for o in outputs]
        wanted = [None if lay is None else str(lay) for lay in binding]
        raise AssertionError(
            f"{node.op}: outputs with layouts {actual}, not implying {wanted}"
        )
