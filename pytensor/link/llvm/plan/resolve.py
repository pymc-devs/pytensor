"""Stage 2: how each node is run.

A node is run by generated code (an emitter, `llvm_emit`), by a Rust kernel, by a numba
kernel, or, for an OpFromGraph with `execution=INTERPRETED`, by a subprogram: its inner
graph planned and compiled on its own. `Resolver` decides, and
says what the choice means for the node's outputs (how they come to be, and their layouts): a
`CalledKernel` describes a kernel, `None` stands for generated code.

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

import numpy as np

from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.graph.basic import Constant
from pytensor.graph.traversal import io_toposort
from pytensor.link.llvm.codegen.blockwise import (
    blockwise_core,
    blockwise_core_layouts,
)
from pytensor.link.llvm.codegen.call import CalledKernel
from pytensor.link.llvm.codegen.function import (
    Output,
    OutputMode,
    Signature,
    build_node_function,
    llvm_emit,
    node_outputs,
    node_signature,
)
from pytensor.link.llvm.codegen.scan import scan_body_layouts
from pytensor.link.llvm.codegen.types import ir, is_absent, repr_of
from pytensor.link.llvm.layout import Layout, itemsize_of, shape_of
from pytensor.link.llvm.plan.flatten import flatten
from pytensor.link.llvm.plan.values import NodeItem, Values
from pytensor.link.llvm.rust_kernels import interp, rust_kernel, rust_kernel_outputs
from pytensor.scalar.basic import ScalarType
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
    #: for a generated Scan: its loop body
    body: Body | None = None

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
    """The inner graph of a Scan, resolved: what one iteration runs."""

    #: the values of the inner graph's inputs and outputs
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

    # ------------------------------------------------------------ the kinds of kernels

    def has_body(self, node) -> bool:
        """A node whose generated code runs a resolved inner graph (`Body`): a Scan
        (its loop body, inlined into the loop), or a Blockwise (its core node)."""
        return isinstance(node.op, Scan | Blockwise)

    def body(self, node, in_layouts) -> Body:
        """The resolved inner graph of a node that `has_body`."""
        if isinstance(node.op, Blockwise):
            return self.blockwise_body(node, in_layouts)
        return self.scan_body(node, in_layouts)

    def rust(self, node, in_layouts, binding) -> CalledKernel | None:
        """The node's Rust kernel, if it has one that gives its outputs the `binding`
        layouts."""
        k = rust_kernel(node.op, node, in_layouts)
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

    def subprogram(self, node, in_layouts) -> CalledKernel:
        """A subprogram for an OpFromGraph: its inner graph, planned and compiled on its
        own (a `Program`, whose function is the kernel)."""
        from pytensor.link.llvm.program import Planner

        if any(isinstance(o.type, ScalarType) for o in node.outputs):
            raise NotImplementedError("a subprogram with ScalarType outputs")
        modes = [
            OutputMode.ABSENT
            if is_absent(o.type)
            else OutputMode.PROVIDED
            if repr_of(o.type).static
            else OutputMode.RETURNED
            for o in node.outputs
        ]
        sub = Planner(self.opts)
        prog = sub.plan(node.op.fgraph, output_modes=modes, input_layouts=in_layouts)
        self.stats.n_subprograms += 1
        outs = []
        for o, mode, lay in zip(node.outputs, modes, sub.output_layouts, strict=True):
            if mode is OutputMode.ABSENT:
                outs.append(Output(repr_of(o.type), mode))
            elif mode is OutputMode.PROVIDED or lay is None:
                outs.append(Output(repr_of(o.type, Layout.c(o.type.ndim)), mode))
            else:
                # a returned output is the inner value if it owns a buffer, else a C copy
                shape = tuple(None if s is None else int(s) for s in o.type.shape)
                lay = lay.join(
                    Layout.c(o.type.ndim), shape, np.dtype(o.type.dtype).itemsize
                )
                outs.append(Output(repr_of(o.type, lay), mode))
        return CalledKernel(
            prog.address, prog.config, outs, f"program: {node.op}", prog
        )

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

    def inside(
        self, node, in_layouts, binding
    ) -> tuple[CalledKernel | None, Body | None]:
        """How a node inside generated code is run: (None, body) for generated code
        (`body`: a Scan's loop body), else (its kernel, None)."""
        if (
            isinstance(node.op, OpFromGraph)
            and getattr(node.op, "execution", None) is Execution.INTERPRETED
        ):
            return self.subprogram(node, in_layouts), None
        if isinstance(node.op, Scan):
            return None, self.scan_body(node, in_layouts)
        if isinstance(node.op, Blockwise):
            try:
                return None, self.blockwise_body(node, in_layouts)
            except NotImplementedError:
                return self.fallback(node, in_layouts, binding), None
        if self.can_generate(node, in_layouts):
            return None, None
        return self.fallback(node, in_layouts, binding), None

    def resolve_items(self, items, layouts: dict) -> list:
        """Resolve `items` in order. `layouts` (vid -> Layout) holds the layouts of what
        they read from outside, and is filled with the (binding) layouts of what they
        produce."""
        resolved = []
        for item in items:
            node = item.node
            in_layouts = [layouts.get(vid) for vid in item.inputs]
            binding = [o.layout for o in node_outputs(node, in_layouts)]
            kernel, body = self.inside(node, in_layouts, binding)
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
            resolved.append(Resolved(item, in_layouts, kernel, signature, body))
        return resolved

    def blockwise_body(self, node, in_layouts) -> Body:
        """The core of a Blockwise: its core graph, resolved for the layouts its inputs
        have in the generated loop (the inputs at one batch index)."""
        values = self.values
        core_inputs, core_outputs = blockwise_core(node)
        scope = {}
        # (constant core inputs are the Blockwise's constants: values of the body)
        in_ids = [
            values.of(v, scope) if isinstance(v, Constant) else values.new(v.type)
            for v in core_inputs
        ]
        scope.update(zip(core_inputs, in_ids, strict=True))
        nodes = io_toposort(core_inputs, core_outputs)
        items = flatten(values, nodes, scope, inside_unit=True)
        out_ids = [values.of(o, scope) for o in core_outputs]
        layouts = {
            k: lay
            for k, lay in zip(
                in_ids, blockwise_core_layouts(node, in_layouts), strict=True
            )
            if lay is not None
        }
        resolved = self.resolve_items(items, layouts)
        return Body(in_ids, out_ids, resolved, values.constants_of(items, out_ids))

    def scan_body(self, node, in_layouts) -> Body:
        """The loop body of a generated Scan: the inner graph's items, resolved for the
        layouts its inputs have in the loop (rows of the sequences and buffers)."""
        values = self.values
        fgraph = node.op.fgraph
        scope = {}
        in_ids = []
        for v in fgraph.inputs:
            k = values.new(v.type)
            scope[v] = k
            in_ids.append(k)
        items = flatten(values, fgraph.toposort(), scope, inside_unit=True)
        out_ids = [values.of(o, scope) for o in fgraph.outputs]
        layouts = {
            k: lay
            for k, lay in zip(
                in_ids, scan_body_layouts(node.op, node, in_layouts), strict=True
            )
            if lay is not None
        }
        resolved = self.resolve_items(items, layouts)
        return Body(in_ids, out_ids, resolved, values.constants_of(items, out_ids))


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
