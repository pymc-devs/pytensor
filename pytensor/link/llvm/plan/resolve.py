"""Stage 2: how each node is run.

A node is run by generated code (an emitter, `llvm_emit`), by a Rust kernel, by a numba
kernel, or, for an interpreted OpFromGraph, by a subprogram. `Resolver` decides, and
says what the choice means for the node's outputs (their modes and layouts): a
`CalledKernel` describes a kernel, `None` stands for generated code.

Tasks of a program and nodes inside generated code (a unit, a Scan's loop body) choose
differently:

* A task prefers a Rust kernel: there is nothing to compile.
* Inside generated code, generated code is preferred: it is inlined into its
  surroundings, a kernel is a call.

`resolve_items` resolves the items of a unit or a loop body in order, following the
layouts from one item to the next. Code generation consumes the result (`Resolved`)
and makes no choices of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.link.llvm.codegen.call import CalledKernel
from pytensor.link.llvm.codegen.function import (
    Output,
    OutputMode,
    Signature,
    build_node_function,
    llvm_emit,
    node_signature,
)
from pytensor.link.llvm.codegen.scan import scan_body_layouts, scan_supported
from pytensor.link.llvm.codegen.types import ir, is_absent, repr_of
from pytensor.link.llvm.layout import Layout
from pytensor.link.llvm.plan.flatten import flatten
from pytensor.link.llvm.plan.values import NodeItem, Values
from pytensor.link.llvm.rust_kernels import rust_kernel, rust_kernel_outputs
from pytensor.scalar.basic import ScalarType
from pytensor.scan.op import Scan


@dataclass
class Resolved:
    """An item of generated code with everything decided."""

    item: NodeItem
    #: the layouts of its inputs
    in_layouts: list
    #: the kernel it calls; None: its code is generated (and inlined)
    kernel: CalledKernel | None
    #: the signature of its function: representations of inputs and outputs
    signature: Signature
    #: for a generated Scan: its loop body
    body: Body | None = None


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

    def generates_scan(self, node) -> bool:
        """A Scan whose loop we generate (its body is inlined into the loop)."""
        return (
            isinstance(node.op, Scan)
            and self.opts.generate_scan
            and scan_supported(node.op)
        )

    def rust(self, node, in_layouts) -> CalledKernel | None:
        """The node's Rust kernel, if it has one."""
        k = rust_kernel(node.op, node, in_layouts)
        if k is None:
            return None
        self.stats.n_rust_kernels += 1
        return CalledKernel(
            k.address, k.config, rust_kernel_outputs(k, node, in_layouts), repr(k), k
        )

    def numba(self, node) -> CalledKernel:
        """The node's numba implementation as a kernel. It allocates its outputs, with
        any layout."""
        from pytensor.link.llvm.numba_kernels import numba_kernel
        from pytensor.tensor.random.type import RandomType

        if any(isinstance(v.type, RandomType) for v in (*node.inputs, *node.outputs)):
            raise NotImplementedError(
                f"LLVM backend: no implementation of the random variable {node.op}"
            )
        kernel = numba_kernel(node)
        self.stats.n_numba += 1
        self.stats.numba.append(str(node.op))
        outs = [
            Output(repr_of(o.type), OutputMode.ABSENT)
            if is_absent(o.type)
            else Output(
                repr_of(o.type, Layout.dynamic(o.type.ndim)), OutputMode.ALLOCATED
            )
            for o in node.outputs
        ]
        return CalledKernel(kernel.address, 0, outs, f"numba: {node.op}", kernel)

    def fallback(self, node, in_layouts) -> CalledKernel:
        """The kernel for a node we don't generate: Rust, else numba."""
        kernel = self.rust(node, in_layouts)
        if kernel is not None:
            return kernel
        if not self.opts.numba_fallback:
            raise NotImplementedError(f"No LLVM backend implementation for {node.op}")
        return self.numba(node)

    def subprogram(self, node, in_layouts) -> CalledKernel:
        """A subprogram for an OpFromGraph: its inner graph, planned on its own."""
        from pytensor.link.llvm.program import Planner

        if any(isinstance(o.type, ScalarType) for o in node.outputs):
            raise NotImplementedError("a subprogram with ScalarType outputs")
        modes = [
            OutputMode.ABSENT
            if is_absent(o.type)
            else OutputMode.PROVIDED
            if repr_of(o.type).static
            else OutputMode.ALLOCATED
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
                # an allocated output is the inner value if it owns a buffer, else a C copy
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

    def inside(self, node, in_layouts) -> tuple[CalledKernel | None, Body | None]:
        """How a node inside generated code is run: (None, body) for generated code
        (`body`: a Scan's loop body), else (its kernel, None)."""
        if (
            isinstance(node.op, OpFromGraph)
            and getattr(node.op, "execution", None) is Execution.INTERPRETED
        ):
            return self.subprogram(node, in_layouts), None
        if self.generates_scan(node):
            return None, self.scan_body(node, in_layouts)
        if self.can_generate(node, in_layouts):
            return None, None
        return self.fallback(node, in_layouts), None

    def resolve_items(self, items, layouts: dict) -> list:
        """Resolve `items` in order. `layouts` (vid -> Layout) holds the layouts of what
        they read from outside, and is filled with the layouts of what they produce."""
        resolved = []
        for item in items:
            node = item.node
            in_layouts = [layouts.get(vid) for vid in item.inputs]
            kernel, body = self.inside(node, in_layouts)
            signature = node_signature(
                node,
                frozenset(),
                in_layouts,
                declared=None if kernel is None else kernel.outputs,
            )
            for vid, o in zip(item.outputs, signature.outputs, strict=True):
                if o.layout is not None:
                    layouts[vid] = o.layout
            resolved.append(Resolved(item, in_layouts, kernel, signature, body))
        return resolved

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
