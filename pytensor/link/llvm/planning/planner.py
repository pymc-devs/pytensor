"""How every node of a graph runs: the planner's decisions.

A node runs as generated code (an emitter, `llvm_emit`), as a Rust kernel, or as a
numba kernel. This is decided here, for the program's graph and for the inner graphs
generated code runs (a Scan's loop body, a Blockwise's core, a compiled unit), and so
are the layouts of all values. Code generation then follows these decisions and makes
none of its own.

* `Values`: the values of a graph being planned, named by value ids (`ValueId`), and
  the data of its constants;
* `flatten`: a graph as a list of `FlatNode`s over value ids. OpFromGraphs are inlined,
  except compiled units outside generated code, which stay one node;
* `Planner`: how each node runs, and what that means for its outputs (how they come
  to be, and their layouts). The result is a `PlannedGraph`: the graph's `Step`s, in
  order.

The layouts of the outputs are not a choice: they are the binding ones (what follows
from `llvm_layout`, `node_outputs`), and what runs a node must give its outputs
layouts that imply them (`conforms`). A Rust kernel that doesn't is not used.

The program's graph (`Planner.program`) and graphs inside generated code
(`Planner.graph`) are planned the same way, with a different rule for each node:

* A node of the program prefers a Rust kernel: there is nothing to compile.
* Inside generated code, generated code is preferred: it is inlined into its
  surroundings, a kernel is a call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.graph.basic import Apply, Constant, Variable
from pytensor.graph.features import ViewFeature, view_feature
from pytensor.graph.traversal import io_toposort
from pytensor.link.llvm.analysis.layout import graph_layouts, layout_feature
from pytensor.link.llvm.codegen.call import Kernel
from pytensor.link.llvm.dispatch.fused import fusable
from pytensor.link.llvm.kernels.rust import (
    runtime,
    rust_implementation,
)
from pytensor.link.llvm.layout import Layout
from pytensor.link.llvm.planning.implementation import (
    GENERATED,
    Implementation,
    NumbaKernel,
    OutputDeclaration,
    OwnModule,
    RustKernel,
)
from pytensor.link.llvm.planning.memory import plan_memory
from pytensor.link.llvm.planning.plan import PlannedGraph, Step, ValueId
from pytensor.link.llvm.planning.signature import (
    known_layouts,
    node_outputs,
    node_signature,
)
from pytensor.link.llvm.registry import can_generate, has_emitter
from pytensor.link.llvm.types import ScalarRepr, TensorRepr, is_absent, repr_of
from pytensor.scalar.basic import ScalarType
from pytensor.scan.op import Scan
from pytensor.tensor.basic import TensorFromScalar
from pytensor.tensor.blockwise import OpWithCoreShape
from pytensor.tensor.rewriting.units import unboxed_output
from pytensor.tensor.type import TensorType


if TYPE_CHECKING:
    from pytensor.graph.fg import FunctionGraph
    from pytensor.graph.type import Type
    from pytensor.link.llvm.program import Options, Stats


# ------------------------------------------------------------------ values


@dataclass
class FlatNode:
    """A node of a flattened graph (`flatten`): the node, with the value ids (the
    planner's names for values) of what it reads and writes. The same PyTensor node can
    appear several times with different value ids (inlined OpFromGraphs)."""

    node: Apply
    inputs: list[ValueId]
    outputs: list[ValueId]
    #: a node of the graph being planned itself (not of an inlined inner graph): what
    #: its op may destroy was validated there (the graph's DestroyHandler)
    of_the_graph: bool = False


class Values:
    """The values of a program: a type per value id, and the data of constants."""

    #: value id -> PyTensor type
    types: dict[ValueId, Type]
    #: value id -> numpy data
    consts: dict[ValueId, np.ndarray]
    _next: int

    def __init__(self):
        self.types = {}
        self.consts = {}
        self._next = 0

    def new(self, t) -> ValueId:
        """A new value of type `t`."""
        k = ValueId(self._next)
        self._next += 1
        self.types[k] = t
        return k

    def of(self, var: Variable, scope: dict[Variable, ValueId]) -> ValueId:
        """The value of `var`: what `scope` says, or a new value for a constant."""
        if var in scope:
            return scope[var]
        if is_absent(var.type):
            # None inputs have no value; nothing ever reads this value
            k = self.new(var.type)
            scope[var] = k
            return k
        if isinstance(var, Constant):
            k = self.new(var.type)
            self.consts[k] = np.asarray(var.data)
            scope[var] = k
            return k
        raise ValueError(f"variable {var} used before it is defined")

    def folded(self, k) -> bool:
        """0-d constants are folded into compiled units."""
        return k in self.consts and self.consts[k].ndim == 0

    def constants_of(self, readers, extra=()) -> dict:
        """{value id: (type, data)} of the constants `readers` (flat nodes or steps)
        read, and of the value ids `extra`."""
        value_ids = [
            k
            for reader in readers
            for k in (reader.arguments if isinstance(reader, Step) else reader.inputs)
        ] + list(extra)
        return {
            k: (self.types[k], self.consts[k]) for k in value_ids if k in self.consts
        }


# ------------------------------------------------------------------ flattening


def flatten(
    values: Values, nodes, scope, inside_unit=False, of_the_graph=False
) -> list[FlatNode]:
    """The flat nodes running `nodes` (in order). `scope` maps variables to values and
    is filled with the nodes' outputs. OpFromGraphs are inlined, except those with
    `Execution.COMPILED`, which are nodes of their own (units, generated as one
    function) outside generated code (`inside_unit`), and a Blockwise or random
    variable with its core shapes (`OpWithCoreShape`), a loop of its own. `of_the_graph`: `nodes` are the
    graph being planned itself (marks the flat nodes, see `FlatNode`)."""
    flat_nodes: list[FlatNode] = []
    for node in nodes:
        op = node.op
        execution = (
            getattr(op, "execution", None) if isinstance(op, OpFromGraph) else None
        )
        if execution is None and fusable(op, node):
            # a FusedElemwise we can generate as one loop: keep it as one node
            ins = [values.of(i, scope) for i in node.inputs]
            outs = [values.new(var.type) for var in node.outputs]
            scope.update(zip(node.outputs, outs, strict=True))
            flat_nodes.append(FlatNode(node, ins, outs))
        elif (
            isinstance(op, OpFromGraph)
            and not isinstance(op, OpWithCoreShape)
            and (execution is None or inside_unit)
        ):
            # (inside generated code every OpFromGraph is inlined, but a Blockwise or
            # random variable with its core shapes: it is a loop of its own)
            inner = op.fgraph
            sub = {
                iv: values.of(ov, scope)
                for iv, ov in zip(inner.inputs, node.inputs, strict=True)
            }
            flat_nodes += flatten(values, inner.toposort(), sub, inside_unit)
            _inlined_outputs(values, inner, node, sub, scope)
        else:
            ins = [values.of(i, scope) for i in node.inputs]
            outs = [values.new(var.type) for var in node.outputs]
            scope.update(zip(node.outputs, outs, strict=True))
            flat_nodes.append(FlatNode(node, ins, outs, of_the_graph))
    return flat_nodes


def _inlined_outputs(values: Values, inner, node, inner_scope, scope) -> None:
    """The outputs of an inlined OpFromGraph are the values of its inner outputs. Where
    the inner graph passes an input through, or returns one variable twice, a value
    stands for several variables of the outer graph: the graph's `ViewFeature` knows."""
    for inner_output, output in zip(inner.outputs, node.outputs, strict=True):
        scope[output] = values.of(inner_output, inner_scope)


# ------------------------------------------------------------------ how nodes run


class Planner:
    """Decides how every node of a graph runs, and with which layouts.

    One planner plans one program: its graph (`program`), and the inner graphs its
    generated code runs (`graph`: a Scan's loop body, a Blockwise's core, a compiled
    unit). Both go the same way: the graph is flattened into flat nodes, and for each
    one it is decided how it runs. The rule differs:

    * the program's nodes prefer a Rust kernel (there is nothing to compile). Generated
      code for one is compiled in a module of its own (`modules`, compiled with the
      program), and the program calls it. Nodes may overwrite inputs their op destroys,
      where the graph's DestroyHandler allows it;
    * inside generated code, generated code is preferred: it is inlined, while a kernel
      is a call."""

    opts: Options
    stats: Stats
    #: the values of the graphs planned, by value id
    values: Values
    #: the program's graph (`program`), and what views of its values there may be
    fgraph: FunctionGraph
    views: ViewFeature
    #: (inputs, outputs, layouts) -> PlannedGraph (`graph`)
    _graphs: dict[tuple, PlannedGraph]
    #: the memory views the DestroyHandler doesn't see may lie in (never overwritten
    #: in place)
    hidden: frozenset
    #: may nodes of the program overwrite inputs? Only for a graph whose
    #: DestroyHandler has validated the order of its nodes
    allow_inplace: bool

    def __init__(self, options: Options, stats: Stats) -> None:
        self.opts = options
        self.stats = stats
        self.values = Values()
        self._graphs = {}
        self.hidden = frozenset()
        self.allow_inplace = False

    # ------------------------------------------------------------ graphs

    def program(self, fgraph) -> PlannedGraph:
        """Decide how every node of the program's graph `fgraph` runs.

        The layouts of the graph's variables are what its `LayoutFeature` says. A scalar
        input arrives as a 0-d array (scalars don't cross the kernel ABI): its value id
        in `PlannedGraph.in_ids` is that array's. A scalar output (`ScalarFromTensor`
        of a 0-d array, from the units rewrite) leaves as that array."""
        values = self.values
        layout_feature(fgraph).set_input_layout(self.opts.input_layout)
        scope: dict = {}
        in_ids = []
        boxed = {}  # scalar input -> the 0-d array it arrives in
        for v in fgraph.inputs:
            if isinstance(v.type, ScalarType):
                # (the units rewrite reads scalar inputs through `TensorFromScalar` of
                # them: that is the 0-d array)
                k = values.new(TensorType(v.type.dtype, shape=()))
                boxed[v] = k
                scope[v] = values.new(v.type)
            else:
                k = scope[v] = values.new(v.type)
            in_ids.append(k)
        nodes = []
        unboxed = {}  # scalar output -> the 0-d array returned for it
        for node in fgraph.toposort():
            if isinstance(node.op, TensorFromScalar) and node.inputs[0] in boxed:
                scope[node.outputs[0]] = boxed[node.inputs[0]]
            elif unboxed_output(fgraph, node.outputs[0]):
                unboxed[node.outputs[0]] = node.inputs[0]
            else:
                nodes.append(node)
        # (the order of a graph with a DestroyHandler respects what its ops destroy)
        self.allow_inplace = hasattr(fgraph, "destroyers")
        self.fgraph, self.views = fgraph, view_feature(fgraph)
        if self.allow_inplace:
            self.hidden = self.views.undeclared_view_roots(fgraph)
        flat_nodes = flatten(values, nodes, scope, of_the_graph=True)
        out_ids = [values.of(unboxed.get(var, var), scope) for var in fgraph.outputs]
        layouts: dict = {k: Layout.c(0) for k in boxed.values()}
        for v, k in scope.items():
            layout = fgraph.llvm_layout(v)
            if layout is not None and k not in values.consts:
                layouts.setdefault(k, layout)
        steps = self._decide(flat_nodes, layouts, top_level=True)
        constants = values.constants_of(steps, out_ids)
        memory = plan_memory(steps, in_ids, out_ids, constants.keys())
        return PlannedGraph(in_ids, out_ids, steps, constants, layouts, memory)

    def graph(self, inputs, outputs, in_layouts) -> PlannedGraph:
        """Decide how every node of a graph runs, when the graph runs inside generated
        code.

        The graph goes from the variables `inputs` to the variables `outputs`.
        `in_layouts` gives the layout of each input (None for inputs that aren't
        tensors). Inputs that are constants stay constants of the graph.

        The layouts of the graph's variables come from `graph_layouts`, with
        OpFromGraphs inlined (that is what generated code does with them).

        The result is cached: the same graph with the same input layouts is planned
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
        nodes = io_toposort(inputs, outputs)
        flat_nodes = flatten(values, nodes, scope, inside_unit=True)
        out_ids = [values.of(var, scope) for var in outputs]
        known = graph_layouts(inputs, outputs, in_layouts, inlined=True)
        layouts: dict = {}
        for v, k in scope.items():
            if known.get(v) is not None and k not in values.consts:
                layouts.setdefault(k, known[v])
        steps = self._decide(flat_nodes, layouts, top_level=False)
        constants = values.constants_of(flat_nodes, out_ids)
        memory = plan_memory(steps, in_ids, out_ids, constants.keys())
        body = PlannedGraph(in_ids, out_ids, steps, constants, layouts, memory)
        self._graphs[key] = body
        return body

    # ------------------------------------------------------------ layouts of values

    def layout(self, k, layouts):
        """The layout of value `k` (None for values that aren't tensors): a constant's
        own memory order, else what `layouts` (value id -> Layout) says."""
        t = self.values.types[k]
        if not hasattr(t, "ndim") or is_absent(t):
            return None
        if k in self.values.consts:
            return Layout.of_const(self.values.consts[k])
        return layouts.get(k)

    def _binding(self, node, value_ids, in_layouts, layouts) -> list:
        """The layouts the values `value_ids` must have: the outputs of a flat node
        running `node` with inputs in `in_layouts`. Those not in `layouts` yet are what
        `node_outputs` gives them, and are added to it."""
        for k, output in zip(value_ids, node_outputs(node, in_layouts), strict=True):
            if output.layout is not None:
                layouts.setdefault(k, output.layout)
        return [self.layout(k, layouts) for k in value_ids]

    # ------------------------------------------------------------ deciding, per node

    def _decide(self, flat_nodes, layouts, top_level) -> list[Step]:
        """Decide how each of `flat_nodes` runs, in order (`top_level`: the program's
        rule, else the rule inside generated code). `layouts` must hold the layouts of
        the values they read from outside; it is filled with those of what they
        produce."""
        return [
            self._program_step(flat, layouts)
            if top_level
            else self._inner_step(flat, layouts)
            for flat in flat_nodes
        ]

    def _inner_step(self, flat, layouts) -> Step:
        """How a flat node inside generated code runs: generated and inlined if
        possible, else a call of its kernel."""
        node = flat.node
        in_layouts = [self.layout(k, layouts) for k in flat.inputs]
        binding = self._binding(node, flat.outputs, in_layouts, layouts)
        implementation = self._inside(node, in_layouts, binding)
        signature = node_signature(node, in_layouts, implementation=implementation)
        check_conforms(node, signature.outputs, binding)
        description = (
            str(node.op) if implementation is GENERATED else implementation.description
        )
        return Step(
            node,
            flat.inputs,
            flat.outputs,
            in_layouts,
            implementation,
            signature,
            description,
        )

    def _inside(self, node, in_layouts, binding) -> Implementation:
        """How a node inside generated code runs: generated (and inlined) if possible,
        else a call of its kernel."""
        if isinstance(node.op, Scan):
            return GENERATED
        if can_generate(node):
            return GENERATED
        return self._kernel(node, in_layouts, binding)

    def _program_step(self, flat, layouts) -> Step:
        """How a flat node of the program runs: a Rust kernel if it has one, else
        generated code compiled on its own, else a numba kernel."""
        node = flat.node
        in_layouts = [self.layout(k, layouts) for k in flat.inputs]
        if _is_unit(node):
            # (if it can't be generated, its nodes run on their own: `_decide`)
            return self._generated_step(flat, {}, in_layouts, layouts)
        binding = self._binding(node, flat.outputs, in_layouts, layouts)
        types = self.values.types
        # constant scalar inputs are folded into the code
        folded = {
            pos: self.values.consts[k].item()
            for pos, (k, var) in enumerate(zip(flat.inputs, node.inputs, strict=True))
            if k in self.values.consts and isinstance(var.type, ScalarType)
        }
        if any(
            isinstance(repr_of(var.type), ScalarRepr) for var in node.outputs
        ) or any(
            isinstance(repr_of(types[k]), ScalarRepr)
            for pos, k in enumerate(flat.inputs)
            if pos not in folded
        ):
            raise NotImplementedError("scalar node outside a unit")
        # A Rust kernel first: there is nothing to compile. Not for a Scan or Blockwise
        # (its inner graph is inlined into the generated loop), and not with folded
        # constants (a kernel would need them as arguments).
        nested = isinstance(node.op, Scan | OpWithCoreShape)
        writable = self._writable_inputs(flat, folded)
        if not folded and not nested and not writable:
            rust = self._rust(node, in_layouts, binding)
            if rust is not None:
                return self._kernel_step(flat, rust, layouts)
        if can_generate(node):
            return self._generated_step(flat, folded, in_layouts, layouts, writable)
        # (a kernel doesn't overwrite its inputs: it can stand in for an in-place node
        # too)
        if nested or writable:
            rust = self._rust(node, in_layouts, binding)
            if rust is not None:
                return self._kernel_step(flat, rust, layouts)
        if not self.opts.numba_fallback:
            raise _no_implementation(node)
        numba = self._numba(node, binding)
        # it overwrites the inputs its op destroys: that is what the graph asks for,
        # where nothing else may see the memory (as in `_writable_inputs`). Else it gets
        # copies.
        writable = frozenset(
            i for i in numba.always_overwrites if self._may_destroy(flat, i)
        )
        return self._kernel_step(flat, numba, layouts, writable)

    def _kernel_step(
        self, flat, implementation: Implementation, layouts, writable=frozenset()
    ) -> Step:
        """A step of the program run by a kernel (Rust, numba)."""
        in_layouts = [self.layout(k, layouts) for k in flat.inputs]
        signature = node_signature(
            flat.node, in_layouts, implementation=implementation, writable=writable
        )
        binding = self._binding(flat.node, flat.outputs, in_layouts, layouts)
        check_conforms(flat.node, signature.outputs, binding)
        return Step(
            flat.node,
            flat.inputs,
            flat.outputs,
            in_layouts,
            implementation,
            signature,
            implementation.description,
        )

    def _generated_step(
        self, flat, folded, in_layouts, layouts, writable=frozenset()
    ) -> Step:
        """A step of the program whose code is generated and compiled in a module of
        its own, which codegen builds and links with the program. `folded`: the constant
        scalar inputs compiled into the code; `writable`: the inputs it may overwrite
        (`_writable_inputs`)."""
        node = flat.node
        signature = node_signature(node, in_layouts, folded=folded, writable=writable)
        # An output overwrites a writable input only where its layout allows that.
        if any(writable & set(o.aliased_inputs) for o in signature.outputs):
            self.stats.inplace.append(str(node.op))
        if _is_unit(node):
            self.stats.n_units += 1
            nodes = node.op.fgraph.toposort()
            names = ", ".join(str(n.op) for n in nodes)
            desc = f"unit[{len(nodes)}]: {names[:120]}"
        else:
            desc = str(node.op) + _layouts_note(
                [
                    representation
                    for representation in signature.inputs
                    if isinstance(representation, TensorRepr)
                    and representation.ndim > 0
                ]
            )
        implementation = OwnModule(
            tuple(GENERATED.outputs(node, known_layouts(node, in_layouts))), desc
        )
        binding = self._binding(node, flat.outputs, in_layouts, layouts)
        check_conforms(node, signature.outputs, binding)
        return Step(
            node,
            flat.inputs,
            flat.outputs,
            in_layouts,
            implementation,
            signature,
            desc,
        )

    # ------------------------------------------------------------ in place

    def _roots_of(self, var) -> frozenset:
        """The variables of the program's graph whose memory `var` may lie in."""
        return self.views.view_roots(self.fgraph, var)

    def _may_destroy(self, flat, i) -> bool:
        """May the flat node, a node of the program's graph, overwrite its input `i`?
        Its op destroys it, so the DestroyHandler says no one else needs it; unless the
        memory is a constant's, or views it doesn't see may lie in it."""
        return (
            self.allow_inplace
            and flat.of_the_graph
            and flat.inputs[i] not in self.values.consts
            and not self._roots_of(flat.node.inputs[i]) & self.hidden
        )

    def _writable_inputs(self, flat, folded) -> frozenset:
        """The inputs a node's generated code may overwrite: tensors its op destroys,
        that nothing else can see, and whose memory isn't another input's. Its emitter
        decides whether it does (codegen/interface.py)."""
        node = flat.node
        writable = set()
        for i in {i for ins in node.op.destroy_map.values() for i in ins}:
            if i in folded or not self._may_destroy(flat, i):
                continue
            mine = self._roots_of(node.inputs[i])
            if not any(
                mine & self._roots_of(var)
                for pos, var in enumerate(node.inputs)
                if pos != i and hasattr(var.type, "ndim")
            ):
                writable.add(i)
        return frozenset(writable)

    # ------------------------------------------------------------ kernels

    def _rust(self, node, in_layouts, binding) -> RustKernel | None:
        """The node's Rust kernel, if it has one that gives its outputs the `binding`
        layouts."""
        implementation = rust_implementation(
            node, in_layouts, self.opts.linalg_parallel
        )
        if implementation is None:
            return None
        if not conforms(
            node, node_outputs(node, in_layouts, (), implementation), binding
        ):
            return None
        self.stats.n_rust_kernels += 1
        return implementation

    def _numba(self, node, binding) -> NumbaKernel:
        """The node's numba implementation as a kernel. It allocates its outputs, packed
        in the `binding` layouts (C-contiguous where they aren't packed: they must be
        implied by it)."""
        from pytensor.link.llvm.kernels.numba import numba_kernel

        layouts = [
            None if layout is None else _allocated_layout(node.op, var, layout)
            for var, layout in zip(node.outputs, binding, strict=True)
        ]
        orders = [
            None
            if layout is None or layout.is_c(var.type.shape)
            else tuple(layout.order)
            for var, layout in zip(node.outputs, layouts, strict=True)
        ]
        kernel = numba_kernel(node, None if not any(orders) else orders)
        self.stats.n_numba += 1
        self.stats.numba.append(str(node.op))
        destroyed = getattr(node.op, "destroy_map", None) or {}
        return NumbaKernel(
            Kernel(
                runtime().numba_entry(), kernel.address, f"numba: {node.op}", kernel
            ),
            tuple(OutputDeclaration(layout, False) for layout in layouts),
            destroys=frozenset(i for ins in destroyed.values() for i in ins),
        )

    def _kernel(self, node, in_layouts, binding) -> Implementation:
        """The kernel for a node we don't generate: Rust, else numba."""
        rust = self._rust(node, in_layouts, binding)
        if rust is not None:
            return rust
        if not self.opts.numba_fallback:
            raise _no_implementation(node)
        return self._numba(node, binding)


def _no_implementation(node) -> NotImplementedError:
    """The error for a node nothing can run (no Rust kernel, the numba fallback off)."""
    if has_emitter(node.op):
        why = "its emitter doesn't handle this node (`llvm_supports`)"
    else:
        why = "it has no emitter"
    return NotImplementedError(
        f"No LLVM backend implementation for {node.op}: {why}, it has no Rust "
        "kernel, and the numba fallback is off"
    )


def _is_unit(node) -> bool:
    """Is the node a compiled unit, an OpFromGraph with `execution=COMPILED`? Its inner
    graph is generated as one function."""
    return isinstance(node.op, OpFromGraph) and node.op.execution is Execution.COMPILED


def _layouts_note(reprs):
    """` {C x 2, [?, P0]}`: the input layouts a generated step is specialized for, in
    the order of the inputs. A layout that repeats is written once, with a count."""

    def unknown(representation):
        n = sum(1 for s in representation.shape if s != 1)
        return n > 0 and len(representation.stride_axes) == n

    if not reprs or all(unknown(representation) for representation in reprs):
        return ""
    layouts = [
        "?"
        if unknown(representation)
        else representation.layout.describe(representation.shape)
        for representation in reprs
    ]
    runs: list[list] = []
    for layout in layouts:
        if runs and runs[-1][0] == layout:
            runs[-1][1] += 1
        else:
            runs.append([layout, 1])
    parts = [layout if n == 1 else f"{layout} x {n}" for layout, n in runs]
    return " {" + ", ".join(parts) + "}"


def _allocated_layout(op, var, layout: Layout) -> Layout:
    """The layout a kernel allocates `var` in to give it the layout `lay`: `lay` if it
    is packed, else C-contiguous (which must imply it)."""
    shape, size = var.type.shape, var.type.numpy_dtype.itemsize
    if layout.fully_packed(shape):
        return layout
    if not Layout.c(var.type.ndim).implies(layout, shape, size):
        raise NotImplementedError(
            f"{op}: new memory can't have the layout {layout.describe(shape)}"
        )
    return Layout.c(var.type.ndim)


def conforms(node, outputs, binding) -> bool:
    """Do the `outputs` (`Output`s) of something running `node` have layouts that
    imply the `binding` ones?"""
    return all(
        layout is None
        or output.layout.implies(layout, var.type.shape, var.type.numpy_dtype.itemsize)
        for var, output, layout in zip(node.outputs, outputs, binding, strict=True)
    )


def check_conforms(node, outputs, binding):
    """`conforms`, or an error: what runs `node` was chosen to give its outputs the
    binding layouts."""
    if not conforms(node, outputs, binding):
        actual = [
            None if output.layout is None else str(output.layout) for output in outputs
        ]
        wanted = [None if layout is None else str(layout) for layout in binding]
        raise AssertionError(
            f"{node.op}: outputs with layouts {actual}, not implying {wanted}"
        )
