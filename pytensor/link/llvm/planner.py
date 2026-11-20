"""How every node of a graph runs: the planner's decisions.

A node runs as generated code (an emitter, `llvm_emit`), as a Rust kernel, or as a
numba kernel. This is decided here, for the program's graph and for the inner graphs
generated code runs (a Scan's loop body, a Blockwise's core, a compiled unit), and so
are the layouts of all values. Code generation then follows these decisions and makes
none of its own.

* `Values`: the values of a graph being planned, named by value ids (vids), and the
  data of its constants;
* `flatten`: a graph as a list of `FlatNode`s over vids. OpFromGraphs are inlined,
  except compiled units outside generated code, which stay one node;
* `Planner`: how each node runs, and what that means for its outputs (how they come
  to be, and their layouts). The result is a `PlannedGraph`: the graph's `Step`s, in
  order.

The layouts of the outputs are not a choice: they are the binding ones (what
`llvm_layout` declares, `node_outputs`), and what runs a node must give its outputs
layouts that imply them (`conforms`). A Rust kernel that doesn't is not used.

The program's graph (`Planner.program`) and graphs inside generated code
(`Planner.graph`) are planned the same way, with a different rule for each node:

* A node of the program prefers a Rust kernel: there is nothing to compile.
* Inside generated code, generated code is preferred: it is inlined into its
  surroundings, a kernel is a call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pytensor.compile.builders import Execution, OpFromGraph
from pytensor.graph.basic import Apply, Constant
from pytensor.graph.features import view_feature
from pytensor.graph.traversal import io_toposort
from pytensor.link.llvm.analysis.layout import (
    Layout,
    graph_layouts,
    itemsize_of,
    layout_feature,
    shape_of,
)
from pytensor.link.llvm.codegen.call import Kernel, LinkTable
from pytensor.link.llvm.codegen.interface import llvm_emit
from pytensor.link.llvm.codegen.kernel import build_kernel_wrapper
from pytensor.link.llvm.codegen.node import build_node_function
from pytensor.link.llvm.codegen.signature import (
    Output,
    OutputMode,
    Signature,
    node_outputs,
    node_signature,
    skipped_inputs,
)
from pytensor.link.llvm.codegen.types import ScalarRepr, ir, is_absent, repr_of
from pytensor.link.llvm.dispatch.blockwise import (
    blockwise_core,
    blockwise_core_layouts,
)
from pytensor.link.llvm.kernels.rust import interp, rust_kernel, rust_kernel_outputs
from pytensor.scalar.basic import ScalarType
from pytensor.scan.op import Scan
from pytensor.tensor.basic import TensorFromScalar
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.type import TensorType


# ------------------------------------------------------------------ values


@dataclass
class FlatNode:
    """A node of a flattened graph (`flatten`): the node, with the value ids (vids: the
    planner's names for values) of what it reads and writes. The same PyTensor node can
    appear several times with different vids (inlined OpFromGraphs)."""

    node: Apply
    inputs: list[int]
    outputs: list[int]
    #: a node of the graph being planned itself (not of an inlined inner graph): what
    #: its op may destroy was validated there (the graph's DestroyHandler)
    of_the_graph: bool = False


class Values:
    """The values of a program: a type per value id, and the data of constants."""

    def __init__(self):
        self.types = {}  # vid -> PyTensor type
        self.consts = {}  # vid -> numpy data
        self._next = 0

    def new(self, t) -> int:
        """A new value of type `t`."""
        k = self._next
        self._next += 1
        self.types[k] = t
        return k

    def of(self, var, scope) -> int:
        """The value of `var`: what `scope` says, or a new value for a constant."""
        if var in scope:
            return scope[var]
        if is_absent(var.type):
            # None inputs have no value; nothing ever reads this vid
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
        """{vid: (type, data)} of the constants `readers` (flat nodes or steps) read,
        and of the vids `extra`."""
        vids = [k for r in readers for k in r.inputs] + list(extra)
        return {k: (self.types[k], self.consts[k]) for k in vids if k in self.consts}


# ------------------------------------------------------------------ flattening


def flatten(
    values: Values, nodes, scope, inside_unit=False, of_the_graph=False
) -> list[FlatNode]:
    """The flat nodes running `nodes` (in order). `scope` maps variables to values and
    is filled with the nodes' outputs. OpFromGraphs are inlined, except those with
    `Execution.COMPILED`, which are nodes of their own (units, generated as one
    function) outside generated code (`inside_unit`). `of_the_graph`: `nodes` are the
    graph being planned itself (marks the flat nodes, see `FlatNode`)."""
    flat_nodes: list[FlatNode] = []
    for node in nodes:
        op = node.op
        execution = (
            getattr(op, "execution", None) if isinstance(op, OpFromGraph) else None
        )
        if isinstance(op, OpFromGraph) and execution is None and fusable(op, node):
            # a FusedElemwise we can generate as one loop: keep it as one node
            ins = [values.of(i, scope) for i in node.inputs]
            outs = [values.new(o.type) for o in node.outputs]
            scope.update(zip(node.outputs, outs, strict=True))
            flat_nodes.append(FlatNode(node, ins, outs))
        elif isinstance(op, OpFromGraph) and (execution is None or inside_unit):
            # (inside generated code every OpFromGraph is inlined)
            inner = op.fgraph
            sub = {
                iv: values.of(ov, scope)
                for iv, ov in zip(inner.inputs, node.inputs, strict=True)
            }
            flat_nodes += flatten(values, inner.toposort(), sub, inside_unit)
            _inlined_outputs(values, inner, node, sub, scope)
        else:
            ins = [values.of(i, scope) for i in node.inputs]
            outs = [values.new(o.type) for o in node.outputs]
            scope.update(zip(node.outputs, outs, strict=True))
            flat_nodes.append(FlatNode(node, ins, outs, of_the_graph))
    return flat_nodes


def _inlined_outputs(values: Values, inner, node, inner_scope, scope) -> None:
    """The outputs of an inlined OpFromGraph are the values of its inner outputs. Where
    the inner graph passes an input through, or returns one variable twice, a value
    stands for several variables of the outer graph: the graph's `ViewFeature` knows."""
    for inner_output, output in zip(inner.outputs, node.outputs, strict=True):
        scope[output] = values.of(inner_output, inner_scope)


def fusable(op, node) -> bool:
    """A FusedElemwise we can generate as one loop?"""
    from pytensor.link.llvm.dispatch.fused import fused_spec
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    if not isinstance(op, FusedElemwise):
        return False
    try:
        fused_spec(op, node)
    except NotImplementedError:
        return False
    return True


# ------------------------------------------------------------------ how nodes run


@dataclass
class Step:
    """One thing a planned graph runs, in order, with everything decided about how."""

    #: the node it runs; None for code compiled on its own, which it calls with
    #: `inputs` (a node or a unit of the program)
    node: Apply | None
    #: the values (vids) it reads: one per input of the node, or per argument of the
    #: code compiled on its own
    inputs: list
    #: the values it writes, one per output
    outputs: list
    #: the layouts of `inputs`
    in_layouts: list
    #: the kernel it calls; None: its code is generated (and inlined)
    kernel: Kernel | None
    #: the signature of its function: representations of inputs and outputs
    signature: Signature
    #: what it runs, for dumps and profiles
    description: str

    @property
    def arguments(self) -> list:
        """The values passed, one per input of the signature (inputs without data
        aren't passed)."""
        if self.node is None:
            return self.inputs
        return [
            vid
            for vid, var in zip(self.inputs, self.node.inputs, strict=True)
            if not is_absent(var.type)
        ]


@dataclass
class PlannedGraph:
    """A graph with every decision made about how its nodes run (`Planner.graph`,
    `Planner.program`): a Scan's loop body, a Blockwise's core, a unit's inner graph,
    or the program.

    Its steps are in the order they run, and refer to values by their vids."""

    #: the values (vids) of the graph's inputs and outputs
    in_ids: list
    out_ids: list
    steps: list[Step]
    #: {vid: (type, data)} of the constants the steps read
    constants: dict
    #: the layouts of its values (vid -> Layout; constants: `Planner.layout`)
    layouts: dict


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

    def __init__(self, options, stats):
        self.opts = options
        self.stats = stats
        self.values = Values()
        #: modules generated for nodes of the program, to compile with the program's
        #: function: (module, the `Kernel` whose address is its "kernel")
        self.modules: list = []
        #: (inputs, outputs, layouts) -> PlannedGraph (`graph`)
        self._graphs: dict[tuple, PlannedGraph] = {}
        #: the program's graph (`program`), what views of its values there may be, and
        #: the memory views the DestroyHandler doesn't see may lie in (never
        #: overwritten in place)
        self.fgraph = None
        self.views = None
        self.hidden: frozenset = frozenset()
        #: may nodes of the program overwrite inputs? Only for a graph whose
        #: DestroyHandler has validated the order of its nodes
        self.allow_inplace = False

    # ------------------------------------------------------------ graphs

    def program(self, fgraph) -> PlannedGraph:
        """Decide how every node of the program's graph `fgraph` runs.

        The layouts of the graph's variables are what its `LayoutFeature` says. A scalar
        input arrives as a 0-d array (scalars don't cross the kernel ABI): its vid in
        `PlannedGraph.in_ids` is that array's."""
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
        for node in fgraph.toposort():
            if isinstance(node.op, TensorFromScalar) and node.inputs[0] in boxed:
                scope[node.outputs[0]] = boxed[node.inputs[0]]
            else:
                nodes.append(node)
        # (the order of a graph with a DestroyHandler respects what its ops destroy)
        self.allow_inplace = hasattr(fgraph, "destroyers")
        self.fgraph, self.views = fgraph, view_feature(fgraph)
        if self.allow_inplace:
            self.hidden = self.views.undeclared_view_roots(fgraph)
        flat_nodes = flatten(values, nodes, scope, of_the_graph=True)
        out_ids = [values.of(o, scope) for o in fgraph.outputs]
        layouts: dict = {k: Layout.c(0) for k in boxed.values()}
        for v, k in scope.items():
            lay = fgraph.llvm_layout(v)
            if lay is not None and k not in values.consts:
                layouts.setdefault(k, lay)
        steps = self._decide(flat_nodes, layouts, top_level=True)
        constants = values.constants_of(steps, out_ids)
        return PlannedGraph(in_ids, out_ids, steps, constants, layouts)

    def graph(self, inputs, outputs, in_layouts) -> PlannedGraph:
        """Decide how every node of a graph runs, when the graph runs inside generated
        code.

        The graph goes from the variables `inputs` to the variables `outputs`.
        `in_layouts` gives the layout of each input (None for inputs that aren't
        tensors). Inputs that are constants stay constants of the graph.

        The layouts of the graph's variables come from `graph_layouts`, with every
        OpFromGraph inlined (that is what generated code does with them).

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
        out_ids = [values.of(o, scope) for o in outputs]
        known = graph_layouts(inputs, outputs, in_layouts, inlined=True)
        layouts: dict = {}
        for v, k in scope.items():
            if known.get(v) is not None and k not in values.consts:
                layouts.setdefault(k, known[v])
        steps = self._decide(flat_nodes, layouts, top_level=False)
        body = PlannedGraph(
            in_ids, out_ids, steps, values.constants_of(flat_nodes, out_ids), layouts
        )
        self._graphs[key] = body
        return body

    # ------------------------------------------------------------ layouts of values

    def layout(self, k, layouts):
        """The layout of value `k` (None for values that aren't tensors): a constant's
        own memory order, else what `layouts` (vid -> Layout) says."""
        t = self.values.types[k]
        if not hasattr(t, "ndim") or is_absent(t):
            return None
        if k in self.values.consts:
            return Layout.of_const(self.values.consts[k])
        return layouts.get(k)

    def _binding(self, node, vids, in_layouts, layouts) -> list:
        """The layouts the values `vids` must have: the outputs of a flat node running
        `node` with inputs in `in_layouts`. Those not in `layouts` yet are what
        `node_outputs` gives them, and are added to it."""
        for k, o in zip(vids, node_outputs(node, in_layouts), strict=True):
            if o.layout is not None:
                layouts.setdefault(k, o.layout)
        return [self.layout(k, layouts) for k in vids]

    # ------------------------------------------------------------ deciding, per node

    def _decide(self, flat_nodes, layouts, top_level) -> list[Step]:
        """Decide how each of `flat_nodes` runs, in order (`top_level`: the program's
        rule, else the rule inside generated code). `layouts` must hold the layouts of
        the values they read from outside; it is filled with those of what they
        produce."""
        steps: list[Step] = []
        for flat in flat_nodes:
            try:
                steps.append(
                    self._program_step(flat, layouts)
                    if top_level
                    else self._inner_step(flat, layouts)
                )
            except NotImplementedError:
                if not (top_level and _is_unit(flat.node)):
                    raise
                # a compiled OpFromGraph we can't generate: its nodes on their own,
                # and a copy of each result into the value the OpFromGraph's output is
                inner = flat.node.op.fgraph
                sub = dict(zip(inner.inputs, flat.inputs, strict=True))
                inner_flat = flatten(
                    self.values, inner.toposort(), sub, inside_unit=True
                )
                steps += self._decide(inner_flat, layouts, top_level)
                for io, outer in zip(inner.outputs, flat.outputs, strict=True):
                    src = self.values.of(io, sub)
                    steps.append(self._copy_step(src, outer, layouts))
        return steps

    def _inner_step(self, flat, layouts) -> Step:
        """How a flat node inside generated code runs: generated and inlined if
        possible, else a call of its kernel."""
        node = flat.node
        in_layouts = [self.layout(k, layouts) for k in flat.inputs]
        binding = self._binding(node, flat.outputs, in_layouts, layouts)
        kernel = self._inside(node, in_layouts, binding)
        signature = node_signature(
            node,
            frozenset(),
            in_layouts,
            declared=None if kernel is None else kernel.outputs,
        )
        check_conforms(node, signature.outputs, binding)
        description = str(node.op) if kernel is None else kernel.description
        return Step(
            node, flat.inputs, flat.outputs, in_layouts, kernel, signature, description
        )

    def _inside(self, node, in_layouts, binding) -> Kernel | None:
        """The kernel that runs a node inside generated code, or None if its code is
        generated (and inlined)."""
        if isinstance(node.op, Scan):
            return None
        if isinstance(node.op, Blockwise):
            # generated if its core graph can be planned (the emitter does it again,
            # from the cache)
            core_layouts = blockwise_core_layouts(node, in_layouts)
            try:
                self.graph(*blockwise_core(node), core_layouts)
                return None
            except NotImplementedError:
                return self._kernel(node, in_layouts, binding)
        if self._can_generate(node, in_layouts):
            return None
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
        if any(isinstance(repr_of(o.type), ScalarRepr) for o in node.outputs) or any(
            isinstance(repr_of(types[k]), ScalarRepr)
            for pos, k in enumerate(flat.inputs)
            if pos not in folded
        ):
            raise NotImplementedError("scalar node outside a unit")
        # A Rust kernel first: there is nothing to compile. Not for a Scan or Blockwise
        # (its inner graph is inlined into the generated loop), and not with folded
        # constants (a kernel would need them as arguments).
        nested = isinstance(node.op, Scan | Blockwise)
        writable = self._writable_inputs(flat, folded)
        if not folded and not nested and not writable:
            kernel = self._rust(node, in_layouts, binding)
            if kernel is not None:
                return self._kernel_step(flat, kernel, layouts)
        try:
            return self._generated_step(flat, folded, in_layouts, layouts, writable)
        except NotImplementedError:
            # (a kernel doesn't overwrite its inputs: it can stand in for an in-place
            # node too)
            if nested or writable:
                kernel = self._rust(node, in_layouts, binding)
                if kernel is not None:
                    return self._kernel_step(flat, kernel, layouts)
            if not self.opts.numba_fallback:
                raise
        kernel = self._numba(node, binding)
        # it overwrites the inputs its op destroys: that is what the graph asks for;
        # where nothing else may see the memory (as in `_writable_inputs`). Else it gets
        # copies.
        keep = frozenset(i for i in kernel.destroys if not self._may_destroy(flat, i))
        return self._kernel_step(flat, replace(kernel, destroys=keep), layouts)

    def _kernel_step(self, flat, kernel: Kernel, layouts) -> Step:
        """A step of the program run by a kernel (Rust, numba)."""
        in_layouts = [self.layout(k, layouts) for k in flat.inputs]
        sig = node_signature(flat.node, in_layouts=in_layouts, declared=kernel.outputs)
        binding = self._binding(flat.node, flat.outputs, in_layouts, layouts)
        check_conforms(flat.node, sig.outputs, binding)
        return Step(
            flat.node,
            flat.inputs,
            flat.outputs,
            in_layouts,
            kernel,
            sig,
            kernel.description,
        )

    def _generated_step(
        self, flat, folded, in_layouts, layouts, writable=frozenset()
    ) -> Step:
        """A step of the program whose code is generated and compiled in a module of
        its own. Raises NotImplementedError if no emitter handles it. `folded`: the
        constant scalar inputs compiled into the code; `writable`: the inputs it may
        overwrite (`_writable_inputs`)."""
        node = flat.node
        module = ir.Module(
            f"node_{type(node.op).__name__}",
            fastmath=self.opts.fastmath,
            debug_info=self.opts.debug_info,
        )
        nf = build_node_function(
            module,
            node,
            "node",
            folded=folded,
            in_layouts=in_layouts,
            parallel=self.opts.parallel,
            links=LinkTable(),
            planner=self,
            writable=writable,
        )
        if writable:
            self.stats.inplace.append(str(node.op))
        skipped = skipped_inputs(node, folded)
        args = [k for pos, k in enumerate(flat.inputs) if pos not in skipped]
        if _is_unit(node):
            self.stats.n_units += 1
            nodes = node.op.fgraph.toposort()
            names = ", ".join(str(n.op) for n in nodes)
            desc = f"unit[{len(nodes)}]: {names[:120]}"
        else:
            desc = str(node.op) + _layouts_note(
                [r for r in nf.signature.inputs if hasattr(r, "layout") and r.ndim > 0]
            )
        kernel = self._compiled(module, nf, desc)
        binding = self._binding(node, flat.outputs, in_layouts, layouts)
        check_conforms(node, nf.signature.outputs, binding)
        return Step(
            None,
            args,
            list(flat.outputs),
            [self.layout(k, layouts) for k in args],
            kernel,
            nf.signature,
            desc,
        )

    def _copy_step(self, src, dst, layouts) -> Step:
        """A step of the program copying value `src` into value `dst`: a `DeepCopyOp`,
        generated."""
        node = _copy_node(self.values.types[dst])
        in_layouts = [self.layout(src, layouts)]
        self._binding(node, [dst], in_layouts, layouts)
        flat = FlatNode(node, [src], [dst])
        return self._generated_step(flat, {}, in_layouts, layouts)

    def _compiled(self, module, nf, description) -> Kernel:
        """The kernel of the module `module` (its function `kernel`, around `nf`), to
        compile with the program (its address is set then)."""
        build_kernel_wrapper(module, nf, "kernel", self.opts.check_layouts)
        links = nf.links
        kernel = Kernel(
            0,
            0 if links is None else links.address,
            list(nf.signature.outputs),
            description,
            None if links is None else links.keepalive,
        )
        self.modules.append((module, kernel))
        return kernel

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
        decides whether it does (interface.py)."""
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

    def _rust(self, node, in_layouts, binding) -> Kernel | None:
        """The node's Rust kernel, if it has one that gives its outputs the `binding`
        layouts."""
        k = rust_kernel(node.op, node, in_layouts, self.opts.linalg_parallel)
        if k is None:
            return None
        outputs = rust_kernel_outputs(k, node)
        if not conforms(node, outputs, binding):
            return None
        self.stats.n_rust_kernels += 1
        return Kernel(k.address, k.config, outputs, repr(k), k)

    def _numba(self, node, binding) -> Kernel:
        """The node's numba implementation as a kernel. It allocates its outputs, packed
        in the `binding` layouts (C-contiguous where they aren't packed: they must be
        implied by it)."""
        from pytensor.link.llvm.kernels.numba import numba_kernel
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
        return Kernel(
            interp().numba_entry(),
            kernel.address,
            outs,
            f"numba: {node.op}",
            kernel,
            frozenset(i for ins in destroyed.values() for i in ins),
        )

    def _kernel(self, node, in_layouts, binding) -> Kernel:
        """The kernel for a node we don't generate: Rust, else numba."""
        kernel = self._rust(node, in_layouts, binding)
        if kernel is not None:
            return kernel
        if not self.opts.numba_fallback:
            raise NotImplementedError(f"No LLVM backend implementation for {node.op}")
        return self._numba(node, binding)

    def _can_generate(self, node, in_layouts) -> bool:
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


def _is_unit(node) -> bool:
    """Is the node a compiled unit, an OpFromGraph with `execution=COMPILED`? Its inner
    graph is generated as one function."""
    return isinstance(node.op, OpFromGraph) and node.op.execution is Execution.COMPILED


def _copy_node(t):
    from pytensor.compile.ops import deep_copy_op

    return deep_copy_op(t()).owner


def _layouts_note(reprs):
    """` {C, [?, P0]}`: the input layouts a generated step is specialized for."""

    def unknown(r):
        n = sum(1 for s in r.shape if s != 1)
        return n > 0 and len(r.stride_axes) == n

    if not reprs or all(unknown(r) for r in reprs):
        return ""
    return (
        " {"
        + ", ".join("?" if unknown(r) else r.layout.describe(r.shape) for r in reprs)
        + "}"
    )


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
