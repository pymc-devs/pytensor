"""Planning programs: a FunctionGraph becomes a list of tasks for the Rust interpreter.

1. Flatten: OpFromGraphs without a `unit` are inlined (with their own variable scope);
   OpFromGraphs with `execution=COMPILED` become compiled units; `execution=INTERPRETED` become
   subprograms. Nodes touching ScalarType values are grouped into implicit compiled
   units, since scalars never cross the kernel ABI.
2. Each remaining node becomes a task running a kernel: a Rust kernel (`rust_kernel`),
   a generated kernel (`llvm_emit`), or a numba kernel.
3. Output modes decide where values live: provided outputs are arena temporaries (or
   fresh buffers for graph outputs), allocated outputs are released after their last
   use, views keep their source alive. Arena offsets (also for task frames) are
   assigned here by liveness.
4. Generated modules are compiled in one parallel batch.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field

import numpy as np

from pytensor.compile.builders import OpFromGraph, Execution
from pytensor.graph.basic import Constant
import pytensor.link.llvm.codegen  # noqa: F401 (registers the emitters)
from pytensor.link.llvm.codegen.fused import fused_spec
from pytensor.link.llvm.codegen.function import (
    OutputMode,
    build_kernel_wrapper,
    build_node_function,
    node_outputs,
)
from pytensor.link.llvm.codegen.function import skipped_inputs
from pytensor.link.llvm.codegen.types import ScalarRepr, is_absent, repr_of
from pytensor.link.llvm.codegen.unit import NodeItem, build_unit_function
from pytensor.link.llvm.codegen.call import LinkTable
from pytensor.link.llvm.codegen.scan import scan_supported
from pytensor.scan.op import Scan
from pytensor.link.llvm.rust_kernels import rust_kernel_outputs, rust_kernel
from pytensor.link.llvm.layout import Layout
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.basic import scalar_from_tensor
from pytensor.tensor.type import TensorType


def _ir():
    from pytensor_llvm import ir

    return ir


def _interp():
    from pytensor_llvm import interp

    return interp


@dataclass
class Options:
    fastmath: bool = True
    opt_level: int = 3
    cache: bool = True
    numba_fallback: bool = True
    #: what generated code may assume about the graph inputs: "any" strides, or "c"
    #: (C-contiguous: Python calls copy other arrays, the C API rejects them)
    input_layout: str = "any"
    #: may generated loops run in parallel (chunks on the runtime's thread pool, when
    #: there is enough work; see work.py)
    parallel: bool = True
    #: generated tasks verify the layouts their signatures claim (for testing; also
    #: turned on by PYTENSOR_LLVM_CHECK_LAYOUTS=1)
    check_layouts: bool = False
    #: generate Scan loops (False: always the Rust kernel running the loop body
    #: as a subprogram)
    generate_scan: bool = True

    def __post_init__(self):
        import os

        if os.environ.get("PYTENSOR_LLVM_CHECK_LAYOUTS", "0") not in ("", "0"):
            self.check_layouts = True


#: the options of the program being planned (subprograms, e.g. a Scan kernel's loop body, use them)
_current_options: contextvars.ContextVar = contextvars.ContextVar("llvm_options", default=None)


def current_options() -> Options:
    return _current_options.get() or Options()


@dataclass
class UnitItem:
    """Several items compiled as one function."""

    items: list
    inputs: list  # vids read from outside
    outputs: list  # (inner vid, outer vid)
    description: str


@dataclass
class SubprogramItem:
    """An interpreted OpFromGraph: a subprogram."""

    node: object
    inputs: list
    outputs: list


@dataclass
class Task:
    """One entry of a program: a kernel called with the values `inputs`, producing the
    values `outputs` (vids), with its config and a frame of `frame_size` bytes."""

    inputs: list
    outputs: list
    modes: list  # OutputMode per output
    view_of: list  # input index per output (VIEW), else None
    abi: str = "kernel"
    address: int = 0
    config: int = 0
    frame_size: int = 0
    description: str = ""
    keepalive: object = None
    #: (module, function name) to compile; the address is filled in afterwards
    module: object = None


@dataclass
class Stats:
    plan_s: float = 0.0
    codegen_s: float = 0.0
    compile_s: float = 0.0
    n_modules: int = 0
    n_rust_kernels: int = 0
    n_numba: int = 0
    n_units: int = 0
    n_subprograms: int = 0
    numba: list = field(default_factory=list)


class Planner:
    def __init__(self, options: Options):
        self.opts = options
        self.types = {}  # vid -> PyTensor type
        self.consts = {}  # vid -> numpy data
        self._next = 0
        self.stats = Stats()
        self.layouts = {}  # vid -> Layout of tensor values (see layout.py)
        #: vid -> packed layout of the memory the planner provides (arena, fresh)
        self.memory_layouts = {}
        self.graph_outputs = set()
        #: (temporary, graph output) pairs to copy after the current task
        self.pending_copies = []
        #: layouts of the program's outputs (after `plan`)
        self.output_layouts = []

    # ------------------------------------------------------------ value ids and flattening

    def new_value(self, t):
        k = self._next
        self._next += 1
        self.types[k] = t
        return k

    def value_of(self, var, scope):
        if var in scope:
            return scope[var]
        if is_absent(var.type):
            # None inputs have no value; nothing ever reads this vid
            k = self.new_value(var.type)
            scope[var] = k
            return k
        if isinstance(var, Constant):
            k = self.new_value(var.type)
            self.consts[k] = np.asarray(var.data)
            scope[var] = k
            return k
        raise ValueError(f"variable {var} used before it is defined")

    def flatten(self, nodes, scope, inside_unit=False):
        items = []
        for node in nodes:
            op = node.op
            execution = getattr(op, "execution", None) if isinstance(op, OpFromGraph) else None
            if isinstance(op, OpFromGraph) and execution is None and fusable(op, node):
                # a FusedElemwise we can generate as one loop: keep it as one node
                ins = [self.value_of(i, scope) for i in node.inputs]
                outs = [self.new_value(o.type) for o in node.outputs]
                scope.update(zip(node.outputs, outs, strict=True))
                items.append(NodeItem(node, ins, outs))
            elif isinstance(op, OpFromGraph) and (execution is None or inside_unit):
                # (inside generated code every OpFromGraph is inlined)
                inner = op.fgraph
                sub = {iv: self.value_of(ov, scope) for iv, ov in zip(inner.inputs, node.inputs, strict=True)}
                items += self.flatten(inner.toposort(), sub, inside_unit)
                for io, oo in zip(inner.outputs, node.outputs, strict=True):
                    scope[oo] = self.value_of(io, sub)
            elif isinstance(op, OpFromGraph) and execution is Execution.COMPILED:
                inner = op.fgraph
                sub = {iv: self.value_of(ov, scope) for iv, ov in zip(inner.inputs, node.inputs, strict=True)}
                inner_scope = dict(sub)
                try:
                    inner_items = self.flatten(inner.toposort(), inner_scope, inside_unit=True)
                except NotImplementedError:
                    # can't be one unit (e.g. it contains an interpreted OpFromGraph):
                    # run its nodes as tasks
                    items += self.flatten(inner.toposort(), sub, inside_unit)
                    for io, oo in zip(inner.outputs, node.outputs, strict=True):
                        scope[oo] = self.value_of(io, sub)
                    continue
                sub = inner_scope
                outs = []
                for io, oo in zip(inner.outputs, node.outputs, strict=True):
                    k = self.new_value(oo.type)
                    scope[oo] = k
                    outs.append((self.value_of(io, sub), k))
                items.append(UnitItem(inner_items, self._unit_inputs(inner_items), outs, str(op)))
            elif isinstance(op, OpFromGraph) and execution is Execution.INTERPRETED:
                ins = [self.value_of(i, scope) for i in node.inputs]
                outs = [self.new_value(o.type) for o in node.outputs]
                scope.update(zip(node.outputs, outs, strict=True))
                items.append(SubprogramItem(node, ins, outs))
            else:
                ins = [self.value_of(i, scope) for i in node.inputs]
                outs = [self.new_value(o.type) for o in node.outputs]
                scope.update(zip(node.outputs, outs, strict=True))
                items.append(NodeItem(node, ins, outs))
        return items

    def _unit_inputs(self, items):
        produced = {k for it in items for k in it.outputs}
        ins = []
        for it in items:
            for k in it.inputs:
                if k in produced or k in ins or self._folded(k) or is_absent(self.types[k]):
                    continue
                ins.append(k)
        return ins

    def _memory_order(self, k):
        lay = self.memory_layouts.get(k)
        return None if lay is None else [int(a) for a in lay.order]

    def lay(self, k):
        """The layout of value `k` (None for non-tensors)."""
        t = self.types[k]
        if not hasattr(t, "ndim") or is_absent(t):
            return None
        if k in self.consts:
            return Layout.c(t.ndim)  # constants are stored C-contiguous
        return self.layouts.get(k)

    def set_layouts(self, vids, layouts):
        for k, lay in zip(vids, layouts, strict=True):
            if lay is not None:
                self.layouts[k] = lay

    def _folded(self, k):
        """0-d constants are folded into compiled units."""
        return k in self.consts and self.consts[k].ndim == 0

    # ------------------------------------------------------------ scalar islands

    def group_scalars(self, items, graph_outputs):
        """Group items that touch ScalarType values into compiled units."""
        is_scalar = {k for k, t in self.types.items() if isinstance(t, ScalarType)}

        def touches(it):
            # constant scalars are folded into a node's code: they don't need a unit
            return isinstance(it, NodeItem) and any(
                k in is_scalar and k not in self.consts for k in (*it.inputs, *it.outputs)
            )

        if not any(touches(it) for it in items):
            return items
        from pytensor.tensor.basic import scalar_from_tensor, tensor_from_scalar

        # Greedy convex groups in program order (like the `llvm_units` rewrite): an item
        # touching scalars joins the current group unless it depends on an item outside
        # the group that depends on the group; then a new group starts. A scalar made in
        # an earlier group crosses into a later one as a 0-d tensor.
        groups = []  # lists of items
        made_in = {}  # scalar vid -> index of the group producing it
        as_tensor = {}  # scalar vid -> 0-d tensor vid carrying it out of its group
        out = []  # items outside groups, and group placeholders (ints)
        produced, tainted = set(), set()
        for it in items:
            if not touches(it):
                if any(k in produced or k in tainted for k in it.inputs):
                    tainted.update(o[1] if isinstance(o, tuple) else o for o in it.outputs)
                out.append(it)
                continue
            if not groups or any(k in tainted for k in it.inputs):
                groups.append([])
                out.append(len(groups) - 1)
                produced, tainted = set(), set()
            g = len(groups) - 1
            ins = list(it.inputs)
            for pos, k in enumerate(ins):
                if k not in is_scalar or k in self.consts:
                    continue
                if k not in made_in:
                    raise NotImplementedError("a ScalarType graph input")
                if made_in[k] == g:
                    continue
                if k not in as_tensor:
                    conv = tensor_from_scalar(self.types[k]()).owner
                    t = self.new_value(conv.outputs[0].type)
                    groups[made_in[k]].append(NodeItem(conv, [k], [t]))
                    as_tensor[k] = t
                t = as_tensor[k]
                back = scalar_from_tensor(self.types[t]()).owner
                k2 = self.new_value(self.types[k])
                groups[g].append(NodeItem(back, [t], [k2]))
                made_in[k2] = g
                produced.add(k2)
                ins[pos] = k2
            it = NodeItem(it.node, ins, it.outputs) if ins != it.inputs else it
            groups[g].append(it)
            for k in it.outputs:
                produced.add(k)
                if k in is_scalar:
                    made_in[k] = g
        consumers = {}
        for x in out:
            for s in groups[x] if isinstance(x, int) else [x]:
                for k in s.inputs:
                    consumers.setdefault(k, set()).add(id(s))
        result = []
        for x in out:
            if not isinstance(x, int):
                result.append(x)
                continue
            sub = groups[x]
            inside = {id(s) for s in sub}
            outputs = []
            for s in sub:
                for k in s.outputs:
                    if k in graph_outputs or any(c not in inside for c in consumers.get(k, ())):
                        if k in is_scalar:
                            raise NotImplementedError("a ScalarType value used outside its compiled unit")
                        outputs.append((k, k))
            result.append(UnitItem(sub, self._unit_inputs(sub), outputs, "scalars"))
        return toposort_items(result)

    # ------------------------------------------------------------ tasks

    def provide(self, vids, declared):
        """Memory for the PROVIDED outputs of a task, in their declared layouts: arena
        memory in the declared order (any packed order works); for a layout that isn't
        packed (any strides) C-contiguous memory. Graph outputs must be C-contiguous
        (that's what callers get and what the C API provides): one declared otherwise
        is written to a temporary and copied (`pending_copies`). Returns the vids the
        task writes."""
        out_ids = []
        for vid, o in zip(vids, declared, strict=True):
            if o.mode is not OutputMode.PROVIDED:
                out_ids.append(vid)
                continue
            packed = o.layout.fully_packed(o.repr.shape)
            memory = o.layout if packed else Layout.c(o.repr.ndim)
            if vid in self.graph_outputs and not memory.is_c(o.repr.shape):
                tmp = self.new_value(self.types[vid])
                self.pending_copies.append((tmp, vid))
                vid = tmp
            self.memory_layouts[vid] = memory
            out_ids.append(vid)
        return out_ids

    def task_for_item(self, it):
        ir = _ir()
        if isinstance(it, UnitItem):
            modes = [OutputMode.PROVIDED if repr_of(self.types[o]).static else OutputMode.ALLOCATED for _, o in it.outputs]
            module = ir.Module(f"unit_{len(it.items)}", fastmath=self.opts.fastmath)
            consts = {k: (self.types[k], self.consts[k]) for s in it.items for k in s.inputs if self._folded(k)}
            in_lays = [self.lay(k) for k in it.inputs]
            nf = build_unit_function(
                module,
                "unit",
                [(k, self.types[k]) for k in it.inputs],
                [(inner, self.types[outer], mode) for (inner, outer), mode in zip(it.outputs, modes, strict=True)],
                it.items,
                consts,
                input_layouts=in_lays,
                output_layouts=[Layout.c(self.types[o].ndim) for _, o in it.outputs],
                parallel=self.opts.parallel,
                kernel_for=self.kernel_for,
                planner=self,
            )
            build_kernel_wrapper(module, nf, "kernel", self.opts.check_layouts)
            # provided memory is C-contiguous, and so is what a unit allocates
            self.set_layouts([o for _, o in it.outputs], [Layout.c(self.types[o].ndim) for _, o in it.outputs])
            self.stats.n_units += 1
            return Task(it.inputs, [o for _, o in it.outputs], modes, [None] * len(modes), frame_size=nf.frame_size,
                        description=f"unit[{len(it.items)}]: " + ", ".join(str(s.node.op) for s in it.items if isinstance(s, NodeItem))[:120],
                        module=module, config=nf.links.address, keepalive=nf.links.keepalive)
        if isinstance(it, SubprogramItem):
            node = it.node
            modes = [
                OutputMode.ABSENT if is_absent(o.type)
                else OutputMode.PROVIDED if repr_of(o.type).static
                else OutputMode.ALLOCATED
                for o in node.outputs
            ]
            sub = Planner(self.opts)
            prog = sub.plan(node.op.fgraph, output_modes=modes, input_layouts=[self.lay(k) for k in it.inputs])
            # (values without data aren't arguments of the subprogram either)
            ins = [k for k, i in zip(it.inputs, node.inputs, strict=True) if not is_absent(i.type)]
            present = [j for j, m in enumerate(modes) if m is not OutputMode.ABSENT]
            # an allocated output is the inner value if it owns a buffer, else a C copy
            self.set_layouts(
                it.outputs,
                [
                    None if m is OutputMode.ABSENT
                    else Layout.c(o.type.ndim) if m is OutputMode.PROVIDED or lay is None
                    else lay.join(Layout.c(o.type.ndim), _shape(o.type), _isz(o.type))
                    for o, m, lay in zip(node.outputs, modes, sub.output_layouts, strict=True)
                ],
            )
            self.stats.n_subprograms += 1
            return Task(ins, [it.outputs[j] for j in present], [modes[j] for j in present], [None] * len(present),
                        address=prog.address, config=prog.config, description=f"program: {node.op}", keepalive=prog)
        node = it.node
        # constant scalar inputs are folded into the code
        folded = {
            pos: self.consts[k].item()
            for pos, (k, var) in enumerate(zip(it.inputs, node.inputs, strict=True))
            if k in self.consts and isinstance(var.type, ScalarType)
        }
        if any(isinstance(repr_of(o.type), ScalarRepr) for o in node.outputs) or any(
            isinstance(repr_of(self.types[k]), ScalarRepr) for pos, k in enumerate(it.inputs) if pos not in folded
        ):
            raise NotImplementedError("scalar node outside a unit")
        present = [k for k, var in zip(it.inputs, node.inputs, strict=True) if not is_absent(var.type)]
        in_lays = [self.lay(k) for k in it.inputs]
        # the implementation first, then what it says it produces (a Scan we generate
        # rather than run the kernel: its body is inlined into the loop)
        generate_first = isinstance(node.op, Scan) and scan_supported(node.op) and self.opts.generate_scan
        k = None if folded or generate_first else rust_kernel(node.op, node, in_lays)
        if k is not None:
            declared = rust_kernel_outputs(k, node, in_lays)
            modes = [o.mode for o in declared]
            views = [o.view_of for o in declared]
            self.stats.n_rust_kernels += 1
            outs = self.provide(it.outputs, declared)
            self.set_layouts(outs, [o.layout for o in declared])
            outs, modes, views = _present_outputs(declared, outs, modes, views)
            return Task(present, outs, modes, views, address=k.address, config=k.config, description=repr(k), keepalive=k)
        try:
            module = ir.Module(f"node_{type(node.op).__name__}", fastmath=self.opts.fastmath)
            links = LinkTable()
            nf = build_node_function(
                module, node, "node", folded=folded, in_layouts=in_lays, parallel=self.opts.parallel, links=links, planner=self
            )
            build_kernel_wrapper(module, nf, "kernel", self.opts.check_layouts)
            skipped = skipped_inputs(node, folded)
            inputs = [k for pos, k in enumerate(it.inputs) if pos not in skipped]
            # view_of counts the task's inputs
            modes = [o.mode for o in nf.signature.outputs]
            vmap = [o.view_of for o in nf.signature.outputs]  # (counts the task's inputs)
            outs = self.provide(it.outputs, nf.signature.outputs)
            self.set_layouts(outs, [o.layout for o in nf.signature.outputs])
            outs, modes, vmap = _present_outputs(nf.signature.outputs, outs, modes, vmap)
            desc = str(node.op) + _layouts_note([r for r in nf.signature.inputs if hasattr(r, "layout") and r.ndim > 0])
            return Task(inputs, outs, modes, vmap, frame_size=nf.frame_size, description=desc, module=module,
                        config=links.address, keepalive=links.keepalive)
        except NotImplementedError:
            if generate_first:
                k = rust_kernel(node.op, node, in_lays)
                if k is not None:
                    declared = rust_kernel_outputs(k, node, in_lays)
                    outs = self.provide(it.outputs, declared)
                    self.set_layouts(outs, [o.layout for o in declared])
                    outs, modes, views = _present_outputs(declared, outs, [o.mode for o in declared], [o.view_of for o in declared])
                    self.stats.n_rust_kernels += 1
                    return Task(present, outs, modes, views, address=k.address, config=k.config, description=repr(k), keepalive=k)
            if not self.opts.numba_fallback:
                raise
        return self.numba_task(it)

    def kernel_for(self, node, in_layouts):
        """How a node inside generated code (a unit, a loop) is implemented: None to
        generate it (inlined: preferred), else the `CalledKernel` it calls: its Rust kernel,
        or the numba implementation."""
        from pytensor.link.llvm.codegen.call import CalledKernel
        from pytensor.link.llvm.codegen.function import Output, llvm_emit

        if isinstance(node.op, OpFromGraph) and getattr(node.op, "execution", None) is Execution.INTERPRETED:
            return self._subprogram_kernel(node, in_layouts)
        if isinstance(node.op, Scan) and self.opts.generate_scan and scan_supported(node.op):
            return None  # generated (everything inside can be called)
        if llvm_emit.dispatch(type(node.op)) is not llvm_emit.dispatch(object) and self._can_generate(node, in_layouts):
            return None
        k = rust_kernel(node.op, node, in_layouts)
        if k is not None:
            self.stats.n_rust_kernels += 1
            return CalledKernel(k.address, k.config, "kernel", rust_kernel_outputs(k, node, in_layouts), repr(k), k)
        if not self.opts.numba_fallback:
            raise NotImplementedError(f"No LLVM backend implementation for {node.op}")
        from pytensor.link.llvm.numba_kernels import numba_kernel
        from pytensor.tensor.random.type import RandomType

        if any(isinstance(v.type, RandomType) for v in (*node.inputs, *node.outputs)):
            raise NotImplementedError(f"LLVM backend: no implementation of the random variable {node.op}")
        kernel = numba_kernel(node)
        self.stats.n_numba += 1
        self.stats.numba.append(str(node.op))
        outs = [Output(repr_of(o.type, Layout.dynamic(o.type.ndim)), OutputMode.ALLOCATED) for o in node.outputs]
        return CalledKernel(kernel.address, 0, "numba", outs, f"numba: {node.op}", kernel)

    def _subprogram_kernel(self, node, in_layouts):
        """A subprogram for an OpFromGraph, called from generated code."""
        from pytensor.link.llvm.codegen.call import CalledKernel
        from pytensor.link.llvm.codegen.function import Output

        if any(isinstance(o.type, ScalarType) for o in node.outputs):
            raise NotImplementedError("a subprogram with ScalarType outputs")
        modes = [
            OutputMode.ABSENT if is_absent(o.type) else OutputMode.PROVIDED if repr_of(o.type).static else OutputMode.ALLOCATED
            for o in node.outputs
        ]
        sub = Planner(self.opts)
        prog = sub.plan(node.op.fgraph, output_modes=modes, input_layouts=in_layouts)
        self.stats.n_subprograms += 1
        outs = []
        for o, mode, lay in zip(node.outputs, modes, sub.output_layouts, strict=True):
            if mode is OutputMode.ABSENT:
                outs.append(Output(repr_of(o.type), mode))
            elif mode is OutputMode.PROVIDED:
                outs.append(Output(repr_of(o.type, Layout.c(o.type.ndim)), mode))
            else:
                # an allocated output is the inner value if it owns a buffer, else a C copy
                lay = Layout.c(o.type.ndim) if lay is None else lay.join(Layout.c(o.type.ndim), _shape(o.type), _isz(o.type))
                outs.append(Output(repr_of(o.type, lay), mode))
        return CalledKernel(prog.address, prog.config, "kernel", outs, f"program: {node.op}", prog)

    def _can_generate(self, node, in_layouts) -> bool:
        """Can we generate code for `node`? (Tried in a scratch module, so a failure
        leaves no half-built function behind.)"""
        try:
            build_node_function(_ir().Module("probe"), node, "probe", in_layouts=in_layouts, parallel=False)
            return True
        except NotImplementedError:
            return False

    def numba_task(self, it):
        from pytensor.link.llvm.numba_kernels import numba_kernel

        node = it.node
        from pytensor.tensor.random.type import RandomType

        if any(isinstance(v.type, RandomType) for v in (*node.inputs, *node.outputs)):
            raise NotImplementedError(f"LLVM backend: no implementation of the random variable {node.op}")
        kernel = numba_kernel(node)
        self.stats.n_numba += 1
        self.stats.numba.append(str(node.op))
        n = len(node.outputs)
        # numba may return any layout
        self.set_layouts(it.outputs, [Layout.dynamic(o.type.ndim) if hasattr(o.type, "ndim") else None for o in node.outputs])
        return Task(it.inputs, it.outputs, [OutputMode.ALLOCATED] * n, [None] * n, abi="numba", address=kernel.address,
                    description=f"numba: {node.op}", keepalive=kernel)

    # ------------------------------------------------------------ the program

    def plan(self, fgraph, output_modes=None, input_layouts=None):
        """`input_layouts`: what the caller guarantees about each input's layout (default:
        from `Options.input_layout`)."""
        token = _current_options.set(self.opts)
        try:
            return self._plan(fgraph, output_modes, input_layouts)
        finally:
            _current_options.reset(token)

    def _plan(self, fgraph, output_modes, input_layouts):
        t0 = time.perf_counter()
        scope = {}
        in_ids = []
        scalar_inputs = []
        for pos, v in enumerate(fgraph.inputs):
            if isinstance(v.type, ScalarType):
                # scalars don't cross the kernel ABI: the input is a 0-d array, read into the
                # scalar by a ScalarFromTensor (which joins the scalars' compiled unit)
                t = self.new_value(TensorType(v.type.dtype, shape=()))
                k = self.new_value(v.type)
                scalar_inputs.append((t, k))
                in_ids.append(t)
                self.layouts[t] = Layout.c(0)
                scope[v] = k
                continue
            k = self.new_value(v.type)
            scope[v] = k
            in_ids.append(k)
            if hasattr(v.type, "ndim"):
                lay = input_layouts[pos] if input_layouts is not None else None
                if lay is None:
                    lay = Layout.c(v.type.ndim) if self.opts.input_layout == "c" else Layout.dynamic(v.type.ndim)
                self.layouts[k] = lay
        self.input_contiguous = [
            input_layouts is None and self.opts.input_layout == "c" and hasattr(v.type, "ndim") for v in fgraph.inputs
        ]
        items = [
            NodeItem(scalar_from_tensor(self.types[t]()).owner, [t], [k]) for t, k in scalar_inputs
        ] + self.flatten(fgraph.toposort(), scope)
        out_ids = [self.value_of(o, scope) for o in fgraph.outputs]
        self.graph_outputs = set(out_ids)
        items = self.group_scalars(items, set(out_ids))
        tasks = self.tasks_for(items)
        self.output_layouts = [self.lay(k) for k in out_ids]
        t1 = time.perf_counter()

        # compile all generated modules at once
        modules = []
        for s in tasks:
            if s.module is not None:
                s.module.label = "ptl:" + " ".join(s.description.split())[:160]  # perf maps
                modules.append(s.module)
        if modules:
            compiled = _ir().compile_many(modules, opt_level=self.opts.opt_level, cache=self.opts.cache)
            for s, c in zip([s for s in tasks if s.module is not None], compiled, strict=True):
                s.address = c.address("kernel")
                s.keepalive = (s.keepalive, c)  # (the links table, if any)
                s.module = None
        self.stats.n_modules = len(modules)
        t2 = time.perf_counter()

        # values without data (random generators) aren't inputs or outputs of the program
        self.program_inputs = [pos for pos, v in enumerate(fgraph.inputs) if not is_absent(v.type)]
        self.program_outputs = [pos for pos, v in enumerate(fgraph.outputs) if not is_absent(v.type)]
        modes = output_modes or [OutputMode.ALLOCATED] * len(out_ids)
        self.input_contiguous = [self.input_contiguous[pos] for pos in self.program_inputs]
        prog = self.build(
            tasks,
            [in_ids[pos] for pos in self.program_inputs],
            [out_ids[pos] for pos in self.program_outputs],
            [modes[pos] for pos in self.program_outputs],
        )
        self.stats.codegen_s = t1 - t0
        self.stats.compile_s = t2 - t1
        self.stats.plan_s = time.perf_counter() - t2
        return prog

    def tasks_for(self, items):
        tasks = []
        for it in items:
            try:
                tasks.append(self.task_for_item(it))
                while self.pending_copies:
                    tasks.append(self.copy_task(*self.pending_copies.pop(0)))
            except NotImplementedError:
                if not isinstance(it, UnitItem) or it.description == "scalars":
                    raise
                # a compiled OpFromGraph we can't generate: run its nodes as tasks, and
                # copy each result into the value the OpFromGraph's output is
                inner_outs = {inner for inner, _ in it.outputs}
                tasks += self.tasks_for(self.group_scalars(it.items, inner_outs))
                for inner, outer in it.outputs:
                    tasks.append(self.copy_task(inner, outer))
        return tasks

    def copy_task(self, src, dst):
        node = _copy_node(self.types[dst])
        self.layouts[dst] = Layout.c(self.types[dst].ndim)
        self.memory_layouts[dst] = Layout.c(self.types[dst].ndim)
        k = rust_kernel(node.op, node, [self.lay(src)])
        mode = rust_kernel_outputs(k, node, [self.lay(src)])[0].mode
        return Task([src], [dst], [mode], [None], address=k.address, config=k.config, description="copy", keepalive=k)

    def build(self, tasks, in_ids, out_ids, output_modes):
        interp = _interp()
        n_tasks = len(tasks)
        INF = n_tasks + 1
        kind = {}  # vid -> ("input"|"const"|"provided"|"allocated"|"view", source)
        for k in in_ids:
            kind[k] = ("input", None)
        for s in tasks:
            for k in s.inputs:
                if k in self.consts and k not in kind:
                    kind[k] = ("const", None)
        defined, end = {}, {}
        for k in kind:
            defined[k] = -1
        for i, s in enumerate(tasks):
            for k in s.inputs:
                end[k] = max(end.get(k, -1), i)
            for j, (k, mode) in enumerate(zip(s.outputs, s.modes, strict=True)):
                defined[k] = i
                end.setdefault(k, i)
                if mode is OutputMode.VIEW:
                    kind[k] = ("view", s.inputs[s.view_of[j]])
                elif mode is OutputMode.PROVIDED:
                    kind[k] = ("provided", None)
                else:
                    kind[k] = ("allocated", None)
        for k in out_ids:
            if k in self.consts and k not in kind:
                kind[k] = ("const", None)
                defined[k] = -1
            end[k] = INF
        # views keep their source alive (later values first, so chains propagate)
        for k in sorted(kind, key=lambda k: defined[k], reverse=True):
            what, src = kind[k]
            if what == "view":
                end[src] = max(end.get(src, -1), end[k])

        # arena: static provided temporaries and task frames, first fit by liveness
        intervals = []
        first_out = {}
        for o, k in enumerate(out_ids):
            first_out.setdefault(k, o)
        for k, (what, _) in kind.items():
            if what == "provided" and k not in first_out:
                t = self.types[k]
                nbytes = int(np.prod(t.shape, dtype=np.int64)) * np.dtype(t.dtype).itemsize
                intervals.append((defined[k], end[k], nbytes, ("value", k)))
        for i, s in enumerate(tasks):
            if s.frame_size:
                intervals.append((i, i, s.frame_size, ("frame", i)))
        offsets, arena = {}, 0
        active = []
        for start, stop, nbytes, what in sorted(intervals, key=lambda x: (x[0], x[1])):
            size = max(64, -(-nbytes // 64) * 64)
            active = [a for a in active if a[2] >= start]
            off = 0
            for a_off, a_size, _ in sorted(active):
                if off + size <= a_off:
                    break
                off = max(off, a_off + a_size)
            active.append((off, size, stop))
            offsets[what] = off
            arena = max(arena, off + size)

        pb = interp.ProgramBuilder()
        ids = {}

        def dtype(k):
            return np.dtype(self.types[k].dtype)

        def shape(k):
            t = self.types[k]
            return [None if s is None else int(s) for s in t.shape]

        for k, contiguous in zip(in_ids, self.input_contiguous, strict=True):
            ids[k] = pb.input(dtype(k), shape(k), contiguous)
        for k, (what, src) in sorted(kind.items(), key=lambda x: defined[x[0]]):
            if k in ids:
                continue
            if what == "const":
                ids[k] = pb.constant(np.asarray(self.consts[k], dtype=self.types[k].dtype))  # stored C-contiguous
            elif what == "provided":
                if k in first_out:
                    ids[k] = pb.fresh(dtype(k), shape(k))
                else:
                    ids[k] = pb.arena(dtype(k), shape(k), offsets[("value", k)], self._memory_order(k))
            elif what == "allocated":
                ids[k] = pb.allocated(dtype(k), shape(k))
            elif what == "view":
                ids[k] = pb.view(dtype(k), shape(k), ids[src])
        release = [[] for _ in tasks]
        for k, (what, _) in kind.items():
            if what in ("allocated", "view", "provided") and end.get(k, INF) < INF:
                release[max(end[k], defined[k])].append(ids[k])
        for i, s in enumerate(tasks):
            pb.task(
                s.address,
                s.config,
                interp.Abi.Numba if s.abi == "numba" else interp.Abi.Kernel,
                [ids[k] for k in s.inputs],
                [ids[k] for k in s.outputs],
                release[i],
                offsets.get(("frame", i)),
                s.description,
                s.keepalive,
            )
        pb.set_arena_bytes(arena)
        modes = [interp.OutputMode.Provided if m is OutputMode.PROVIDED else interp.OutputMode.Allocated for m in output_modes]
        return pb.build([ids[k] for k in out_ids], modes)


def fusable(op, node):
    from pytensor.tensor.rewriting.fused_elemwise import FusedElemwise

    if not isinstance(op, FusedElemwise):
        return False
    try:
        fused_spec(op, node)
    except NotImplementedError:
        return False
    return True


def _outputs_of(it):
    if isinstance(it, UnitItem):
        return [o for _, o in it.outputs]
    return it.outputs


def toposort_items(items):
    """Order items by data dependencies, keeping the given order where possible."""
    producer = {k: i for i, it in enumerate(items) for k in _outputs_of(it)}
    deps = [{producer[k] for k in it.inputs if k in producer and producer[k] != i} for i, it in enumerate(items)]
    done, order = set(), []
    pending = list(range(len(items)))
    while pending:
        progress = False
        rest = []
        for i in pending:
            if deps[i] <= done:
                order.append(items[i])
                done.add(i)
                progress = True
            else:
                rest.append(i)
        if not progress:
            raise NotImplementedError("scalar operations interleaved with other nodes (a cycle)")
        pending = rest
    return order


def _present_outputs(declared, vids, modes, views):
    """Drop the outputs without a value (random generators): they aren't task outputs."""
    keep = [j for j, o in enumerate(declared) if o.mode is not OutputMode.ABSENT]
    return [vids[j] for j in keep], [modes[j] for j in keep], [views[j] for j in keep]


def _shape(t):
    return tuple(None if s is None else int(s) for s in t.shape)


def _isz(t):
    return np.dtype(t.dtype).itemsize


def _layouts_note(reprs):
    """` {C, [?, P0]}`: the input layouts a generated task is specialized for."""

    def unknown(r):
        n = sum(1 for s in r.shape if s != 1)
        return n > 0 and len(r.stride_axes) == n

    if not reprs or all(unknown(r) for r in reprs):
        return ""
    return " {" + ", ".join("?" if unknown(r) else r.layout.describe(r.shape) for r in reprs) + "}"


def _copy_node(t):
    from pytensor.compile.ops import deep_copy_op

    return deep_copy_op(t()).owner


def build_program(fgraph, options: Options | None = None):
    """Plan and compile `fgraph`; returns (callable, stats). The callable is the program,
    or, when the graph has random generator inputs or outputs (values without data, see
    `is_absent`), a wrapper that leaves them out of the call and returns each generator
    output as the generator it came from (draws come from the program's own random
    state)."""
    planner = Planner(options or Options())
    prog = planner.plan(fgraph)
    n_in, n_out = len(fgraph.inputs), len(fgraph.outputs)
    if planner.program_inputs == list(range(n_in)) and planner.program_outputs == list(range(n_out)):
        return prog, planner.stats
    ins, outs = planner.program_inputs, planner.program_outputs
    sources = {pos: _generator_source(fgraph, o) for pos, o in enumerate(fgraph.outputs) if pos not in outs}

    def call(*args):
        res = prog(*[args[k] for k in ins])
        full = [None] * n_out
        for k, r in zip(outs, res, strict=True):
            full[k] = r
        for pos, src in sources.items():
            full[pos] = None if src is None else args[src]
        return full

    call.program = prog
    return call, planner.stats


def _generator_source(fgraph, var):
    """The position of the graph input a generator output passes on (through random
    variables and copies), or None."""
    seen = set()
    while var.owner is not None and var not in seen:
        seen.add(var)
        node = var.owner
        rng_inputs = [i for i in node.inputs if is_absent(i.type) and not _is_none_type(i.type)]
        if not rng_inputs:
            return None
        var = rng_inputs[0]
    return fgraph.inputs.index(var) if var in fgraph.inputs else None


def _is_none_type(t):
    from pytensor.tensor.type_other import NoneTypeT

    return isinstance(t, NoneTypeT)
