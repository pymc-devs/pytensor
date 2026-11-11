"""Planning programs: a FunctionGraph becomes a list of tasks for the Rust interpreter.

`Planner.plan` runs the stages of `pytensor.link.llvm.plan` and does what lies between
them:

1. Flatten (`plan/flatten.py`): the graph becomes items. OpFromGraphs are inlined,
   become units (`execution=COMPILED`) or subprograms (`execution=INTERPRETED`); nodes
   touching ScalarType values are grouped into units, since scalars never cross the
   kernel ABI.
2. Each item becomes a task (`Planner.task`): its implementation is resolved
   (`plan/resolve.py`: a Rust kernel, generated code, a numba kernel, a subprogram),
   code is generated for it if needed, and the layouts of its outputs are noted for
   the items after it.
3. The generated modules are compiled in one parallel batch.
4. Memory is planned (`plan/memory.py`): provided outputs are arena temporaries (or
   fresh buffers for graph outputs), returned ones are released after their last use,
   and whatever a value may be a view of is kept alive as long as it.
5. The program is built for the interpreter (`Planner.build`).
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field, replace

import numpy as np

import pytensor.link.llvm.codegen  # noqa: F401 (registers the emitters)
from pytensor.link.llvm.codegen.call import CalledKernel, LinkTable
from pytensor.link.llvm.codegen.function import (
    OutputMode,
    build_kernel_wrapper,
    build_node_function,
    inplace_outputs,
    skipped_inputs,
)
from pytensor.link.llvm.codegen.types import ScalarRepr, ir, is_absent, repr_of
from pytensor.link.llvm.codegen.unit import build_unit_function, unit_input_layouts
from pytensor.link.llvm.layout import Layout
from pytensor.link.llvm.plan.flatten import flatten, group_scalars
from pytensor.link.llvm.plan.memory import plan_memory
from pytensor.link.llvm.plan.resolve import Resolver
from pytensor.link.llvm.plan.values import NodeItem, SubprogramItem, UnitItem, Values
from pytensor.link.llvm.rust_kernels import rust_kernel, rust_kernel_outputs
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.basic import scalar_from_tensor
from pytensor.tensor.type import TensorType


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
_current_options: contextvars.ContextVar = contextvars.ContextVar(
    "llvm_options", default=None
)


def current_options() -> Options:
    return _current_options.get() or Options()


#: the stats of the program being planned: the planner of a subprogram or a Scan body
#: hands its modules to it
_current_stats: contextvars.ContextVar = contextvars.ContextVar(
    "llvm_stats", default=None
)


@dataclass
class Task:
    """One entry of a program: a kernel called with the values `inputs`, producing the
    values `outputs` (vids), with its config."""

    inputs: list
    outputs: list
    modes: list  # OutputMode per output
    aliases: list  # per output: the positions of the inputs it may be a view of
    address: int = 0
    config: int = 0
    description: str = ""
    keepalive: object = None
    #: (module, function name) to compile; the address is filled in afterwards
    module: object = None


@dataclass
class GeneratedModule:
    """A module the planner generated for a task, compiled."""

    description: str
    compiled: object  # ir.CompiledModule

    @property
    def ir(self) -> str:
        """The LLVM IR as generated."""
        return self.compiled.ir

    def optimized_ir(self, opt_level: int | None = None) -> str:
        """The LLVM IR after optimization."""
        return self.compiled.optimized_ir(opt_level)


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
    #: the ops of the tasks that overwrite an input
    inplace: list = field(default_factory=list)
    #: the generated modules (GeneratedModule), including those of subprograms and
    #: Scan bodies planned for this program
    modules: list = field(default_factory=list)

    def ir(self, optimized: bool = False, opt_level: int | None = None) -> str:
        """The LLVM IR of all generated modules, each under a comment naming its task.
        `optimized`: after the optimization pipeline instead of as generated."""
        parts = []
        for m in self.modules:
            text = m.optimized_ir(opt_level) if optimized else m.ir
            parts.append(f"; ===== {' '.join(m.description.split())}\n{text}")
        return "\n".join(parts)


class Planner:
    def __init__(self, options: Options):
        self.opts = options
        self.stats = Stats()
        self.values = Values()
        self.resolver = Resolver(options, self.stats, self.values)
        self.layouts = {}  # vid -> Layout of tensor values (see layout.py)
        #: vid -> packed layout of the memory the planner provides (arena, fresh)
        self.memory_layouts = {}
        self.graph_outputs = set()
        #: (temporary, graph output) pairs to copy after the current task
        self.pending_copies = []
        #: copy tasks that have to run before the task being made (`keep_destroyed`)
        self.copies_first = []
        #: layouts of the program's outputs (after `plan`)
        self.output_layouts = []
        #: vid -> the values whose memory it may lie in (itself, for new memory)
        self.roots = {}
        #: may tasks overwrite inputs their ops destroy? Only for a graph whose
        #: DestroyHandler has validated the order of its nodes (`plan`)
        self.allow_inplace = False

    # ------------------------------------------------------------ layouts of values

    def lay(self, k):
        """The layout of value `k` (None for non-tensors)."""
        t = self.values.types[k]
        if not hasattr(t, "ndim") or is_absent(t):
            return None
        if k in self.values.consts:
            return Layout.c(t.ndim)  # constants are stored C-contiguous
        return self.layouts.get(k)

    def set_layouts(self, vids, layouts):
        for k, lay in zip(vids, layouts, strict=True):
            if lay is not None:
                self.layouts[k] = lay

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
                tmp = self.values.new(self.values.types[vid])
                self.pending_copies.append((tmp, vid))
                vid = tmp
            self.memory_layouts[vid] = memory
            out_ids.append(vid)
        return out_ids

    # ------------------------------------------------------------ tasks

    def tasks_for(self, items):
        tasks = []
        for it in items:
            try:
                task = self.task(it)
                tasks += self.copies_first
                self.copies_first = []
                tasks.append(task)
                self.note_roots(task)
                while self.pending_copies:
                    tasks.append(self.copy_task(*self.pending_copies.pop(0)))
                    self.note_roots(tasks[-1])
            except NotImplementedError:
                self.copies_first = []
                if not isinstance(it, UnitItem) or it.description == "scalars":
                    raise
                # a compiled OpFromGraph we can't generate: run its nodes as tasks, and
                # copy each result into the value the OpFromGraph's output is
                inner_outs = {inner for inner, _ in it.outputs}
                tasks += self.tasks_for(
                    group_scalars(self.values, it.items, inner_outs)
                )
                for inner, outer in it.outputs:
                    tasks.append(self.copy_task(inner, outer))
                    self.note_roots(tasks[-1])
        return tasks

    def roots_of(self, k) -> frozenset:
        return self.roots.get(k, frozenset([k]))

    def note_roots(self, task) -> None:
        """Record whose memory the outputs of `task` may lie in."""
        for k, mode, may_alias in zip(
            task.outputs, task.modes, task.aliases, strict=True
        ):
            sources = [task.inputs[pos] for pos in may_alias]
            if mode is OutputMode.INPLACE or (mode is OutputMode.RETURNED and sources):
                self.roots[k] = frozenset().union(
                    *(self.roots_of(src) for src in sources)
                )

    def grant_inplace(self, it, folded) -> dict:
        """{output: input} for the outputs of a node that may overwrite an input: its
        op destroys it, its emitter can (`llvm_inplace`), and nothing else can see it."""
        if not (self.allow_inplace and it.of_the_graph):
            return {}
        values = self.values
        granted = {}
        for k, i in inplace_outputs(it.node, folded).items():
            vid = it.inputs[i]
            if vid in values.consts or it.inputs.count(vid) > 1:
                continue
            # The graph says the destroyed variable is not used again. The value must
            # not stand for other variables too, nor share memory with another input.
            mine = self.roots_of(vid)
            if any(root in values.shared for root in mine):
                continue
            others = [
                k2
                for pos, k2 in enumerate(it.inputs)
                if pos != i and hasattr(values.types[k2], "ndim")
            ]
            if any(mine & self.roots_of(k2) for k2 in others):
                continue
            # a graph output in memory the planner provides is C-contiguous
            memory = self.memory_layouts.get(vid)
            shape = tuple(
                None if s is None else int(s) for s in values.types[vid].shape
            )
            if (
                it.outputs[k] in self.graph_outputs
                and memory is not None
                and not memory.is_c(shape)
            ):
                continue
            granted[k] = i
        return granted

    def task(self, it) -> Task:
        """The task running an item."""
        if isinstance(it, UnitItem):
            return self.unit_task(it)
        in_layouts = [self.lay(k) for k in it.inputs]
        if isinstance(it, SubprogramItem):
            return self.kernel_task(it, self.resolver.subprogram(it.node, in_layouts))
        node = it.node
        types = self.values.types
        # constant scalar inputs are folded into the code
        folded = {
            pos: self.values.consts[k].item()
            for pos, (k, var) in enumerate(zip(it.inputs, node.inputs, strict=True))
            if k in self.values.consts and isinstance(var.type, ScalarType)
        }
        if any(isinstance(repr_of(o.type), ScalarRepr) for o in node.outputs) or any(
            isinstance(repr_of(types[k]), ScalarRepr)
            for pos, k in enumerate(it.inputs)
            if pos not in folded
        ):
            raise NotImplementedError("scalar node outside a unit")
        # A Rust kernel first: there is nothing to compile. Not for a Scan whose loop we
        # generate (its body is inlined into the loop), and not with folded constants
        # (a kernel would need them as arguments).
        scan = self.resolver.generates_scan(node)
        inplace = self.grant_inplace(it, folded)
        if not folded and not scan and not inplace:
            kernel = self.resolver.rust(node, in_layouts)
            if kernel is not None:
                return self.kernel_task(it, kernel)
        try:
            return self.generated_task(it, folded, in_layouts, inplace)
        except NotImplementedError:
            if scan:
                kernel = self.resolver.rust(node, in_layouts)
                if kernel is not None:
                    return self.kernel_task(it, kernel)
            if not self.opts.numba_fallback:
                raise
        return self.kernel_task(self.keep_destroyed(it), self.resolver.numba(node))

    def keep_destroyed(self, it):
        """A numba implementation overwrites the inputs its op destroys. That is what
        the graph asks for; where the planner can't be sure nothing else sees the
        memory (as in `grant_inplace`), the kernel gets a copy to overwrite."""
        values = self.values
        destroyed = {
            i for ins in getattr(it.node.op, "destroy_map", {}).values() for i in ins
        }
        inputs = list(it.inputs)
        for i in sorted(destroyed):
            vid = inputs[i]
            if not isinstance(values.types[vid], TensorType):
                continue
            safe = self.allow_inplace and it.of_the_graph and vid not in values.consts
            if safe and not any(root in values.shared for root in self.roots_of(vid)):
                continue
            inputs[i] = values.new(values.types[vid])
            self.copies_first.append(self.copy_task(vid, inputs[i]))
        return replace(it, inputs=inputs)

    def kernel_task(self, it, kernel: CalledKernel) -> Task:
        """A task that runs a kernel (Rust, numba, a subprogram)."""
        declared = kernel.outputs
        outs = self.provide(it.outputs, declared)
        self.set_layouts(outs, [o.layout for o in declared])
        outs, modes, aliases = _present_outputs(declared, outs)
        # (values without data aren't arguments)
        present = [
            k
            for k, var in zip(it.inputs, it.node.inputs, strict=True)
            if not is_absent(var.type)
        ]
        return Task(
            present,
            outs,
            modes,
            aliases,
            address=kernel.address,
            config=kernel.config,
            description=kernel.description,
            keepalive=kernel.keepalive,
        )

    def generated_task(self, it, folded, in_layouts, inplace=None) -> Task:
        """A task whose kernel is generated for the node. Raises NotImplementedError if
        no emitter handles it. `inplace`: {output: input} for the outputs that
        overwrite an input (`grant_inplace`)."""
        node = it.node
        body = (
            self.resolver.scan_body(node, in_layouts)
            if self.resolver.generates_scan(node)
            else None
        )
        module = ir.Module(
            f"node_{type(node.op).__name__}", fastmath=self.opts.fastmath
        )
        links = LinkTable()
        nf = build_node_function(
            module,
            node,
            "node",
            folded=folded,
            in_layouts=in_layouts,
            parallel=self.opts.parallel,
            links=links,
            scan_body=body,
            inplace=inplace,
        )
        build_kernel_wrapper(module, nf, "kernel", self.opts.check_layouts)
        declared = nf.signature.outputs
        skipped = skipped_inputs(node, folded)
        inputs = [k for pos, k in enumerate(it.inputs) if pos not in skipped]
        outs = self.provide(it.outputs, declared)
        self.set_layouts(outs, [o.layout for o in declared])
        for k, i in (inplace or {}).items():
            if it.inputs[i] in self.memory_layouts:
                self.memory_layouts[outs[k]] = self.memory_layouts[it.inputs[i]]
        if inplace:
            self.stats.inplace.append(str(node.op))
        outs, modes, aliases = _present_outputs(declared, outs)
        desc = str(node.op) + _layouts_note(
            [r for r in nf.signature.inputs if hasattr(r, "layout") and r.ndim > 0]
        )
        return Task(
            inputs,
            outs,
            modes,
            aliases,
            description=desc,
            module=module,
            config=links.address,
            keepalive=links.keepalive,
        )

    def unit_task(self, it: UnitItem) -> Task:
        """A task whose kernel is one function for all the items of a unit."""
        types = self.values.types
        out_types = [types[o] for _, o in it.outputs]
        modes = [
            OutputMode.PROVIDED if repr_of(t).static else OutputMode.RETURNED
            for t in out_types
        ]
        # provided memory is C-contiguous, and so is what a unit allocates
        out_layouts = [Layout.c(t.ndim) for t in out_types]
        in_layouts = unit_input_layouts(
            [types[k] for k in it.inputs], [self.lay(k) for k in it.inputs]
        )
        resolved = self.resolver.resolve_items(
            it.items,
            {
                k: lay
                for k, lay in zip(it.inputs, in_layouts, strict=True)
                if lay is not None
            },
        )
        module = ir.Module(f"unit_{len(it.items)}", fastmath=self.opts.fastmath)
        nf = build_unit_function(
            module,
            "unit",
            [(k, types[k]) for k in it.inputs],
            [
                (inner, t, mode)
                for (inner, _), t, mode in zip(
                    it.outputs, out_types, modes, strict=True
                )
            ],
            resolved,
            {
                k: c
                for k, c in self.values.constants_of(it.items).items()
                if self.values.folded(k)
            },
            input_layouts=in_layouts,
            output_layouts=out_layouts,
            parallel=self.opts.parallel,
        )
        build_kernel_wrapper(module, nf, "kernel", self.opts.check_layouts)
        outs = [o for _, o in it.outputs]
        self.set_layouts(outs, out_layouts)
        self.stats.n_units += 1
        names = ", ".join(str(s.node.op) for s in it.items if isinstance(s, NodeItem))[
            :120
        ]
        return Task(
            it.inputs,
            outs,
            modes,
            [()] * len(modes),
            description=f"unit[{len(it.items)}]: {names}",
            module=module,
            config=nf.links.address,
            keepalive=nf.links.keepalive,
        )

    def copy_task(self, src, dst) -> Task:
        t = self.values.types[dst]
        node = _copy_node(t)
        self.layouts[dst] = Layout.c(t.ndim)
        self.memory_layouts[dst] = Layout.c(t.ndim)
        k = rust_kernel(node.op, node, [self.lay(src)])
        mode = rust_kernel_outputs(k, node, [self.lay(src)])[0].mode
        return Task(
            [src],
            [dst],
            [mode],
            [()],
            address=k.address,
            config=k.config,
            description="copy",
            keepalive=k,
        )

    # ------------------------------------------------------------ the program

    def plan(self, fgraph, output_modes=None, input_layouts=None):
        """`input_layouts`: what the caller guarantees about each input's layout (default:
        from `Options.input_layout`)."""
        token = _current_options.set(self.opts)
        parent = _current_stats.get()
        stats_token = _current_stats.set(self.stats)
        try:
            prog = self._plan(fgraph, output_modes, input_layouts)
        finally:
            _current_stats.reset(stats_token)
            _current_options.reset(token)
        if parent is not None:
            parent.modules.extend(self.stats.modules)
        return prog

    def _plan(self, fgraph, output_modes, input_layouts):
        t0 = time.perf_counter()
        values = self.values
        scope = {}
        in_ids = []
        scalar_inputs = []
        for pos, v in enumerate(fgraph.inputs):
            if isinstance(v.type, ScalarType):
                # scalars don't cross the kernel ABI: the input is a 0-d array, read into the
                # scalar by a ScalarFromTensor (which joins the scalars' compiled unit)
                t = values.new(TensorType(v.type.dtype, shape=()))
                k = values.new(v.type)
                scalar_inputs.append((t, k))
                in_ids.append(t)
                self.layouts[t] = Layout.c(0)
                scope[v] = k
                continue
            k = values.new(v.type)
            scope[v] = k
            in_ids.append(k)
            if hasattr(v.type, "ndim"):
                lay = input_layouts[pos] if input_layouts is not None else None
                if lay is None:
                    lay = (
                        Layout.c(v.type.ndim)
                        if self.opts.input_layout == "c"
                        else Layout.dynamic(v.type.ndim)
                    )
                self.layouts[k] = lay
        self.input_contiguous = [
            input_layouts is None
            and self.opts.input_layout == "c"
            and hasattr(v.type, "ndim")
            for v in fgraph.inputs
        ]

        # 1. items
        items = [
            NodeItem(scalar_from_tensor(values.types[t]()).owner, [t], [k])
            for t, k in scalar_inputs
        ]
        # (the order of a graph with a DestroyHandler respects what its ops destroy)
        self.allow_inplace = hasattr(fgraph, "destroyers")
        items += flatten(values, fgraph.toposort(), scope, of_the_graph=True)
        out_ids = [values.of(o, scope) for o in fgraph.outputs]
        self.graph_outputs = set(out_ids)
        items = group_scalars(values, items, set(out_ids))

        # 2. tasks: implementations, generated code, layouts
        tasks = self.tasks_for(items)
        self.output_layouts = [self.lay(k) for k in out_ids]
        t1 = time.perf_counter()

        # 3. compile all generated modules at once
        generated = [s for s in tasks if s.module is not None]
        for s in generated:
            s.module.label = "ptl:" + " ".join(s.description.split())[:160]  # perf maps
        if generated:
            compiled = ir.compile_many(
                [s.module for s in generated],
                opt_level=self.opts.opt_level,
                cache=self.opts.cache,
            )
            for s, c in zip(generated, compiled, strict=True):
                s.address = c.address("kernel")
                s.keepalive = (s.keepalive, c)  # (the links table, if any)
                s.module = None
                self.stats.modules.append(GeneratedModule(s.description, c))
        self.stats.n_modules = len(generated)
        t2 = time.perf_counter()

        # 4. and 5. memory, and the program
        # values without data (random generators) aren't inputs or outputs of the program
        self.program_inputs = [
            pos for pos, v in enumerate(fgraph.inputs) if not is_absent(v.type)
        ]
        self.program_outputs = [
            pos for pos, v in enumerate(fgraph.outputs) if not is_absent(v.type)
        ]
        modes = output_modes or [OutputMode.RETURNED] * len(out_ids)
        self.input_contiguous = [
            self.input_contiguous[pos] for pos in self.program_inputs
        ]
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

    def build(self, tasks, in_ids, out_ids, output_modes):
        """The program for the interpreter: the values with their storage, the tasks
        with what to release after each."""
        interp = _interp()
        types, consts = self.values.types, self.values.consts
        memory = plan_memory(self.values, tasks, in_ids, out_ids)
        pb = interp.ProgramBuilder()
        ids = {}

        def dtype(k):
            return np.dtype(types[k].dtype)

        def shape(k):
            return [None if s is None else int(s) for s in types[k].shape]

        def order(k):
            lay = self.memory_layouts.get(k)
            return None if lay is None else [int(a) for a in lay.order]

        for k, contiguous in zip(in_ids, self.input_contiguous, strict=True):
            ids[k] = pb.input(dtype(k), shape(k), contiguous)
        for k in memory.in_order():
            if k in ids:
                continue
            what = memory.kind[k]
            if what == "const":
                ids[k] = pb.constant(
                    np.asarray(consts[k], dtype=types[k].dtype)
                )  # stored C-contiguous
            elif what == "provided":
                if k in memory.first_out:
                    ids[k] = pb.fresh(dtype(k), shape(k))
                else:
                    ids[k] = pb.arena(dtype(k), shape(k), memory.offsets[k], order(k))
            else:
                ids[k] = pb.returned(
                    dtype(k), shape(k), [ids[a] for a in memory.aliases[k]]
                )
        for k, source in memory.same.items():
            ids[k] = ids[source]  # (overwritten in place: the same value)
        release = memory.release_points(len(tasks))
        for s, after in zip(tasks, release, strict=True):
            pb.task(
                s.address,
                s.config,
                [ids[k] for k in s.inputs],
                [ids[k] for k in s.outputs],
                [ids[k] for k in after],
                s.description,
                s.keepalive,
            )
        pb.set_arena_bytes(memory.arena_bytes)
        modes = [
            interp.OutputMode.Provided
            if m is OutputMode.PROVIDED
            else interp.OutputMode.Allocated
            for m in output_modes
        ]
        return pb.build([ids[k] for k in out_ids], modes)


def _present_outputs(declared, vids):
    """(vids, modes, aliases) of the outputs that have a value. Random generators have
    none: they aren't task outputs."""
    keep = [j for j, o in enumerate(declared) if o.mode is not OutputMode.ABSENT]
    return (
        [vids[j] for j in keep],
        [declared[j].mode for j in keep],
        [declared[j].aliases for j in keep],
    )


def _layouts_note(reprs):
    """` {C, [?, P0]}`: the input layouts a generated task is specialized for."""

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
    if planner.program_inputs == list(range(n_in)) and planner.program_outputs == list(
        range(n_out)
    ):
        return prog, planner.stats
    ins, outs = planner.program_inputs, planner.program_outputs
    sources = {
        pos: _generator_source(fgraph, o)
        for pos, o in enumerate(fgraph.outputs)
        if pos not in outs
    }

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
        rng_inputs = [
            i for i in node.inputs if is_absent(i.type) and not _is_none_type(i.type)
        ]
        if not rng_inputs:
            return None
        var = rng_inputs[0]
    return fgraph.inputs.index(var) if var in fgraph.inputs else None


def _is_none_type(t):
    from pytensor.tensor.type_other import NoneTypeT

    return isinstance(t, NoneTypeT)
