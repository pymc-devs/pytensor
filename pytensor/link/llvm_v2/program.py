"""Planning programs: a FunctionGraph becomes a list of steps for the Rust interpreter.

1. Flatten: OpFromGraphs without a `unit` are inlined (with their own variable scope);
   OpFromGraphs with `unit=COMPILED` become compiled units; `unit=INTERPRETED` become
   nested programs. Nodes touching ScalarType values are grouped into implicit compiled
   units, since scalars never cross the C ABI.
2. Each remaining node runs a precompiled Rust kernel (`llvm_kernel`), generated code
   (`llvm_emit`), or its numba implementation.
3. Output modes decide where values live: provided outputs are arena temporaries (or
   fresh buffers for graph outputs), allocated outputs are released after their last
   use, views keep their source alive. Arena offsets (also for step workspaces) are
   assigned here by liveness.
4. Generated modules are compiled in one parallel batch.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field

import numpy as np

from pytensor.compile.builders import OpFromGraph, UnitKind
from pytensor.graph.basic import Constant
import pytensor.link.llvm_v2.codegen  # noqa: F401 (registers the emitters)
from pytensor.link.llvm_v2.codegen.fused import fused_spec
from pytensor.link.llvm_v2.codegen.function import (
    OutMode,
    build_abi_wrapper,
    build_node_function,
    node_outputs,
)
from pytensor.link.llvm_v2.codegen.function import skipped_inputs
from pytensor.link.llvm_v2.codegen.types import ScalarRepr, is_absent, repr_of
from pytensor.link.llvm_v2.codegen.unit import Item, build_unit_function
from pytensor.link.llvm_v2.kernels import llvm_kernel
from pytensor.link.llvm_v2.layout import Layout
from pytensor.scalar.basic import ScalarType


def _ir():
    from pytensor_llvm import ir

    return ir


def _rt():
    from pytensor_llvm import rt

    return rt


@dataclass
class Options:
    fastmath: bool = True
    opt_level: int = 3
    cache: bool = True
    numba_fallback: bool = True
    #: what generated code may assume about the graph inputs: "any" strides, or "c"
    #: (C-contiguous: Python calls copy other arrays, the C API rejects them)
    input_layout: str = "any"
    #: generated steps verify the layouts their signatures claim (for testing; also
    #: turned on by PYTENSOR_LLVM_CHECK_LAYOUTS=1)
    check_layouts: bool = False

    def __post_init__(self):
        import os

        if os.environ.get("PYTENSOR_LLVM_CHECK_LAYOUTS", "0") not in ("", "0"):
            self.check_layouts = True


#: the options of the program being planned (nested programs, e.g. Scan's step, use them)
_current_options: contextvars.ContextVar = contextvars.ContextVar("llvm_v2_options", default=None)


def current_options() -> Options:
    return _current_options.get() or Options()


@dataclass
class UnitItem:
    """Several items compiled as one function."""

    items: list
    inputs: list  # keys read from outside
    outputs: list  # (inner key, outer key)
    description: str


@dataclass
class ProgramItem:
    """An interpreted OpFromGraph: a nested program."""

    node: object
    inputs: list
    outputs: list


@dataclass
class Step:
    inputs: list
    outputs: list
    modes: list  # OutMode per output
    view_of: list  # input index per output (VIEW), else None
    abi: str = "standard"
    address: int = 0
    env: int = 0
    workspace: int = 0
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
    n_kernels: int = 0
    n_foreign: int = 0
    n_units: int = 0
    n_nested: int = 0
    foreign: list = field(default_factory=list)


class Planner:
    def __init__(self, options: Options):
        self.opts = options
        self.types = {}  # key -> PyTensor type
        self.consts = {}  # key -> numpy data
        self._next = 0
        self.stats = Stats()
        self.layouts = {}  # key -> Layout of tensor values (see layout.py)
        #: key -> packed layout of the memory the planner provides (arena, fresh)
        self.memory_layouts = {}
        self.graph_outputs = set()
        #: layouts of the program's outputs (after `plan`)
        self.output_layouts = []

    # ------------------------------------------------------------ keys and flattening

    def new_key(self, t):
        k = self._next
        self._next += 1
        self.types[k] = t
        return k

    def key_of(self, var, scope):
        if var in scope:
            return scope[var]
        if is_absent(var.type):
            # None inputs have no value; nothing ever reads this key
            k = self.new_key(var.type)
            scope[var] = k
            return k
        if isinstance(var, Constant):
            k = self.new_key(var.type)
            self.consts[k] = np.asarray(var.data)
            scope[var] = k
            return k
        raise ValueError(f"variable {var} used before it is defined")

    def flatten(self, nodes, scope, inside_unit=False):
        items = []
        for node in nodes:
            op = node.op
            unit = getattr(op, "unit", None) if isinstance(op, OpFromGraph) else None
            if isinstance(op, OpFromGraph) and unit is None and fusable(op, node):
                # a FusedElemwise we can generate as one loop: keep it as one node
                ins = [self.key_of(i, scope) for i in node.inputs]
                outs = [self.new_key(o.type) for o in node.outputs]
                scope.update(zip(node.outputs, outs, strict=True))
                items.append(Item(node, ins, outs))
            elif isinstance(op, OpFromGraph) and (unit is None or (inside_unit and unit is UnitKind.COMPILED)):
                inner = op.fgraph
                sub = {iv: self.key_of(ov, scope) for iv, ov in zip(inner.inputs, node.inputs, strict=True)}
                items += self.flatten(inner.toposort(), sub, inside_unit)
                for io, oo in zip(inner.outputs, node.outputs, strict=True):
                    scope[oo] = self.key_of(io, sub)
            elif isinstance(op, OpFromGraph) and unit is UnitKind.COMPILED:
                inner = op.fgraph
                sub = {iv: self.key_of(ov, scope) for iv, ov in zip(inner.inputs, node.inputs, strict=True)}
                inner_scope = dict(sub)
                try:
                    inner_items = self.flatten(inner.toposort(), inner_scope, inside_unit=True)
                except NotImplementedError:
                    # can't be one unit (e.g. it contains an interpreted OpFromGraph):
                    # run its nodes as steps
                    items += self.flatten(inner.toposort(), sub, inside_unit)
                    for io, oo in zip(inner.outputs, node.outputs, strict=True):
                        scope[oo] = self.key_of(io, sub)
                    continue
                sub = inner_scope
                outs = []
                for io, oo in zip(inner.outputs, node.outputs, strict=True):
                    k = self.new_key(oo.type)
                    scope[oo] = k
                    outs.append((self.key_of(io, sub), k))
                items.append(UnitItem(inner_items, self._unit_inputs(inner_items), outs, str(op)))
            elif isinstance(op, OpFromGraph) and unit is UnitKind.INTERPRETED:
                if inside_unit:
                    raise NotImplementedError("an interpreted OpFromGraph inside a compiled unit")
                ins = [self.key_of(i, scope) for i in node.inputs]
                outs = [self.new_key(o.type) for o in node.outputs]
                scope.update(zip(node.outputs, outs, strict=True))
                items.append(ProgramItem(node, ins, outs))
            else:
                ins = [self.key_of(i, scope) for i in node.inputs]
                outs = [self.new_key(o.type) for o in node.outputs]
                scope.update(zip(node.outputs, outs, strict=True))
                items.append(Item(node, ins, outs))
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

    def set_layouts(self, keys, layouts):
        for k, lay in zip(keys, layouts, strict=True):
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
            return isinstance(it, Item) and any(
                k in is_scalar and k not in self.consts for k in (*it.inputs, *it.outputs)
            )

        if not any(touches(it) for it in items):
            return items
        # union items connected through scalar values
        parent = list(range(len(items)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        producer = {}
        for i, it in enumerate(items):
            for k in getattr(it, "outputs", []):
                producer[k[1] if isinstance(k, tuple) else k] = i
        for i, it in enumerate(items):
            if not touches(it):
                continue
            for k in (*it.inputs, *it.outputs):
                if k in is_scalar:
                    if k in producer:
                        parent[find(i)] = find(producer[k])
                    elif k not in self.consts:
                        raise NotImplementedError("a ScalarType graph input")
        groups = {}
        for i, it in enumerate(items):
            if touches(it):
                groups.setdefault(find(i), []).append(i)
        members = {i: root for root, idx in groups.items() for i in idx}
        out = []
        done = set()
        consumers = {}
        for i, it in enumerate(items):
            for k in it.inputs:
                consumers.setdefault(k, set()).add(i)
        for i, it in enumerate(items):
            root = members.get(i)
            if root is None:
                out.append(it)
                continue
            if root in done:
                continue
            done.add(root)
            idx = groups[root]
            # convex? every item between the group's first and last that depends on
            # the group and feeds it would have to be inside
            sub = [items[j] for j in idx]
            produced = {k for s in sub for k in s.outputs}
            outputs = []
            for s in sub:
                for k in s.outputs:
                    outside = any(c not in idx for c in consumers.get(k, ())) or k in graph_outputs
                    if outside:
                        if k in is_scalar:
                            raise NotImplementedError("a ScalarType value used outside its compiled unit")
                        outputs.append((k, k))
            for j in range(min(idx), max(idx) + 1):
                if j not in idx and any(k in produced for k in items[j].inputs):
                    for k in getattr(items[j], "outputs", []):
                        kk = k[1] if isinstance(k, tuple) else k
                        if any(c in idx for c in consumers.get(kk, ())):
                            raise NotImplementedError("scalar operations interleaved with other nodes")
            out.append(UnitItem(sub, self._unit_inputs(sub), outputs, "scalars"))
        return toposort_items(out)

    # ------------------------------------------------------------ steps

    def provided_layout(self, key, out):
        """The memory the planner provides for output `key` declared as `out`: what the
        op asks for if it is packed (arena memory can have any packed order), C for
        graph outputs (callers get, and the C API provides, C-contiguous memory)."""
        if key in self.graph_outputs or not out.layout.fully_packed(out.repr.shape):
            return Layout.c(out.repr.ndim)
        return out.layout

    def step_for_item(self, it):
        ir = _ir()
        if isinstance(it, UnitItem):
            modes = [OutMode.PROVIDED if repr_of(self.types[o]).static else OutMode.ALLOCATED for _, o in it.outputs]
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
            )
            build_abi_wrapper(module, nf, "abi", self.opts.check_layouts)
            # provided memory is C-contiguous, and so is what a unit allocates
            self.set_layouts([o for _, o in it.outputs], [Layout.c(self.types[o].ndim) for _, o in it.outputs])
            self.stats.n_units += 1
            return Step(it.inputs, [o for _, o in it.outputs], modes, [None] * len(modes), workspace=nf.workspace,
                        description=f"unit[{len(it.items)}]: " + ", ".join(str(s.node.op) for s in it.items if isinstance(s, Item))[:120],
                        module=module)
        if isinstance(it, ProgramItem):
            node = it.node
            modes = [OutMode.PROVIDED if repr_of(o.type).static else OutMode.ALLOCATED for o in node.outputs]
            sub = Planner(self.opts)
            prog = sub.plan(node.op.fgraph, output_modes=modes, input_layouts=[self.lay(k) for k in it.inputs])
            # an allocated output is the inner value if it owns a buffer, else a C copy
            self.set_layouts(
                it.outputs,
                [
                    Layout.c(o.type.ndim) if m is OutMode.PROVIDED or lay is None else lay.join(Layout.c(o.type.ndim), _shape(o.type), _isz(o.type))
                    for o, m, lay in zip(node.outputs, modes, sub.output_layouts, strict=True)
                ],
            )
            self.stats.n_nested += 1
            return Step(it.inputs, it.outputs, modes, [None] * len(modes), address=prog.address, env=prog.env,
                        description=f"program: {node.op}", keepalive=prog)
        node = it.node
        # constant scalar inputs are folded into the code
        folded = {
            pos: self.consts[k].item()
            for pos, (k, var) in enumerate(zip(it.inputs, node.inputs, strict=True))
            if k in self.consts and isinstance(var.type, ScalarType)
        }
        if any(isinstance(repr_of(o.type), ScalarRepr) for o in node.outputs) or any(
            isinstance(repr_of(i.type), ScalarRepr) for pos, i in enumerate(node.inputs) if pos not in folded
        ):
            raise NotImplementedError("scalar node outside a unit")
        present = [k for k, var in zip(it.inputs, node.inputs, strict=True) if not is_absent(var.type)]
        in_lays = [self.lay(k) for k in it.inputs]
        declared = node_outputs(node, in_lays)
        modes = [o.mode for o in declared]
        views = [o.view_of for o in declared]
        provided = [
            self.provided_layout(key, o) if o.mode is OutMode.PROVIDED else None
            for key, o in zip(it.outputs, declared, strict=True)
        ]
        for key, lay in zip(it.outputs, provided, strict=True):
            if lay is not None:
                self.memory_layouts[key] = lay
        k = None if folded else llvm_kernel(node.op, node, modes, in_lays)
        if k is not None:
            self.stats.n_kernels += 1
            # kernels write provided memory as it is and allocate C-contiguous
            self.set_layouts(
                it.outputs,
                [
                    provided[j] if o.mode is OutMode.PROVIDED
                    else Layout.c(o.repr.ndim) if o.mode is OutMode.ALLOCATED
                    else o.layout
                    for j, o in enumerate(declared)
                ],
            )
            return Step(present, it.outputs, modes, views, address=k.address, env=k.env, description=repr(k), keepalive=k)
        try:
            module = ir.Module(f"node_{type(node.op).__name__}", fastmath=self.opts.fastmath)
            nf = build_node_function(module, node, "node", folded=folded, in_layouts=in_lays, provided=provided)
            build_abi_wrapper(module, nf, "abi", self.opts.check_layouts)
            skipped = skipped_inputs(node, folded)
            inputs = [k for pos, k in enumerate(it.inputs) if pos not in skipped]
            # view_of counts the step's inputs
            vmap = [None if v is None else v - sum(1 for p in skipped if p < v) for v in views]
            self.set_layouts(it.outputs, [o.layout for o in nf.signature.outputs])
            desc = str(node.op) + _layouts_note([r for r in nf.signature.inputs if hasattr(r, "layout") and r.ndim > 0])
            return Step(inputs, it.outputs, modes, vmap, workspace=nf.workspace, description=desc, module=module)
        except NotImplementedError:
            if not self.opts.numba_fallback:
                raise
        return self.foreign_step(it)

    def foreign_step(self, it):
        from pytensor.link.llvm.numba_fallback import numba_fallback_kernel

        node = it.node
        kernel = numba_fallback_kernel(node)
        self.stats.n_foreign += 1
        self.stats.foreign.append(str(node.op))
        n = len(node.outputs)
        # numba may return any layout
        self.set_layouts(it.outputs, [Layout.dynamic(o.type.ndim) if hasattr(o.type, "ndim") else None for o in node.outputs])
        return Step(it.inputs, it.outputs, [OutMode.ALLOCATED] * n, [None] * n, abi="foreign", address=kernel.address,
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
        in_keys = []
        for pos, v in enumerate(fgraph.inputs):
            k = self.new_key(v.type)
            scope[v] = k
            in_keys.append(k)
            if hasattr(v.type, "ndim"):
                lay = input_layouts[pos] if input_layouts is not None else None
                if lay is None:
                    lay = Layout.c(v.type.ndim) if self.opts.input_layout == "c" else Layout.dynamic(v.type.ndim)
                self.layouts[k] = lay
        self.input_contiguous = [
            input_layouts is None and self.opts.input_layout == "c" and hasattr(v.type, "ndim") for v in fgraph.inputs
        ]
        items = self.flatten(fgraph.toposort(), scope)
        out_keys = [self.key_of(o, scope) for o in fgraph.outputs]
        self.graph_outputs = set(out_keys)
        items = self.group_scalars(items, set(out_keys))
        steps = self.steps_for(items)
        self.output_layouts = [self.lay(k) for k in out_keys]
        t1 = time.perf_counter()

        # compile all generated modules at once
        modules = []
        for s in steps:
            if s.module is not None:
                s.module.label = "ptl:" + " ".join(s.description.split())[:160]  # perf maps
                modules.append(s.module)
        if modules:
            compiled = _ir().compile_many(modules, opt_level=self.opts.opt_level, cache=self.opts.cache)
            for s, c in zip([s for s in steps if s.module is not None], compiled, strict=True):
                s.address = c.address("abi")
                s.keepalive = c
                s.module = None
        self.stats.n_modules = len(modules)
        t2 = time.perf_counter()

        prog = self.build(steps, in_keys, out_keys, output_modes or [OutMode.ALLOCATED] * len(out_keys))
        self.stats.codegen_s = t1 - t0
        self.stats.compile_s = t2 - t1
        self.stats.plan_s = time.perf_counter() - t2
        return prog

    def steps_for(self, items):
        steps = []
        for it in items:
            try:
                steps.append(self.step_for_item(it))
            except NotImplementedError:
                if not isinstance(it, UnitItem) or it.description == "scalars":
                    raise
                # a compiled OpFromGraph we can't generate: run its nodes as steps, and
                # copy each result into the value the OpFromGraph's output is
                inner_outs = {inner for inner, _ in it.outputs}
                steps += self.steps_for(self.group_scalars(it.items, inner_outs))
                for inner, outer in it.outputs:
                    steps.append(self.copy_step(inner, outer))
        return steps

    def copy_step(self, src, dst):
        node = _copy_node(self.types[dst])
        self.layouts[dst] = Layout.c(self.types[dst].ndim)
        self.memory_layouts[dst] = Layout.c(self.types[dst].ndim)
        mode = node_outputs(node)[0].mode
        k = llvm_kernel(node.op, node, [mode], [self.lay(src)])
        return Step([src], [dst], [mode], [None], address=k.address, env=k.env, description="copy", keepalive=k)

    def build(self, steps, in_keys, out_keys, output_modes):
        rt = _rt()
        n_steps = len(steps)
        INF = n_steps + 1
        kind = {}  # key -> ("input"|"const"|"provided"|"allocated"|"view", source)
        for k in in_keys:
            kind[k] = ("input", None)
        for s in steps:
            for k in s.inputs:
                if k in self.consts and k not in kind:
                    kind[k] = ("const", None)
        defined, end = {}, {}
        for k in kind:
            defined[k] = -1
        for i, s in enumerate(steps):
            for k in s.inputs:
                end[k] = max(end.get(k, -1), i)
            for j, (k, mode) in enumerate(zip(s.outputs, s.modes, strict=True)):
                defined[k] = i
                end.setdefault(k, i)
                if mode is OutMode.VIEW:
                    kind[k] = ("view", s.inputs[s.view_of[j]])
                elif mode is OutMode.PROVIDED:
                    kind[k] = ("provided", None)
                else:
                    kind[k] = ("allocated", None)
        for k in out_keys:
            if k in self.consts and k not in kind:
                kind[k] = ("const", None)
                defined[k] = -1
            end[k] = INF
        # views keep their source alive (later values first, so chains propagate)
        for k in sorted(kind, key=lambda k: defined[k], reverse=True):
            what, src = kind[k]
            if what == "view":
                end[src] = max(end.get(src, -1), end[k])

        # arena: static provided temporaries and step workspaces, first fit by liveness
        intervals = []
        first_out = {}
        for o, k in enumerate(out_keys):
            first_out.setdefault(k, o)
        for k, (what, _) in kind.items():
            if what == "provided" and k not in first_out:
                t = self.types[k]
                nbytes = int(np.prod(t.shape, dtype=np.int64)) * np.dtype(t.dtype).itemsize
                intervals.append((defined[k], end[k], nbytes, ("value", k)))
        for i, s in enumerate(steps):
            if s.workspace:
                intervals.append((i, i, s.workspace, ("ws", i)))
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

        pb = rt.ProgramBuilder()
        ids = {}

        def dtype(k):
            return np.dtype(self.types[k].dtype)

        def shape(k):
            t = self.types[k]
            return [None if s is None else int(s) for s in t.shape]

        for k, contiguous in zip(in_keys, self.input_contiguous, strict=True):
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
        release = [[] for _ in steps]
        for k, (what, _) in kind.items():
            if what in ("allocated", "view", "provided") and end.get(k, INF) < INF:
                release[max(end[k], defined[k])].append(ids[k])
        for i, s in enumerate(steps):
            pb.step(
                s.address,
                s.env,
                rt.Abi.Foreign if s.abi == "foreign" else rt.Abi.Standard,
                [ids[k] for k in s.inputs],
                [ids[k] for k in s.outputs],
                release[i],
                offsets.get(("ws", i)),
                s.description,
                s.keepalive,
            )
        pb.set_arena_bytes(arena)
        modes = [rt.OutputMode.Provided if m is OutMode.PROVIDED else rt.OutputMode.Allocated for m in output_modes]
        return pb.build([ids[k] for k in out_keys], modes)


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


def _shape(t):
    return tuple(None if s is None else int(s) for s in t.shape)


def _isz(t):
    return np.dtype(t.dtype).itemsize


def _layouts_note(reprs):
    """` {C, [?, P0]}`: the input layouts a generated step is specialized for."""

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
    """Plan and compile `fgraph`; returns (program, stats)."""
    planner = Planner(options or Options())
    prog = planner.plan(fgraph)
    return prog, planner.stats
