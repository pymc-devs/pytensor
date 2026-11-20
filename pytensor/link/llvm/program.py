"""Planning programs: a FunctionGraph becomes one generated function.

`Planner.plan` runs the stages of `pytensor.link.llvm.plan` and does what lies between
them:

1. Flatten (`plan/flatten.py`): the graph becomes items. OpFromGraphs are inlined,
   become units (`execution=COMPILED`) or subprograms (`execution=INTERPRETED`).
   ScalarType values never cross the kernel ABI: the `llvm_units` rewrite keeps them
   inside units, and a scalar input arrives as a 0-d array.
2. Each item is resolved (`Planner.resolve_item`; `plan/resolve.py`): a Rust kernel, a
   numba kernel, a subprogram, or code generated for it, compiled on its own (a node,
   a unit): each is called by the program's function. The layouts of the values are
   not decided here: the graph's `LayoutFeature` has them (`layout.py`), and what runs
   an item must give its outputs those layouts.
3. The program's function is generated like a unit's (`codegen/unit.py`): it calls the
   items in order, its temporaries in one arena laid out by liveness
   (`plan/memory.py`).
4. All generated modules are compiled in one parallel batch, and the function becomes
   a `Program`.
"""

from __future__ import annotations

import contextvars
import re
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

import pytensor.link.llvm.codegen  # noqa: F401 (registers the emitters)
from pytensor.graph.features import view_feature
from pytensor.link.llvm.codegen.call import CalledKernel, LinkTable
from pytensor.link.llvm.codegen.function import (
    OutputMode,
    build_kernel_wrapper,
    build_node_function,
    inplace_outputs,
    node_outputs,
    node_signature,
    skipped_inputs,
)
from pytensor.link.llvm.codegen.types import ScalarRepr, ir, is_absent, repr_of
from pytensor.link.llvm.codegen.unit import (
    PROFILE_CALLS,
    PROFILE_ITEMS,
    PROFILE_ON,
    build_unit_function,
    unit_input_layouts,
)
from pytensor.link.llvm.layout import (
    Layout,
    infer_layouts,
    layout_feature,
)
from pytensor.link.llvm.plan.flatten import flatten
from pytensor.link.llvm.plan.resolve import (
    Call,
    Resolved,
    Resolver,
    check_conforms,
)
from pytensor.link.llvm.plan.values import NodeItem, SubprogramItem, UnitItem, Values
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.basic import TensorFromScalar


if TYPE_CHECKING:
    from pytensor_llvm.ir import CompiledModule


def _interp():
    from pytensor_llvm import interp

    return interp


@dataclass
class Options:
    fastmath: bool = True
    opt_level: int = 3
    numba_fallback: bool = True
    #: what generated code may assume about the graph inputs: "any" strides, or "c"
    #: (C-contiguous: Python calls copy other arrays, the C API rejects them)
    input_layout: str = "any"
    #: may generated loops run in parallel (chunks on the runtime's thread pool, when
    #: there is enough work; see work.py)
    parallel: bool = True
    #: may Dot and the linear algebra kernels use faer's parallelism, on the runtime's
    #: thread pool (for large enough inputs, when the session allows parallelism; see
    #: `faer_kernels!` in linalg.rs). Results can differ from the sequential ones in the
    #: last bits.
    linalg_parallel: bool = False
    #: generated tasks verify the layouts their signatures claim (for testing; also
    #: turned on by PYTENSOR_LLVM_CHECK_LAYOUTS=1)
    check_layouts: bool = False
    #: give the generated code debug information: every instruction is located at the
    #: Python code that emitted it, in a function named after its node (for remarks,
    #: perf, debuggers; also turned on by PYTENSOR_LLVM_DEBUG_INFO=1)
    debug_info: bool = False

    def __post_init__(self):
        import os

        if os.environ.get("PYTENSOR_LLVM_CHECK_LAYOUTS", "0") not in ("", "0"):
            self.check_layouts = True
        if os.environ.get("PYTENSOR_LLVM_DEBUG_INFO", "0") not in ("", "0"):
            self.debug_info = True


#: the options of the program being planned (subprograms use them)
_current_options: contextvars.ContextVar = contextvars.ContextVar(
    "llvm_options", default=None
)


def current_options() -> Options:
    return _current_options.get() or Options()


#: the stats of the program being planned: the planner of a subprogram hands its
#: modules to it
_current_stats: contextvars.ContextVar = contextvars.ContextVar(
    "llvm_stats", default=None
)


_DEBUG_METADATA = re.compile(r"^!(\d+) = (?:distinct )?!(DI\w+)\((.*)\)$")
_FIELD = re.compile(r'(\w+): ("(?:[^"\\]|\\.)*"|[^,]+)')
_DBG = re.compile(r", !dbg !(\d+)")


def with_locations(text: str) -> str:
    """LLVM IR `text` of a module with debug information (`Options.debug_info`), with
    where every instruction comes from as a comment above it (once for a run of
    instructions from the same place): the Python frames that emitted it, innermost
    first ("file:line function"), then the function (node) it was emitted for. The
    debug metadata itself is left out."""
    md = {}
    for line in text.splitlines():
        if m := _DEBUG_METADATA.match(line):
            md[m[1]] = dict(_FIELD.findall(m[3]))

    def node(ref, field):
        """`field` of the metadata node `ref` ("!12"), unquoted."""
        return md.get(ref.lstrip("!"), {}).get(field, "").strip('"')

    cache = {}

    def where(ref):
        if ref not in cache:
            parts = []
            loc = md.get(ref)
            while loc is not None:
                scope = loc.get("scope", "")
                at = loc.get("inlinedAt")
                if at is None:  # the function the code is in
                    parts.append(node(scope, "name"))
                    break
                if node(scope, "line"):  # a Python function
                    file = node(
                        md.get(scope.lstrip("!"), {}).get("file", ""), "filename"
                    )
                    line = loc.get("line", "0")
                    parts.append(f"{file}:{line} {node(scope, 'name')}")
                else:  # a generated function, inlined: its node
                    parts.append(node(scope, "name"))
                loc = md.get(at.lstrip("!"))
            cache[ref] = " < ".join(parts)
        return cache[ref]

    out = []
    last = None
    for line in text.splitlines():
        if _DEBUG_METADATA.match(line) or line.startswith("!llvm.dbg.cu"):
            continue
        if m := _DBG.search(line):
            here = where(m[1])
            if here != last:
                out.append(f"  ; {here}")
                last = here
            line = line[: m.start()] + line[m.end() :]
        elif not line.startswith("  "):
            last = None
            line = re.sub(r" !dbg !\d+", "", line)
        out.append(line)
    return "\n".join(out)


@dataclass
class GeneratedModule:
    """A module the planner generated (for an item, or the program's function),
    compiled."""

    description: str
    compiled: CompiledModule

    @property
    def ir(self) -> str:
        """The LLVM IR as generated."""
        return self.compiled.ir

    def optimized_ir(
        self,
        opt_level: int | None = None,
        remarks: bool = False,
        locations: bool = False,
    ) -> str:
        """The LLVM IR after optimization. `remarks`: start with LLVM's optimization
        remarks (what the loop vectorizer did or why not, inlining, unrolling...), as
        comments, each after the instruction of `ir` it is about. `locations`: with
        debug information (`debug_info`), the Python code and node every instruction
        comes from, as comments (`with_locations`)."""
        text = self.compiled.optimized_ir(opt_level, remarks)
        return with_locations(text) if locations else text


@dataclass
class Stats:
    #: planning, with the code generated for the items
    plan_s: float = 0.0
    #: generating the program's function
    codegen_s: float = 0.0
    compile_s: float = 0.0
    n_modules: int = 0
    n_rust_kernels: int = 0
    n_numba: int = 0
    n_units: int = 0
    n_subprograms: int = 0
    numba: list = field(default_factory=list)
    #: the ops of the items that overwrite an input
    inplace: list = field(default_factory=list)
    #: the generated modules (GeneratedModule), including those of subprograms
    #: planned for this program
    modules: list = field(default_factory=list)

    def ir(
        self,
        optimized: bool = False,
        opt_level: int | None = None,
        remarks=False,
        locations=False,
    ) -> str:
        """The LLVM IR of all generated modules, each under a comment naming its item.
        `optimized`: after the optimization pipeline instead of as generated, with the
        optimization remarks first if `remarks` (`GeneratedModule.optimized_ir`).
        `locations`: with the location of every instruction (`with_locations`)."""
        parts = []
        for m in self.modules:
            text = m.optimized_ir(opt_level, remarks) if optimized else m.ir
            if locations:
                text = with_locations(text)
            parts.append(f"; ===== {' '.join(m.description.split())}\n{text}")
        return "\n".join(parts)


class Planner:
    def __init__(self, options: Options):
        self.opts = options
        self.stats = Stats()
        self.values = Values()
        self.resolver = Resolver(options, self.stats, self.values)
        #: vid -> Layout of tensor values: of the graph's variables, what its
        #: LayoutFeature says; of others (inside inlined OpFromGraphs), what
        #: `node_outputs` gives them, as the feature does
        self.layouts = {}
        #: modules generated for items, to compile with the program's function
        self.modules = []  # (module, CalledKernel whose address is its "kernel")
        #: layouts of the program's outputs (after `plan`)
        self.output_layouts = []
        #: may items overwrite inputs their ops destroy? Only for a graph whose
        #: DestroyHandler has validated the order of its nodes (`plan`)
        self.allow_inplace = False
        #: the graph being planned, and its `ViewFeature`
        self.fgraph = None
        self.views = None
        #: memory of the graph that views the DestroyHandler doesn't see may lie in
        #: (`ViewFeature.undeclared_view_roots`): never overwritten in place
        self.hidden = frozenset()

    # ------------------------------------------------------------ layouts of values

    def lay(self, k):
        """The layout of value `k` (None for non-tensors)."""
        t = self.values.types[k]
        if not hasattr(t, "ndim") or is_absent(t):
            return None
        if k in self.values.consts:
            return Layout.of_const(self.values.consts[k])
        return self.layouts.get(k)

    def binding(self, node, vids, in_layouts) -> list:
        """The layouts of the values `vids`, the outputs of an item running `node` with
        inputs in `in_layouts`: those its outputs must have."""
        if any(k not in self.layouts for k in vids):
            for k, o in zip(vids, node_outputs(node, in_layouts), strict=True):
                if o.layout is not None:
                    self.layouts.setdefault(k, o.layout)
        return [self.lay(k) for k in vids]

    # ------------------------------------------------------------ items

    def resolve(self, items) -> list[Resolved]:
        resolved = []
        for it in items:
            try:
                resolved.append(self.resolve_item(it))
            except NotImplementedError:
                if not isinstance(it, UnitItem):
                    raise
                # a compiled OpFromGraph we can't generate: its items on their own,
                # and a copy of each result into the value the OpFromGraph's output is
                resolved += self.resolve(it.items)
                for inner, outer in it.outputs:
                    resolved.append(self.copy_item(inner, outer))
        return resolved

    def roots_of(self, var) -> frozenset:
        """The variables of the graph whose memory `var` may lie in."""
        return self.views.view_roots(self.fgraph, var)

    def may_destroy(self, it, i) -> bool:
        """May the item, a node of the graph, overwrite its input `i`? Its op destroys
        it, so the DestroyHandler says no one else needs it; unless the memory is a
        constant's, or views it doesn't see may lie in it."""
        return (
            self.allow_inplace
            and it.of_the_graph
            and it.inputs[i] not in self.values.consts
            and not self.roots_of(it.node.inputs[i]) & self.hidden
        )

    def grant_inplace(self, it, folded) -> dict:
        """{output: input} for the outputs of a node that may overwrite an input: its
        op destroys it, its emitter can (`llvm_inplace`), and nothing else can see it."""
        node = it.node
        granted = {}
        for k, i in inplace_outputs(node, folded).items():
            if not self.may_destroy(it, i):
                continue
            # nor may the memory be another input's
            mine = self.roots_of(node.inputs[i])
            if any(
                mine & self.roots_of(var)
                for pos, var in enumerate(node.inputs)
                if pos != i and hasattr(var.type, "ndim")
            ):
                continue
            granted[k] = i
        return granted

    def resolve_item(self, it) -> Resolved:
        """How the program's function runs an item."""
        if isinstance(it, UnitItem):
            return self.unit_item(it)
        in_layouts = [self.lay(k) for k in it.inputs]
        if isinstance(it, SubprogramItem):
            return self.kernel_item(it, self.resolver.subprogram(it.node, in_layouts))
        node = it.node
        binding = self.binding(node, it.outputs, in_layouts)
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
        # A Rust kernel first: there is nothing to compile. Not for a Scan or Blockwise
        # (its body is inlined into the generated loop), and not with folded constants
        # (a kernel would need them as arguments).
        nested = self.resolver.has_body(node)
        inplace = self.grant_inplace(it, folded)
        if not folded and not nested and not inplace:
            kernel = self.resolver.rust(node, in_layouts, binding)
            if kernel is not None:
                return self.kernel_item(it, kernel)
        try:
            return self.generated_item(it, folded, in_layouts, inplace)
        except NotImplementedError:
            # (a kernel doesn't overwrite its inputs: it can stand in for an in-place
            # node too)
            if nested or inplace:
                kernel = self.resolver.rust(node, in_layouts, binding)
                if kernel is not None:
                    return self.kernel_item(it, kernel)
            if not self.opts.numba_fallback:
                raise
        kernel = self.resolver.numba(node, binding)
        # it overwrites the inputs its op destroys: that is what the graph asks for;
        # where nothing else may see the memory (as in `grant_inplace`). Else it gets
        # copies.
        keep = frozenset(i for i in kernel.destroys if not self.may_destroy(it, i))
        return self.kernel_item(it, replace(kernel, destroys=keep))

    def kernel_item(self, it, kernel: CalledKernel) -> Resolved:
        """A node run by a kernel (Rust, numba, a subprogram)."""
        in_layouts = [self.lay(k) for k in it.inputs]
        sig = node_signature(it.node, in_layouts=in_layouts, declared=kernel.outputs)
        check_conforms(
            it.node, sig.outputs, self.binding(it.node, it.outputs, in_layouts)
        )
        return Resolved(it, in_layouts, kernel, sig)

    def compiled(self, module, nf, description) -> CalledKernel:
        """The kernel of the module `module` (its function `kernel`, around `nf`), to
        compile with the program (its address is set then)."""
        build_kernel_wrapper(module, nf, "kernel", self.opts.check_layouts)
        links = nf.links
        kernel = CalledKernel(
            0,
            0 if links is None else links.address,
            list(nf.signature.outputs),
            description,
            None if links is None else links.keepalive,
        )
        self.modules.append((module, kernel))
        return kernel

    def generated_item(self, it, folded, in_layouts, inplace=None) -> Resolved:
        """A node whose code is generated, compiled on its own. Raises
        NotImplementedError if no emitter handles it. `inplace`: {output: input} for
        the outputs that overwrite an input (`grant_inplace`)."""
        node = it.node
        body = (
            self.resolver.body(node, in_layouts)
            if self.resolver.has_body(node)
            else None
        )
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
            body=body,
            inplace=inplace,
        )
        if inplace:
            self.stats.inplace.append(str(node.op))
        skipped = skipped_inputs(node, folded)
        args = [k for pos, k in enumerate(it.inputs) if pos not in skipped]
        desc = str(node.op) + _layouts_note(
            [r for r in nf.signature.inputs if hasattr(r, "layout") and r.ndim > 0]
        )
        kernel = self.compiled(module, nf, desc)
        check_conforms(
            node, nf.signature.outputs, self.binding(node, it.outputs, in_layouts)
        )
        return Resolved(
            Call(args, list(it.outputs), desc),
            [self.lay(k) for k in args],
            kernel,
            nf.signature,
        )

    def unit_item(self, it: UnitItem) -> Resolved:
        """The items of a unit, generated as one function, compiled on its own."""
        types = self.values.types
        out_types = [types[o] for _, o in it.outputs]
        outs = [o for _, o in it.outputs]
        binding = self.binding(it.node, outs, [self.lay(k) for k in it.node_inputs])
        in_layouts = unit_input_layouts(
            [types[k] for k in it.inputs], [self.lay(k) for k in it.inputs]
        )
        inner_layouts = {
            k: lay
            for k, lay in zip(it.inputs, in_layouts, strict=True)
            if lay is not None
        }
        resolved = self.resolver.resolve_items(it.items, inner_layouts)
        # outputs that may be views of the unit's inputs are returned as such, where
        # their layout implies the binding one; the others are provided memory when
        # static, else allocated, in the binding layout
        views = _unit_views(it, resolved)
        modes, aliases = [], []
        for (inner, _), t, lay in zip(it.outputs, out_types, binding, strict=True):
            if views.get(inner) and inner_layouts[inner].implies(
                lay, list(t.shape), np.dtype(t.dtype).itemsize
            ):
                modes.append(OutputMode.RETURNED)
                aliases.append(tuple(sorted(views[inner])))
            else:
                modes.append(
                    OutputMode.PROVIDED if repr_of(t).static else OutputMode.RETURNED
                )
                aliases.append(())
        module = ir.Module(
            f"unit_{len(it.items)}",
            fastmath=self.opts.fastmath,
            debug_info=self.opts.debug_info,
        )
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
            output_layouts=binding,
            output_aliases=aliases,
            parallel=self.opts.parallel,
        )
        self.stats.n_units += 1
        names = ", ".join(str(s.node.op) for s in it.items if isinstance(s, NodeItem))
        desc = f"unit[{len(it.items)}]: {names[:120]}"
        kernel = self.compiled(module, nf, desc)
        return Resolved(
            Call(list(it.inputs), outs, desc), in_layouts, kernel, nf.signature
        )

    def copy_item(self, src, dst) -> Resolved:
        """A copy of value `src` into value `dst`."""
        node = _copy_node(self.values.types[dst])
        in_layouts = [self.lay(src)]
        kernel = self.resolver.fallback(
            node, in_layouts, self.binding(node, [dst], in_layouts)
        )
        kernel.description = "copy"
        return self.kernel_item(NodeItem(node, [src], [dst]), kernel)

    # ------------------------------------------------------------ the program

    def plan(self, fgraph, output_modes=None, input_layouts=None) -> Program:
        """`input_layouts`: what the caller guarantees about each input's layout (default:
        from `Options.input_layout`). `output_modes`: how each output is produced
        (PROVIDED or RETURNED; default: what avoids copies)."""
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
        if input_layouts is None:
            feature = layout_feature(fgraph)
            feature.set_input_layout(self.opts.input_layout)
            layout_of = fgraph.llvm_layout
        else:
            # (a subprogram: its graph is an OpFromGraph's, shared by all its nodes)
            layout_of = infer_layouts(fgraph, input_layouts).get
        scope = {}
        in_ids = []
        boxed = {}  # scalar input -> the 0-d array it arrives in
        for v in fgraph.inputs:
            if isinstance(v.type, ScalarType):
                # scalars don't cross the kernel ABI: the input is a 0-d array, which is
                # what `TensorFromScalar` of it is (the units rewrite reads scalar inputs
                # through it)
                from pytensor.tensor.type import TensorType

                t = values.new(TensorType(v.type.dtype, shape=()))
                in_ids.append(t)
                self.layouts[t] = Layout.c(0)
                boxed[v] = t
                scope[v] = values.new(v.type)
                continue
            k = values.new(v.type)
            scope[v] = k
            in_ids.append(k)
        contiguous = [
            input_layouts is None
            and self.opts.input_layout == "c"
            and hasattr(v.type, "ndim")
            for v in fgraph.inputs
        ]

        # 1. items
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
        items = flatten(values, nodes, scope, of_the_graph=True)
        out_ids = [values.of(o, scope) for o in fgraph.outputs]
        for v, k in scope.items():
            lay = layout_of(v)
            if lay is not None and k not in values.consts:
                self.layouts.setdefault(k, lay)

        # 2. how each item runs
        resolved = self.resolve(items)
        self.output_layouts = [self.lay(k) for k in out_ids]

        # 3. the program's function
        # values without data (random generators) aren't inputs or outputs of it
        self.program_inputs = [
            pos for pos, v in enumerate(fgraph.inputs) if not is_absent(v.type)
        ]
        self.program_outputs = [
            pos for pos, v in enumerate(fgraph.outputs) if not is_absent(v.type)
        ]
        in_ids = [in_ids[pos] for pos in self.program_inputs]
        contiguous = [contiguous[pos] for pos in self.program_inputs]
        out_ids = [out_ids[pos] for pos in self.program_outputs]
        if output_modes is None:
            modes = _output_modes(resolved, out_ids, values.types)
        else:
            modes = [output_modes[pos] for pos in self.program_outputs]
        self.stats.plan_s = time.perf_counter() - t0
        return self.program(resolved, in_ids, contiguous, out_ids, modes)

    def program(self, resolved, in_ids, contiguous, out_ids, modes) -> Program:
        types = self.values.types
        t0 = time.perf_counter()
        module = ir.Module(
            "program", fastmath=self.opts.fastmath, debug_info=self.opts.debug_info
        )
        profile = np.zeros(PROFILE_ITEMS + len(resolved), dtype=np.uint64)
        in_layouts = unit_input_layouts(
            [types[k] for k in in_ids], [self.lay(k) for k in in_ids]
        )
        nf = build_unit_function(
            module,
            "program",
            [(k, types[k]) for k in in_ids],
            [(k, types[k], m) for k, m in zip(out_ids, modes, strict=True)],
            resolved,
            self.values.constants_of([r.item for r in resolved], out_ids),
            input_layouts=in_layouts,
            # (returned outputs are handed over in any layout)
            output_layouts=[
                Layout.c(types[k].ndim)
                if m is OutputMode.PROVIDED
                else Layout.dynamic(types[k].ndim)
                for k, m in zip(out_ids, modes, strict=True)
            ],
            parallel=self.opts.parallel,
            links=LinkTable(),
            profile=profile,
            inline=False,
        )
        build_kernel_wrapper(module, nf, "kernel", self.opts.check_layouts)
        t1 = time.perf_counter()

        # 4. compile everything at once
        for m, kernel in self.modules:
            m.label = "ptl:" + " ".join(kernel.description.split())[:160]  # perf maps
        module.label = f"ptl:program[{len(resolved)}]"
        compiled = ir.compile_many(
            [*(m for m, _ in self.modules), module],
            opt_level=self.opts.opt_level,
        )
        for (_, kernel), c in zip(self.modules, compiled[:-1], strict=True):
            kernel.address = c.address("kernel")
            kernel.keepalive = (kernel.keepalive, c)
            self.stats.modules.append(GeneratedModule(kernel.description, c))
        self.stats.n_modules = len(self.modules)
        self.stats.modules.append(GeneratedModule("program", compiled[-1]))
        nf.links.relink()
        self.stats.codegen_s = t1 - t0
        self.stats.compile_s = time.perf_counter() - t1
        return Program(
            self,
            nf,
            compiled[-1],
            profile,
            resolved,
            in_ids,
            contiguous,
            out_ids,
            modes,
        )


def _output_modes(resolved, out_ids, types) -> list:
    """How the program produces each output: in memory the caller provides if its item
    writes provided memory (or it has a static shape and no item makes it: an input, a
    constant), else handed over in a buffer, so that nothing is copied that needn't
    be."""
    declared = {}
    for r in resolved:
        for k, o in zip(r.item.outputs, r.signature.outputs, strict=True):
            declared[k] = o.mode
    modes = []
    for k in out_ids:
        static = repr_of(types[k]).static
        mode = declared.get(k, OutputMode.PROVIDED if static else OutputMode.RETURNED)
        modes.append(
            OutputMode.PROVIDED
            if static and mode is OutputMode.PROVIDED
            else OutputMode.RETURNED
        )
    return modes


class Program:
    """A planned and compiled program: its function, callable from Python
    (`function`, or the program itself), from C (`c_function`), and as a kernel of
    other generated code (`address`, `config`: a subprogram)."""

    def __init__(
        self,
        planner,
        nf,
        compiled,
        profile,
        resolved,
        in_ids,
        contiguous,
        out_ids,
        modes,
    ):
        types = planner.values.types
        self.descriptions = [r.description for r in resolved]
        self.arena_bytes = nf.arena[1]
        self.profile_cells = profile
        self.address = compiled.address("kernel")
        self.config = nf.links.address
        #: per item, the programs it runs (subprograms), for profiling
        self.children = [
            [r.kernel.keepalive]
            if r.kernel is not None and isinstance(r.kernel.keepalive, Program)
            else []
            for r in resolved
        ]

        def shape(k):
            return [None if n is None else int(n) for n in types[k].shape]

        self.function = _interp().Function(
            self.address,
            self.config,
            [
                (np.dtype(types[k].dtype), shape(k), c)
                for k, c in zip(in_ids, contiguous, strict=True)
            ],
            [
                (np.dtype(types[k].dtype), shape(k), m is OutputMode.PROVIDED)
                for k, m in zip(out_ids, modes, strict=True)
            ],
            [compiled, nf.links.keepalive],
        )
        self._dump = _dump(types, nf, resolved, in_ids, contiguous, out_ids, modes)

    def __call__(self, *args):
        return self.function(*args)

    @property
    def tasks(self) -> list[str]:
        """What each item of the program's function runs."""
        return self.descriptions

    @property
    def n_tasks(self) -> int:
        return len(self.descriptions)

    def dump(self) -> str:
        """The program as text: its values with their types and memory, its items,
        its outputs."""
        return self._dump

    __str__ = dump

    def seed(self, seed: int):
        """Restart the random state of Python calls from `seed` (draws are then
        reproducible). The C API seeds each of its sessions."""
        self.function.seed(seed)

    def set_parallel(self, on: bool):
        """Allow or forbid parallel loops for Python calls (results are the same either
        way)."""
        self.function.set_parallel(on)

    def c_function(self) -> int:
        """A new `PtFunction` reference for the C API (`pytensor_llvm.c_header()`)."""
        return self.function.c_function()

    @staticmethod
    def c_api_table() -> int:
        return _interp().c_api_table()

    # ------------------------------------------------------------ profiling

    def _subprograms(self):
        for children in self.children:
            yield from children

    def set_profiling(self, on: bool):
        """Turn per-item timing on or off, here and in subprograms (two clock reads
        per item)."""
        self.profile_cells[PROFILE_ON] = int(on)
        for c in self._subprograms():
            c.set_profiling(on)

    def reset_profile(self):
        self.profile_cells[PROFILE_CALLS:] = 0
        for c in self._subprograms():
            c.reset_profile()

    def _calls(self) -> int:
        return int(self.profile_cells[PROFILE_CALLS])

    def _secs(self) -> list[float]:
        return [int(n) * 1e-9 for n in self.profile_cells[PROFILE_ITEMS:]]

    def profile(self) -> list[tuple[int, str, int, float]]:
        """Rows `(depth, description, calls, seconds)`: each item with its total time,
        followed by the items of the programs it runs at depth + 1; `calls` is the
        number of runs of the program the item belongs to."""
        rows = []
        self._profile_rows(0, rows)
        return rows

    def _profile_rows(self, depth, rows):
        calls = self._calls()
        for desc, secs, children in zip(
            self.descriptions, self._secs(), self.children, strict=True
        ):
            rows.append((depth, desc, calls, secs))
            for c in children:
                c._profile_rows(depth + 1, rows)

    def profile_report(self) -> str:
        """The profile as a table, slowest first within each program."""
        calls, secs = self._calls(), self._secs()
        total = sum(secs)
        per_call = total / calls * 1e6 if calls else 0.0
        lines = [
            f"{calls} calls, {total * 1e3:.3f} ms in items ({per_call:.2f} us per call)",
            f"{'%':>7} {'total ms':>10} {'us/call':>10}  item (subprogram items indented)",
        ]
        self._report_into(0, total, calls, lines)
        return "\n".join(lines) + "\n"

    def _report_into(self, depth, total, calls, lines):
        secs = self._secs()
        for k in sorted(range(len(secs)), key=lambda k: -secs[k]):
            pct = 100 * secs[k] / total if total > 0 else 0.0
            per = secs[k] / calls * 1e6 if calls else 0.0
            desc = self.descriptions[k][:100]
            lines.append(
                f"{pct:>6.1f}% {secs[k] * 1e3:>10.3f} {per:>10.2f}  {'  ' * depth}{desc}"
            )
            for c in self.children[k]:
                c._report_into(depth + 1, total, calls, lines)


def _dump(types, nf, resolved, in_ids, contiguous, out_ids, modes) -> str:
    """`Program.dump`'s text."""
    ids = {}  # vid -> its number in the dump
    for k in [*in_ids, *nf.constants]:
        ids.setdefault(k, len(ids))
    for r in resolved:
        for k in [*r.arguments, *r.item.outputs]:
            ids.setdefault(k, len(ids))
    for k in out_ids:
        ids.setdefault(k, len(ids))
    offsets = nf.arena[0]
    produced = {}  # vid -> (item, its Output)
    for r in resolved:
        for k, o in zip(r.item.outputs, r.signature.outputs, strict=True):
            produced[k] = (r, o)

    def ty(k):
        shape = ", ".join("?" if n is None else str(n) for n in types[k].shape)
        return f"{np.dtype(types[k].dtype).name}[{shape}]"

    lines = [
        f"program: {len(in_ids)} inputs, {len(out_ids)} outputs, {len(resolved)} "
        f"items, arena {nf.arena[1]} bytes",
        "values:",
    ]
    for k in sorted(ids, key=ids.get):
        if is_absent(types[k]):
            continue
        if k in in_ids:
            pos = in_ids.index(k)
            kind = f"input {pos}" + (" (C-contiguous)" if contiguous[pos] else "")
        elif k in nf.constants:
            kind = f"constant at 0x{int(nf.links.entries[nf.constants[k]]):x}"
        elif k not in produced:
            kind = "constant (folded)"
        else:
            r, o = produced[k]
            if o.mode is OutputMode.INPLACE:
                kind = f"overwrites %{ids[r.arguments[o.aliases[0]]]}"
            elif k in offsets:
                kind = f"arena +{offsets[k]}"
                order = list(o.layout.order) if o.layout is not None else []
                if o.layout.fully_packed(o.repr.shape) and order != sorted(order):
                    kind += f", order {order}"
            elif o.mode is OutputMode.PROVIDED and k in out_ids:
                kind = f"output {out_ids.index(k)}"
            elif o.mode is OutputMode.PROVIDED:
                kind = "temporary"
            elif o.aliases:
                kind = "returned by its item, may be a view of " + ", ".join(
                    f"%{ids[r.arguments[a]]}" for a in o.aliases
                )
            else:
                kind = "allocated by its item"
        lines.append(f"  %{ids[k]:<4} {ty(k):<24} {kind}")
    lines.append("items:")
    for i, r in enumerate(resolved):
        ins = ", ".join(f"%{ids[k]}" for k in r.arguments)
        outs = ", ".join(
            f"%{ids[k]}"
            for k, o in zip(r.item.outputs, r.signature.outputs, strict=True)
            if o.mode is not OutputMode.ABSENT
        )
        lines.append(f"  {i:<4} ({ins}) -> ({outs})  {r.description}")
    outs = ", ".join(
        f"%{ids[k]} ({'provided' if m is OutputMode.PROVIDED else 'returned'})"
        for k, m in zip(out_ids, modes, strict=True)
    )
    lines.append(f"outputs: {outs}")
    return "\n".join(lines) + "\n"


def _unit_views(it: UnitItem, resolved) -> dict:
    """vid -> positions of the unit's inputs the value may be a view of, following the
    views the unit's items may return."""
    views = {k: {pos} for pos, k in enumerate(it.inputs)}
    for r in resolved:
        args = r.arguments
        for vid, o in zip(r.item.outputs, r.signature.outputs, strict=True):
            if o.mode in (OutputMode.RETURNED, OutputMode.INPLACE) and o.aliases:
                views[vid] = set().union(*(views.get(args[a], ()) for a in o.aliases))
    for k in it.inputs:
        del views[k]
    return views


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
    """Plan and compile `fgraph`; returns (callable, stats) (see `compile_program`)."""
    _, fn, stats = compile_program(fgraph, options)
    return fn, stats


def compile_program(fgraph, options: Options | None = None):
    """Plan and compile `fgraph`; returns (program, callable, stats). The callable is
    the program's function, or, when the graph has random generator inputs or outputs
    (values without data, see `is_absent`), a wrapper that leaves them out of the call
    and returns each generator output as the generator it came from (draws come from
    the program's own random state)."""
    planner = Planner(options or Options())
    prog = planner.plan(fgraph)
    fn = prog.function
    n_in, n_out = len(fgraph.inputs), len(fgraph.outputs)
    if planner.program_inputs == list(range(n_in)) and planner.program_outputs == list(
        range(n_out)
    ):
        return prog, fn, planner.stats
    ins, outs = planner.program_inputs, planner.program_outputs
    sources = {
        pos: _generator_source(fgraph, o)
        for pos, o in enumerate(fgraph.outputs)
        if pos not in outs
    }

    def call(*args):
        res = fn(*[args[k] for k in ins])
        full = [None] * n_out
        for k, r in zip(outs, res, strict=True):
            full[k] = r
        for pos, src in sources.items():
            full[pos] = None if src is None else args[src]
        return full

    return prog, call, planner.stats


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
