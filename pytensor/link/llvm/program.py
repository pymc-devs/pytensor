"""Programs: a FunctionGraph becomes one generated function, and that function compiled.

`compile_program` does this:

1. The planner decides how every node of the graph runs (`Planner.program`): a Rust
   kernel, a numba kernel, or code generated for it, compiled in a module of its own.
   The layouts of the values come from the graph's `LayoutFeature`
   (`analysis/layout.py`).
2. The program's function runs the planned graph (`build_graph_function`,
   codegen/graph.py): it calls the steps in order, its temporaries on the stack or in
   an arena laid out by liveness (`analysis/liveness.py`). What is decided here is its
   boundary with the caller: values without data (random generators) aren't passed,
   and each output is written into memory the caller provides or handed over in a
   buffer (`_output_modes`).
3. All generated modules are compiled in one parallel batch, and the function becomes
   a `Program`.
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from pytensor.graph.basic import Constant
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.graph import (
    PROFILE_CALLS,
    PROFILE_ITEMS,
    PROFILE_ON,
    build_graph_function,
)
from pytensor.link.llvm.codegen.kernel import build_kernel_wrapper
from pytensor.link.llvm.codegen.kernel_module import (
    CodegenOptions,
    KernelModule,
    link,
)
from pytensor.link.llvm.codegen.plan import PlannedGraph, ValueId
from pytensor.link.llvm.codegen.signature import Output, Passing
from pytensor.link.llvm.codegen.types import TensorRepr, is_absent, repr_of
from pytensor.link.llvm.planner import Planner
from pytensor.scalar.basic import ScalarType


if TYPE_CHECKING:
    from pytensor_llvm.ir import CompiledModule
    from pytensor_llvm.runtime import Function

    from pytensor.tensor.type import TensorType


def _runtime():
    from pytensor_llvm import runtime

    return runtime


@dataclass
class Options:
    fastmath: bool = True
    opt_level: int = 3
    numba_fallback: bool = True
    #: what generated code may assume about the graph inputs: "any" strides, or "c"
    #: (C-contiguous: Python calls copy other arrays, the C API rejects them)
    input_layout: str = "any"
    #: may generated loops run in parallel (chunks on the runtime's thread pool, when
    #: there is enough work; see analysis/work.py)
    parallel: bool = True
    #: may Dot and the linear algebra kernels use faer's parallelism, on the runtime's
    #: thread pool (for large enough inputs, when the session allows parallelism; see
    #: `faer_kernels!` in kernels/linalg.rs). Results can differ from the sequential
    #: ones in the last bits.
    linalg_parallel: bool = False
    #: generated code verifies the layouts its signatures claim (for testing; also
    #: turned on by PYTENSOR_LLVM_CHECK_LAYOUTS=1)
    check_layouts: bool = False
    #: give the generated code debug information: every instruction is located at the
    #: Python code that emitted it, in a function named after its node (for remarks,
    #: perf, debuggers; also turned on by PYTENSOR_LLVM_DEBUG_INFO=1)
    debug_info: bool = False

    @property
    def codegen(self) -> CodegenOptions:
        """The options of code generation."""
        return CodegenOptions(
            parallel=self.parallel,
            check_layouts=self.check_layouts,
            fastmath=self.fastmath,
            debug_info=self.debug_info,
        )

    def __post_init__(self):
        import os

        if os.environ.get("PYTENSOR_LLVM_CHECK_LAYOUTS", "0") not in ("", "0"):
            self.check_layouts = True
        if os.environ.get("PYTENSOR_LLVM_DEBUG_INFO", "0") not in ("", "0"):
            self.debug_info = True


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

    cache: dict[str, str] = {}

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
    """A module the planner generated (for a step, or the program's function),
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
    #: planning, with the code generated for the steps
    plan_s: float = 0.0
    #: generating the program's function
    codegen_s: float = 0.0
    compile_s: float = 0.0
    n_modules: int = 0
    n_rust_kernels: int = 0
    n_numba: int = 0
    n_units: int = 0
    numba: list = field(default_factory=list)
    #: the ops of the generated steps that may overwrite an input
    inplace: list = field(default_factory=list)
    #: the generated modules (GeneratedModule)
    modules: list = field(default_factory=list)

    def ir(
        self,
        optimized: bool = False,
        opt_level: int | None = None,
        remarks=False,
        locations=False,
    ) -> str:
        """The LLVM IR of all generated modules, each under a comment naming its step.
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


def _output_modes(steps, out_ids, types) -> list:
    """How the program produces each output: in memory the caller provides if its step
    writes provided memory (or it has a static shape and no step makes it: an input, a
    constant), else handed over in a buffer, so that nothing is copied that needn't
    be."""
    declared = {}
    for step in steps:
        for k, output in zip(step.outputs, step.signature.outputs, strict=True):
            declared[k] = output.passing
    modes = []
    for k in out_ids:
        representation = repr_of(types[k])
        static = isinstance(representation, TensorRepr) and representation.static
        default = Passing.CALLER_BUFFER if static else Passing.CALLEE_SLOT
        passing = declared.get(k, default)
        modes.append(
            Passing.CALLER_BUFFER
            if static and passing is Passing.CALLER_BUFFER
            else Passing.CALLEE_SLOT
        )
    return modes


class Program:
    """A planned and compiled program: its function, callable from Python
    (`function`, or the program itself) and from C (`c_function`)."""

    #: what each step runs
    descriptions: list[str]
    #: the size in bytes of the arena, the scratch memory of the temporary values
    arena_bytes: int
    #: the profiling cells: on/off, calls, then nanoseconds per step (`PROFILE_*`)
    profile_cells: np.ndarray
    #: the address and config of the program's kernel
    address: int
    config: int
    #: may it run work on the runtime's thread pool? (Calls then run on the pool.)
    uses_pool: bool
    #: the compiled function, callable from Python
    function: Function
    _dump: str

    def __init__(
        self,
        types,
        unit,
        node_function,
        profile,
        steps,
        in_ids,
        contiguous,
        out_ids,
        modes,
        uses_pool,
    ):
        self.descriptions = [step.description for step in steps]
        self.arena_bytes = node_function.arena[1]
        self.profile_cells = profile
        self.address = unit.address
        self.uses_pool = uses_pool
        self.config = unit.config

        def shape(k):
            return [None if n is None else int(n) for n in types[k].shape]

        self.function = _runtime().Function(
            self.address,
            self.config,
            [
                (np.dtype(types[k].dtype), shape(k), c)
                for k, c in zip(in_ids, contiguous, strict=True)
            ],
            [
                (np.dtype(types[k].dtype), shape(k), m is Passing.CALLER_BUFFER)
                for k, m in zip(out_ids, modes, strict=True)
            ],
            [unit.compiled, unit.links.keepalive],
            uses_pool,
            unit.exceptions.classes,  # (complete: everything is generated)
        )
        self._dump = _dump(
            types, node_function, steps, in_ids, contiguous, out_ids, modes
        )

    def __call__(self, *args):
        return self.function(*args)

    @property
    def steps(self) -> list[str]:
        """What each step of the program's function runs."""
        return self.descriptions

    @property
    def n_steps(self) -> int:
        return len(self.descriptions)

    def dump(self) -> str:
        """The program as text: its values with their types and memory, its steps,
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
        from pytensor_llvm import runtime

        return runtime.c_api_table()

    # ------------------------------------------------------------ profiling

    def set_profiling(self, on: bool):
        """Turn per-step timing on or off (two clock reads per step)."""
        self.profile_cells[PROFILE_ON] = int(on)

    def reset_profile(self):
        self.profile_cells[PROFILE_CALLS:] = 0

    def profile(self) -> list[tuple[str, int, float]]:
        """Rows `(description, calls, seconds)`: each step with its total time over
        `calls` runs of the program."""
        calls = int(self.profile_cells[PROFILE_CALLS])
        secs = [int(n) * 1e-9 for n in self.profile_cells[PROFILE_ITEMS:]]
        return [(d, calls, t) for d, t in zip(self.descriptions, secs, strict=True)]

    def profile_report(self) -> str:
        """The profile as a table, slowest first."""
        rows = self.profile()
        calls = int(self.profile_cells[PROFILE_CALLS])
        total = sum(t for *_, t in rows)
        per_call = total / calls * 1e6 if calls else 0.0
        lines = [
            f"{calls} calls, {total * 1e3:.3f} ms in steps ({per_call:.2f} us per call)",
            f"{'%':>7} {'total ms':>10} {'us/call':>10}  step",
        ]
        for desc, _, t in sorted(rows, key=lambda row: -row[2]):
            pct = 100 * t / total if total > 0 else 0.0
            per = t / calls * 1e6 if calls else 0.0
            lines.append(f"{pct:>6.1f}% {t * 1e3:>10.3f} {per:>10.2f}  {desc[:100]}")
        return "\n".join(lines) + "\n"


def _dump(types, node_function, steps, in_ids, contiguous, out_ids, modes) -> str:
    """`Program.dump`'s text."""
    ids: dict[ValueId, int] = {}  # value id -> its number in the dump
    for k in [*in_ids, *node_function.constants]:
        ids.setdefault(k, len(ids))
    for step in steps:
        for k in [*step.arguments, *step.outputs]:
            ids.setdefault(k, len(ids))
    for k in out_ids:
        ids.setdefault(k, len(ids))
    offsets = node_function.arena[0]
    produced = {}  # value id -> (step, its Output)
    for step in steps:
        for k, output in zip(step.outputs, step.signature.outputs, strict=True):
            produced[k] = (step, output)

    def ty(k):
        shape = ", ".join("?" if n is None else str(n) for n in types[k].shape)
        return f"{np.dtype(types[k].dtype).name}[{shape}]"

    lines = [
        f"program: {len(in_ids)} inputs, {len(out_ids)} outputs, {len(steps)} "
        f"steps, arena {node_function.arena[1]} bytes",
        "values:",
    ]
    for k in ids:  # (in the order of their numbers)
        if is_absent(types[k]):
            continue
        if k in in_ids:
            pos = in_ids.index(k)
            kind = f"input {pos}" + (" (C-contiguous)" if contiguous[pos] else "")
        elif k in node_function.constants:
            kind = f"constant at 0x{int(node_function.links.entries[node_function.constants[k]]):x}"
        elif k not in produced:
            kind = "constant (folded)"
        else:
            step, output = produced[k]
            if k in node_function.temporaries:
                kind = f"arena +{offsets[k]}" if k in offsets else "on the stack"
                order = list(output.layout.order) if output.layout is not None else []
                packed = output.layout.fully_packed(output.representation.shape)
                if packed and order != sorted(order):
                    kind += f", order {order}"
            elif output.passing is Passing.CALLER_BUFFER and k in out_ids:
                kind = f"output {out_ids.index(k)}"
            elif output.passing is Passing.CALLER_BUFFER:
                kind = "temporary"
            elif output.aliased_inputs:
                kind = "returned by its step, may be (a view of) " + ", ".join(
                    f"%{ids[step.arguments[a]]}"
                    for a in step.signature.alias_arguments(output)
                )
            else:
                kind = "allocated by its step"
        lines.append(f"  %{ids[k]:<4} {ty(k):<24} {kind}")
    lines.append("steps:")
    for i, step in enumerate(steps):
        ins = ", ".join(f"%{ids[k]}" for k in step.arguments)
        outs = ", ".join(
            f"%{ids[k]}"
            for k, output in zip(step.outputs, step.signature.outputs, strict=True)
            if not output.is_absent
        )
        lines.append(f"  {i:<4} ({ins}) -> ({outs})  {step.description}")
    outs = ", ".join(
        f"%{ids[k]} ({'provided' if m is Passing.CALLER_BUFFER else 'returned'})"
        for k, m in zip(out_ids, modes, strict=True)
    )
    lines.append(f"outputs: {outs}")
    return "\n".join(lines) + "\n"


def build_program(fgraph, options: Options | None = None):
    """Plan and compile `fgraph`; returns (callable, stats) (see `compile_program`)."""
    _, fn, stats = compile_program(fgraph, options)
    return fn, stats


def compile_program(fgraph, options: Options | None = None):
    """Plan and compile `fgraph`; returns (program, callable, stats). The callable is
    the program's function, or, when the graph has random generator inputs or outputs
    (values without data, see `is_absent`) or scalar outputs, a wrapper that leaves the
    generators out of the call and returns each generator output as the generator it
    came from (a copy of a constant one; draws come from the program's own random
    state), and each scalar output
    as a numpy scalar; also, for a graph without outputs, a wrapper returning None."""
    opts = options or Options()
    stats = Stats()
    planner = Planner(opts, stats)
    t0 = time.perf_counter()
    planned = planner.program(fgraph)
    stats.plan_s = time.perf_counter() - t0
    # values without data (random generators) aren't inputs or outputs of the function
    ins = [pos for pos, v in enumerate(fgraph.inputs) if not is_absent(v.type)]
    outs = [pos for pos, v in enumerate(fgraph.outputs) if not is_absent(v.type)]
    contiguous = [
        opts.input_layout == "c" and hasattr(fgraph.inputs[pos].type, "ndim")
        for pos in ins
    ]
    prog = _build(planner, planned, ins, contiguous, outs)
    fn = prog.function
    # a scalar output comes back as the 0-d array it leaves in
    scalars = {pos for pos in outs if isinstance(fgraph.outputs[pos].type, ScalarType)}
    if (
        not scalars
        and fgraph.outputs
        and ins == list(range(len(fgraph.inputs)))
        and outs == list(range(len(fgraph.outputs)))
    ):
        return prog, fn, stats
    sources = {
        pos: _generator_source(fgraph, var)
        for pos, var in enumerate(fgraph.outputs)
        if pos not in outs
    }

    def call(*args):
        res = fn(*[args[k] for k in ins])
        full = [None] * len(fgraph.outputs)
        for k, result in zip(outs, res, strict=True):
            full[k] = result[()] if k in scalars else result
        for pos, src in sources.items():
            # (a constant generator: a copy each call, as `DeepCopyOp` makes)
            full[pos] = args[src] if isinstance(src, int) else copy.deepcopy(src)
        # (a function without outputs returns None, as the JIT linker expects)
        return full or None

    return prog, call, stats


def _build(planner: Planner, planned: PlannedGraph, ins, contiguous, outs) -> Program:
    """Generate the program's function around the planned graph `planned`, compile it
    with the modules the planner generated for its steps, and wrap it in a `Program`.
    `ins`, `outs`: the positions of the graph's inputs and outputs the function has;
    `contiguous`: per input, is it guaranteed C-contiguous?"""
    opts, stats = planner.opts, planner.stats
    types = planner.values.types
    steps = planned.steps
    in_ids = [planned.in_ids[pos] for pos in ins]
    out_ids = [planned.out_ids[pos] for pos in outs]
    modes = _output_modes(steps, out_ids, types)
    t0 = time.perf_counter()
    unit = KernelModule.new("program", opts.codegen, planner.graph, planner.exceptions)
    profile = np.zeros(PROFILE_ITEMS + len(steps), dtype=np.uint64)
    # (returned outputs are handed over in any layout)
    outputs = []
    for k, passing in zip(out_ids, modes, strict=True):
        # Outputs with data are always tensors: scalar outputs are returned as
        # 0-d arrays.
        ndim = cast("TensorType", types[k]).ndim
        if passing is Passing.CALLER_BUFFER:
            layout = Layout.c(ndim)
        else:
            layout = Layout.dynamic(ndim)
        outputs.append(Output(repr_of(types[k], layout), passing))
    function_graph = PlannedGraph(
        in_ids, out_ids, steps, planned.constants, planned.layouts
    )
    node_function = build_graph_function(
        unit,
        "program",
        function_graph,
        [types[k] for k in in_ids],
        [planner.layout(k, planned.layouts) for k in in_ids],
        outputs,
        profile=profile,
    )
    build_kernel_wrapper(unit, node_function)
    t1 = time.perf_counter()

    # compile and link everything at once
    unit.module.label = f"ptl:program[{len(steps)}]"
    link([*planner.modules, unit], opts.opt_level)
    for m in planner.modules:
        assert m.compiled is not None
        stats.modules.append(GeneratedModule(m.description, m.compiled))
    stats.n_modules = len(planner.modules)
    assert unit.compiled is not None
    stats.modules.append(GeneratedModule("program", unit.compiled))
    stats.codegen_s = t1 - t0
    stats.compile_s = time.perf_counter() - t1
    return Program(
        types,
        unit,
        node_function,
        profile,
        steps,
        in_ids,
        contiguous,
        out_ids,
        modes,
        any(m.uses_pool for m in (*planner.modules, unit)),
    )


def _generator_source(fgraph, var):
    """What a generator output passes on (through random variables and copies): the
    position of the graph input it comes from (an int), else the generator of the
    constant it comes from, else None."""
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
    if var in fgraph.inputs:
        return fgraph.inputs.index(var)
    return var.data if isinstance(var, Constant) else None


def _is_none_type(t):
    from pytensor.tensor.type_other import NoneTypeT

    return isinstance(t, NoneTypeT)
