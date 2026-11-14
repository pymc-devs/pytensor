"""Generated functions: signatures, node functions and kernel ABI wrappers.

See DESIGN.md, sections 2, 3 and 6.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import singledispatch

from pytensor.link.llvm.codegen.emit import (
    Codegen,
    InplaceOutput,
    ProvidedOutput,
    ScalarOutput,
    SlotOutput,
    pack,
    unpack,
)
from pytensor.link.llvm.codegen.types import (
    PTARRAY_SIZE,
    NoneRepr,
    PtArray,
    ScalarRepr,
    Slot,
    TensorRepr,
    ir,
    ptarray_type,
    repr_of,
)
from pytensor.link.llvm.layout import Layout


class OutputMode(enum.Enum):
    #: the caller provides the memory (with its own strides)
    PROVIDED = "provided"
    #: the kernel returns the array through a slot: memory it allocated (it hands over
    #: one reference), or a view of an input the output may alias
    RETURNED = "returned"
    #: the output is an input's memory, overwritten: no argument (`aliases` has the input)
    INPLACE = "inplace"
    #: a ScalarType output, stored through a pointer
    SCALAR = "scalar"
    #: no value (a random generator token): no argument
    ABSENT = "absent"


@dataclass(frozen=True)
class Output:
    repr: TensorRepr | ScalarRepr
    mode: OutputMode
    #: the inputs the output may be a view of (RETURNED; none: it is new memory)
    aliases: tuple = ()

    @property
    def layout(self):
        """The output's layout (None for scalars). For PROVIDED it is the memory the
        op asks for (the provider may give it other memory, see `with_layout`); for
        RETURNED it is what the result has, a view or new memory."""
        return getattr(self.repr, "layout", None)

    def with_layout(self, layout):
        """The same output with another layout (e.g. the memory actually provided)."""
        from dataclasses import replace

        return replace(self, repr=TensorRepr(self.repr.dtype, self.repr.shape, layout))

    @property
    def ir_type(self):
        if self.mode is OutputMode.PROVIDED:
            return self.repr.ir_type
        return ir.ptr


@dataclass(frozen=True)
class Signature:
    inputs: tuple
    outputs: tuple[Output, ...]

    def param_types(self):
        outs = [
            o.ir_type
            for o in self.outputs
            if o.mode not in (OutputMode.ABSENT, OutputMode.INPLACE)
        ]
        return [ir.ptr, ir.ptr, *(r.ir_type for r in self.inputs), *outs]

    @property
    def has_scalars(self):
        return any(isinstance(r, ScalarRepr) for r in self.inputs) or any(
            isinstance(o.repr, ScalarRepr) for o in self.outputs
        )


@singledispatch
def llvm_layout(op, node, in_layouts) -> list:
    """The layout (layout.py) of each output of `node`, given the inputs' layouts
    (`known_layouts` fills in the unknown ones). None for an output: the default.

    The default is C-contiguous for an output in new memory, and nothing known
    (DYNAMIC) for one that may be a view. Ops register their own to say what a view's
    layout is (DimShuffle, Subtensor, ...) or in which memory order they want their
    output (Elemwise: its inputs').

    This says what the result looks like, not how it comes to be: whether an output may
    be a view of an input is the op's `view_map` (`output_aliases`), and whether its
    memory is provided or returned follows from that and from its type (`node_outputs`).
    A layout is binding. New memory is provided or allocated in it (it must then be
    packed; one that isn't means "any strides": C-contiguous memory is used). A view
    must have it (`SlotOutput.set_view` checks), and so must memory a may-be-view output
    allocates instead (a Reshape that copies declares C for both)."""
    return [None] * len(node.outputs)


def output_aliases(op, k: int) -> tuple:
    """The inputs output `k` of `op` may be a view of: what its `view_map` lists."""
    view_map = getattr(op, "view_map", None) or {}
    return tuple(view_map.get(k, ()))


@singledispatch
def llvm_inplace(op, node) -> dict:
    """{output: input} for the outputs the op's emitter can produce in the memory of an
    input the op destroys (`destroy_map`): the result then overwrites the input, element
    for element, and nothing is allocated.

    The default is none, which is always correct: an implementation may ignore that it
    is allowed to destroy an input. An op that registers this must write such an
    output with `out.overwrite(cg, x)` (or `out.allocate`, which is then the input's
    memory, still holding the input's values). The planner decides per task whether the
    output really is the input's memory; if not, both give new memory."""
    return {}


def inplace_outputs(node, folded=frozenset()) -> dict:
    """`llvm_inplace` of the node, for the pairs that can work: the op destroys that
    input, and input and output are tensors of one type."""
    op = node.op
    destroy_map = getattr(op, "destroy_map", None) or {}
    pairs = {}
    for k, i in llvm_inplace(op, node).items():
        out, inp = node.outputs[k], node.inputs[i]
        if i not in destroy_map.get(k, ()) or i in folded or i in pairs.values():
            continue
        if not isinstance(repr_of(out.type), TensorRepr) or not isinstance(
            repr_of(inp.type), TensorRepr
        ):
            continue
        if (
            out.type.dtype != inp.type.dtype
            or out.type.ndim != inp.type.ndim
            or out.type.broadcastable != inp.type.broadcastable
        ):
            continue
        pairs[k] = i
    return pairs


def new_output(var, layout=None) -> Output:
    """An output in new memory: SCALAR, PROVIDED for static shapes, else RETURNED; in
    `layout` (default C-contiguous). ABSENT for values without data."""
    r = repr_of(var.type)
    if isinstance(r, NoneRepr):
        return Output(r, OutputMode.ABSENT)
    if isinstance(r, ScalarRepr):
        return Output(r, OutputMode.SCALAR)
    lay = layout if layout is not None else Layout.c(var.type.ndim)
    return Output(
        repr_of(var.type, lay), OutputMode.PROVIDED if r.static else OutputMode.RETURNED
    )


def known_layouts(node, in_layouts) -> list:
    """`in_layouts` with DYNAMIC for tensors whose layout isn't known."""
    if in_layouts is None:
        in_layouts = [None] * len(node.inputs)
    return [
        lay
        if lay is not None or not hasattr(i.type, "ndim")
        else Layout.dynamic(i.type.ndim)
        for i, lay in zip(node.inputs, in_layouts, strict=True)
    ]


def node_outputs(node, in_layouts=None, inplace=None) -> list[Output]:
    """How each output of `node` comes to be when its code is generated: from its type,
    the op's `view_map` and `llvm_layout`.

    `inplace`: {output: input} for the outputs that are to overwrite an input
    (`inplace_outputs`, if the caller grants it). Without it, an op that may destroy
    an input writes new memory."""
    op = node.op
    in_layouts = known_layouts(node, in_layouts)
    layouts = llvm_layout(op, node, in_layouts)
    inplace = inplace or {}
    outs = []
    for k, (o, lay) in enumerate(zip(node.outputs, layouts, strict=True)):
        if k in inplace:
            # the input's memory, so the input's layout
            outs.append(
                Output(
                    repr_of(o.type, in_layouts[inplace[k]]),
                    OutputMode.INPLACE,
                    (inplace[k],),
                )
            )
            continue
        aliases = (
            output_aliases(op, k) if isinstance(repr_of(o.type), TensorRepr) else ()
        )
        if not aliases:
            outs.append(new_output(o, lay))
            continue
        if lay is None:
            lay = Layout.dynamic(o.type.ndim)
        outs.append(Output(repr_of(o.type, lay), OutputMode.RETURNED, aliases))
    return outs


def skipped_inputs(node, folded=frozenset()):
    """Input positions that aren't arguments: folded constants and None inputs."""
    from pytensor.link.llvm.codegen.types import is_absent

    return frozenset(folded) | {
        k for k, i in enumerate(node.inputs) if is_absent(i.type)
    }


def node_signature(
    node,
    folded=frozenset(),
    in_layouts=None,
    provided=None,
    declared=None,
    inplace=None,
) -> Signature:
    """`folded`: positions of inputs that are constants folded into the code (not
    arguments). None inputs aren't arguments either.

    `in_layouts`: the `Layout` of each input; outputs as `node_outputs` gives them, with
    `aliases` counting the function's arguments.
    The declared layout of a PROVIDED output is binding: whoever provides the memory
    provides it in that layout. `provided[k]` replaces the declaration, for a caller
    that generates a function for memory it fixes (tests writing into a slice).
    `declared`: the outputs instead (a kernel's, see call.py; `aliases` counting the
    node's inputs). `inplace`: see `node_outputs`."""
    folded = skipped_inputs(node, folded)
    if in_layouts is None:
        in_layouts = [None] * len(node.inputs)
    inputs = tuple(
        repr_of(i.type, lay)
        for k, (i, lay) in enumerate(zip(node.inputs, in_layouts, strict=True))
        if k not in folded
    )

    def arg_index(pos):
        # position among the function's (non-folded) inputs
        if pos is None:
            return None
        if pos in folded:
            raise NotImplementedError("a view of a folded constant")
        return pos - sum(1 for p in folded if p < pos)

    outputs = []
    for k, o in enumerate(
        declared if declared is not None else node_outputs(node, in_layouts, inplace)
    ):
        if (
            o.mode is OutputMode.PROVIDED
            and provided is not None
            and provided[k] is not None
        ):
            o = o.with_layout(provided[k])
        if o.aliases:
            o = Output(o.repr, o.mode, tuple(arg_index(pos) for pos in o.aliases))
        outputs.append(o)
    return Signature(inputs, tuple(outputs))


@singledispatch
def llvm_emit(op, node, cg: Codegen, inputs: list, outputs: list) -> None:
    """Emit the code of `node`.

    `inputs` are `Array`s (tensors) or values (scalars); `outputs` are `ProvidedOutput`,
    `SlotOutput` or `ScalarOutput` objects to fill.
    """
    raise NotImplementedError(f"No LLVM backend implementation for {op}")


@dataclass
class NodeFunction:
    function: object
    signature: Signature
    #: its link table (call.py): the config to call it with, or None
    links: object = None


def build_node_function(
    module,
    node,
    name: str,
    temporary_outputs=frozenset(),
    folded=None,
    in_layouts=None,
    provided=None,
    parallel=True,
    kernel=None,
    links=None,
    scan_body=None,
    inplace=None,
) -> NodeFunction:
    """The LLVM function of one node (DESIGN.md, section 2), always inlined.

    * `temporary_outputs`: allocated outputs that are temporaries of a compiled unit;
      they go into temporary memory instead of runtime buffers;
    * `folded`: {input position: value} for 0-d constant inputs compiled into the code;
    * `kernel`: a `CalledKernel` implementing the node (a Rust kernel, a numba kernel, a
      subprogram): the function calls it (call.py), through an entry of `links`, the
      link table that is the config of the kernel being generated. Code nested in the
      node (a Scan's loop body) links its kernels there too;
    * `scan_body`: for a Scan, its resolved loop body (`plan.resolve.Body`);
    * `inplace`: {output: input} for outputs that overwrite an input (`node_outputs`).
    """
    from pytensor.link.llvm.codegen.emit import c_array, constant, store
    from pytensor.link.llvm.codegen.types import memory_type

    folded = folded or {}
    sig = node_signature(
        node,
        frozenset(folded),
        in_layouts,
        provided,
        kernel.outputs if kernel is not None else None,
        inplace,
    )
    sig_inputs = iter(sig.inputs)
    fn = module.function(name, sig.param_types(), ir.i32, inline=ir.Inline.Always)
    b = fn.builder()
    params = fn.params
    ctx, config = params[:2]
    cg = Codegen(
        b,
        ctx,
        config,
        parallel=parallel,
        links=links,
        scan_body=scan_body,
        module=module,
        name=name,
    )
    # Aliasing (DESIGN.md): what LLVM is told. Every array the function writes (a
    # provided or returned output, an input overwritten in place) overlaps no other
    # array of the function; inputs that are only read may overlap each other.
    b.alias_domain(name)
    written = [
        f"o{j}"
        for j, o in enumerate(sig.outputs)
        if o.mode in (OutputMode.PROVIDED, OutputMode.RETURNED, OutputMode.INPLACE)
    ]
    # the node input each argument is
    arg_positions = [
        pos
        for pos in range(len(node.inputs))
        if pos not in skipped_inputs(node, folded)
    ]
    #: node input -> the output that overwrites it
    overwritten = {
        arg_positions[o.aliases[0]]: j
        for j, o in enumerate(sig.outputs)
        if o.mode is OutputMode.INPLACE
    }
    read_only = [
        f"i{pos}"
        for pos, r in zip(arg_positions, sig.inputs, strict=True)
        if isinstance(r, TensorRepr) and pos not in overwritten
    ]
    k = 2
    inputs = []
    from pytensor.link.llvm.codegen.types import is_absent

    for pos, var in enumerate(node.inputs):
        r = repr_of(var.type)
        if is_absent(var.type):
            inputs.append(None)
            continue
        if pos in folded:
            value = constant(b, var.type.dtype, folded[pos])
            if isinstance(r, ScalarRepr):
                inputs.append(value)
            else:
                slot = b.alloca(memory_type(var.type.dtype))
                store(b, var.type.dtype, value, slot)
                inputs.append(c_array(var.type.dtype, slot, [], []))
            continue
        v = params[k]
        k += 1
        r = next(sig_inputs)
        if isinstance(r, ScalarRepr):
            inputs.append(v)
        else:
            a = unpack(b, r, v)
            a.base, a.writable = pos, False
            if pos in overwritten:
                mine = f"o{overwritten[pos]}"
                b.mark_array(
                    a.data, mine, read_only + [w for w in written if w != mine]
                )
            else:
                b.mark_array(a.data, f"i{pos}", written)
            inputs.append(a)
    outputs = []
    for o in sig.outputs:
        if o.mode is OutputMode.ABSENT:
            outputs.append(None)
            continue
        if o.mode is OutputMode.INPLACE:
            source = inputs[arg_positions[o.aliases[0]]]
            outputs.append(
                InplaceOutput(
                    source.view(
                        source.data, source.shape, source.strides, source.layout
                    ),
                    source,
                )
            )
            outputs[-1].array.writable = True
            continue
        v = params[k]
        k += 1
        j = len(outputs)
        alias = (f"o{j}", read_only + [w for w in written if w != f"o{j}"])
        if o.mode is OutputMode.PROVIDED:
            a = unpack(b, o.repr, v)
            b.mark_array(a.data, *alias)
            outputs.append(ProvidedOutput(a))
        elif o.mode is OutputMode.SCALAR:
            outputs.append(ScalarOutput(v, o.repr.dtype))
        else:
            outputs.append(
                SlotOutput(
                    cg,
                    v,
                    o.repr.dtype,
                    o.repr.ndim,
                    aliases=tuple(arg_positions[i] for i in o.aliases),
                    temporary=len(outputs) in temporary_outputs,
                    # what `allocate` produces: the declared layout (a view that copies
                    # instead, like Reshape, declares C)
                    layout=o.layout if o.layout.fully_packed(o.repr.shape) else None,
                    declared=o.layout if o.aliases else None,
                    alias=alias,
                )
            )
    if kernel is not None:
        from pytensor.link.llvm.codegen.call import call_kernel

        if links is None:
            raise ValueError("a node function calling a kernel needs a links")
        ins = [
            (a, getattr(var.type, "dtype", None))
            for a, var in zip(inputs, node.inputs, strict=True)
        ]
        call_kernel(cg, kernel, links.link_kernel(kernel), ins, outputs)
    else:
        llvm_emit(node.op, node, cg, inputs, outputs)
    if not b.terminated:
        for k, out in enumerate(outputs):
            if (
                isinstance(out, SlotOutput | ScalarOutput | InplaceOutput)
                and not out.done
            ):
                raise ValueError(f"{node.op}: output {k} was never set")
        b.ret(0)
    return NodeFunction(fn, sig, links)


def build_kernel_wrapper(
    module, nf: NodeFunction, name: str, check_layouts: bool = False
):
    """`int32 name(PtCtx*, PtArray* args, const void* config)` around `nf`:
    the kernel ABI.

    `check_layouts`: verify that the arrays passed have the layouts the signature claims
    (for testing the layout inference; costs a few compares per call)."""
    sig = nf.signature
    # (outputs without a value have no argument)
    present = [o for o in sig.outputs if o.mode is not OutputMode.ABSENT]
    if sig.has_scalars:
        raise ValueError(
            "functions with ScalarType inputs or outputs have no kernel ABI"
        )
    pa = ptarray_type()
    fn = module.function(name, [ir.ptr, ir.ptr, ir.ptr], ir.i32, exported=True)
    b = fn.builder()
    ctx, args, config = fn.params

    def arg(k):
        return b.gep_bytes(args, k * PTARRAY_SIZE)

    def field(p, f):
        return b.load(ir.ptr, b.field_ptr(pa, p, f))

    def read(k, r: TensorRepr):
        p = arg(k)
        data = field(p, 0)
        if r.ndim == 0:
            return data
        shape_p, strides_p = field(p, 1), field(p, 2)
        lengths = [b.load(ir.i64, b.gep(ir.i64, shape_p, a)) for a in r.dynamic_axes]
        strides = [b.load(ir.i64, b.gep(ir.i64, strides_p, a)) for a in r.stride_axes]
        return b.aggregate(
            r.ir_type,
            [
                data,
                b.aggregate(ir.Type.array(ir.i64, len(lengths)), lengths),
                b.aggregate(ir.Type.array(ir.i64, len(strides)), strides),
            ],
        )

    if check_layouts:
        from pytensor.link.llvm.codegen.emit import Codegen

        cg = Codegen(b, ctx, config)
        reprs = [
            *sig.inputs,
            *(o.repr for o in present if o.mode is OutputMode.PROVIDED),
        ]
        positions = [
            *range(len(sig.inputs)),
            *(
                len(sig.inputs) + k
                for k, o in enumerate(present)
                if o.mode is OutputMode.PROVIDED
            ),
        ]
        for k, r in zip(positions, reprs, strict=True):
            if isinstance(r, TensorRepr) and r.ndim > 0 and len(r.stride_axes) < r.ndim:
                _check_layout(cg, arg(k), r, k)

    n_in = len(sig.inputs)
    call_args = [ctx, config]
    call_args += [read(k, r) for k, r in enumerate(sig.inputs)]
    slots = []
    for k, o in enumerate(present):
        if o.mode is OutputMode.INPLACE:
            continue  # (the caller passes the input's memory again; the input is used)
        if o.mode is OutputMode.PROVIDED:
            call_args.append(read(n_in + k, o.repr))
        else:
            slot = b.alloca(o.repr.slot_type)
            b.store(b.zero(o.repr.slot_type), slot)
            call_args.append(slot)
            slots.append((k, o, slot))
    rc = b.call(nf.function, call_args)
    # copy the slots back, also on failure (the caller releases owners)
    for k, o, slot in slots:
        p = arg(n_in + k)
        st = o.repr.slot_type
        b.store(
            b.load(ir.ptr, b.field_ptr(st, slot, Slot.DATA)),
            b.field_ptr(pa, p, PtArray.DATA),
        )
        shape_p, strides_p = field(p, 1), field(p, 2)
        lp, sp = (
            b.field_ptr(st, slot, Slot.LENGTHS),
            b.field_ptr(st, slot, Slot.STRIDES),
        )
        for a in range(o.repr.ndim):
            b.store(b.load(ir.i64, b.gep(ir.i64, lp, a)), b.gep(ir.i64, shape_p, a))
            b.store(b.load(ir.i64, b.gep(ir.i64, sp, a)), b.gep(ir.i64, strides_p, a))
        owner = b.load(ir.ptr, b.field_ptr(st, slot, Slot.OWNER))
        if len(o.aliases) == 1:
            # a view lies in the input's buffer, unless the kernel allocated one. (With
            # several inputs it may alias, a view's owner stays null: the caller keeps
            # all of them alive.)
            mine = b.ne(owner, b.null())
            owner = b.select(mine, owner, field(arg(o.aliases[0]), PtArray.OWNER))
        b.store(owner, b.field_ptr(pa, p, PtArray.OWNER))
    b.ret(rc)
    return fn


def _check_layout(cg, p, r: TensorRepr, k):
    """Fail unless the PtArray at `p` has layout `r.layout` (numpy's rules: length-1 axes
    and empty arrays don't count)."""
    from pytensor.link.llvm.codegen.types import itemsize

    b = cg.b
    pa = ptarray_type()
    shape_p = b.load(ir.ptr, b.field_ptr(pa, p, PtArray.SHAPE))
    strides_p = b.load(ir.ptr, b.field_ptr(pa, p, PtArray.STRIDES))
    shape = [b.load(ir.i64, b.gep(ir.i64, shape_p, a)) for a in range(r.ndim)]
    actual = [b.load(ir.i64, b.gep(ir.i64, strides_p, a)) for a in range(r.ndim)]
    expected = r.layout.stride_values(
        b, itemsize(r.dtype), shape, dict(enumerate(actual))
    )
    ok = b.const_bool(True)
    empty = b.const_bool(False)
    for a in range(r.ndim):
        one = b.eq(shape[a], 1)
        same = b.eq(actual[a], expected[a])
        ok = b.and_(ok, b.or_(one, same))
        empty = b.or_(empty, b.eq(shape[a], 0))
    cg.check(
        b.or_(ok, empty), f"layout check: argument {k} doesn't have layout {r.layout}"
    )


@dataclass
class CompiledNode:
    compiled: object
    signature: Signature

    @property
    def address(self):
        return self.compiled.address("kernel")


def compile_node(
    node, *, fastmath=True, opt_level=3, cache=True, provided=None
) -> CompiledNode:
    """Compile one node as a kernel (for tests and one-node units).
    `provided`: layouts of the memory the caller will provide (default: as declared)."""
    module = ir.Module(f"node_{type(node.op).__name__}", fastmath=fastmath)
    nf = build_node_function(module, node, "node", provided=provided)
    build_kernel_wrapper(module, nf, "kernel")
    compiled = module.compile(opt_level=opt_level, cache=cache)
    return CompiledNode(compiled, nf.signature)


__all__ = [
    "CompiledNode",
    "NodeFunction",
    "Output",
    "OutputMode",
    "Signature",
    "build_kernel_wrapper",
    "build_node_function",
    "compile_node",
    "inplace_outputs",
    "known_layouts",
    "llvm_emit",
    "llvm_inplace",
    "llvm_layout",
    "new_output",
    "node_outputs",
    "node_signature",
    "output_aliases",
    "pack",
]
