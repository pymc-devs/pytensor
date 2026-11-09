"""Generated functions: signatures, node functions and C ABI wrappers.

See DESIGN.md, sections 2, 3 and 6.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import singledispatch

from pytensor.link.llvm_v2.codegen.emit import (
    EmitContext,
    ProvidedOutput,
    ScalarOutput,
    SlotOutput,
    pack,
    unpack,
)
from pytensor.link.llvm_v2.layout import Layout
from pytensor.link.llvm_v2.codegen.types import (
    PTARRAY_SIZE,
    NoneRepr,
    ScalarRepr,
    TensorRepr,
    ir,
    ptarray_type,
    repr_of,
)


class OutMode(enum.Enum):
    #: the caller provides the memory (with its own strides)
    PROVIDED = "provided"
    #: the callee allocates and hands over one reference through the slot
    ALLOCATED = "allocated"
    #: the callee returns a view of an input through the slot
    VIEW = "view"
    #: a ScalarType output, stored through a pointer
    SCALAR = "scalar"
    #: no value (a random generator token): no argument
    ABSENT = "absent"


@dataclass(frozen=True)
class Output:
    repr: TensorRepr | ScalarRepr
    mode: OutMode
    #: the input a VIEW output lies in
    view_of: int | None = None

    @property
    def layout(self):
        """The output's layout (None for scalars): for PROVIDED the memory the op asks
        for (the provider may give it other memory, see `with_layout`), for ALLOCATED
        what the code allocates, for VIEW what the view has."""
        return getattr(self.repr, "layout", None)

    def with_layout(self, layout):
        """The same output with another layout (e.g. the memory actually provided)."""
        from dataclasses import replace

        return replace(self, repr=TensorRepr(self.repr.dtype, self.repr.shape, layout))

    @property
    def ir_type(self):
        m = ir()
        if self.mode is OutMode.PROVIDED:
            return self.repr.ir_type
        return m.ptr


@dataclass(frozen=True)
class Signature:
    inputs: tuple
    outputs: tuple[Output, ...]

    def param_types(self):
        m = ir()
        outs = [o.ir_type for o in self.outputs if o.mode is not OutMode.ABSENT]
        return [m.ptr, m.ptr, m.ptr, *(r.ir_type for r in self.inputs), *outs]

    @property
    def has_scalars(self):
        return any(isinstance(r, ScalarRepr) for r in self.inputs) or any(
            isinstance(o.repr, ScalarRepr) for o in self.outputs
        )


@singledispatch
def llvm_outputs(op, node, in_layouts) -> list[Output]:
    """How each output of `node` comes to be, and its layout (layout.py), given the
    inputs' layouts (`known_layouts` fills in the unknown ones).

    The default follows the types and the op's view_map: a view of unknown layout,
    scalars, provided memory for static shapes, else allocated; memory C-contiguous.
    Ops register their own to describe views (DimShuffle, Subtensor, ...) or to choose
    where and how their outputs are produced (e.g. Elemwise allocating its output in
    its inputs' memory order). The code must produce what is declared: ALLOCATED
    outputs in the layout given (`SlotOutput.allocate` does that), VIEW outputs with
    the layout claimed (`SlotOutput.set_view` checks). PROVIDED memory has the declared
    layout (packed, or any strides); a provider that can't give it writes the output
    other memory only through a temporary and a copy."""
    view_map = getattr(op, "view_map", {}) or {}
    return [
        Output(repr_of(o.type, Layout.dynamic(o.type.ndim)), OutMode.VIEW, view_map[k][0])
        if k in view_map and isinstance(repr_of(o.type), TensorRepr)
        else fresh_output(o)
        for k, o in enumerate(node.outputs)
    ]


def fresh_output(var, layout=None) -> Output:
    """An output in new memory: SCALAR, PROVIDED for static shapes, else ALLOCATED; in
    `layout` (default C-contiguous). ABSENT for values without data."""
    r = repr_of(var.type)
    if isinstance(r, NoneRepr):
        return Output(r, OutMode.ABSENT)
    if isinstance(r, ScalarRepr):
        return Output(r, OutMode.SCALAR)
    lay = layout if layout is not None else Layout.c(var.type.ndim)
    return Output(repr_of(var.type, lay), OutMode.PROVIDED if r.static else OutMode.ALLOCATED)


def view_output(var, layout, view_of=0) -> Output:
    return Output(repr_of(var.type, layout), OutMode.VIEW, view_of)


def known_layouts(node, in_layouts) -> list:
    """`in_layouts` with DYNAMIC for tensors whose layout isn't known."""
    if in_layouts is None:
        in_layouts = [None] * len(node.inputs)
    return [
        lay if lay is not None or not hasattr(i.type, "ndim") else Layout.dynamic(i.type.ndim)
        for i, lay in zip(node.inputs, in_layouts, strict=True)
    ]


def node_outputs(node, in_layouts=None) -> list[Output]:
    return llvm_outputs(node.op, node, known_layouts(node, in_layouts))


def skipped_inputs(node, folded=frozenset()):
    """Input positions that aren't arguments: folded constants and None inputs."""
    from pytensor.link.llvm_v2.codegen.types import is_absent

    return frozenset(folded) | {k for k, i in enumerate(node.inputs) if is_absent(i.type)}


def node_signature(node, folded=frozenset(), in_layouts=None, provided=None) -> Signature:
    """`folded`: positions of inputs that are constants folded into the code (not
    arguments). None inputs aren't arguments either.

    `in_layouts`: the `Layout` of each input; outputs as `llvm_outputs` declares them.
    The declared layout of a PROVIDED output is binding: whoever provides the memory
    provides it in that layout. `provided[k]` replaces the declaration, for a caller
    that generates a function for memory it fixes (tests writing into a slice)."""
    folded = skipped_inputs(node, folded)
    if in_layouts is None:
        in_layouts = [None] * len(node.inputs)
    inputs = tuple(repr_of(i.type, lay) for k, (i, lay) in enumerate(zip(node.inputs, in_layouts, strict=True)) if k not in folded)

    def arg_index(pos):
        # position among the function's (non-folded) inputs
        if pos is None:
            return None
        if pos in folded:
            raise NotImplementedError("a view of a folded constant")
        return pos - sum(1 for p in folded if p < pos)

    outputs = []
    for k, o in enumerate(node_outputs(node, in_layouts)):
        if o.mode is OutMode.PROVIDED and provided is not None and provided[k] is not None:
            o = o.with_layout(provided[k])
        if o.view_of is not None:
            o = Output(o.repr, o.mode, arg_index(o.view_of))
        outputs.append(o)
    return Signature(inputs, tuple(outputs))


@singledispatch
def llvm_emit(op, node, cx: EmitContext, inputs: list, outputs: list) -> None:
    """Emit the code of `node`.

    `inputs` are `Array`s (tensors) or values (scalars); `outputs` are `ProvidedOutput`,
    `SlotOutput` or `ScalarOutput` objects to fill.
    """
    raise NotImplementedError(f"No LLVM backend implementation for {op}")


@dataclass
class NodeFunction:
    function: object
    signature: Signature
    #: bytes of static workspace
    workspace: int


def build_node_function(
    module, node, name: str, temporary_outputs=frozenset(), folded=None, in_layouts=None, provided=None
) -> NodeFunction:
    """The LLVM function of one node (DESIGN.md, section 2), always inlined.

    * `temporary_outputs`: allocated outputs that are temporaries of a compiled unit;
      they go into the workspace instead of runtime buffers;
    * `folded`: {input position: value} for 0-d constant inputs compiled into the code.
    """
    from pytensor.link.llvm_v2.codegen.emit import c_array, constant, store
    from pytensor.link.llvm_v2.codegen.types import memory_type

    m = ir()
    folded = folded or {}
    sig = node_signature(node, frozenset(folded), in_layouts, provided)
    sig_inputs = iter(sig.inputs)
    fn = module.function(name, sig.param_types(), m.i32, inline=m.Inline.Always)
    b = fn.builder()
    params = fn.params
    ctx, env, ws = params[:3]
    cx = EmitContext(b, ctx, env, ws)
    k = 3
    inputs = []
    from pytensor.link.llvm_v2.codegen.types import is_absent

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
        inputs.append(v if isinstance(r, ScalarRepr) else unpack(b, r, v))
    outputs = []
    for o in sig.outputs:
        if o.mode is OutMode.ABSENT:
            outputs.append(None)
            continue
        v = params[k]
        k += 1
        if o.mode is OutMode.PROVIDED:
            outputs.append(ProvidedOutput(unpack(b, o.repr, v)))
        elif o.mode is OutMode.SCALAR:
            outputs.append(ScalarOutput(v, o.repr.dtype))
        else:
            outputs.append(
                SlotOutput(
                    cx,
                    v,
                    o.repr.dtype,
                    o.repr.ndim,
                    view_of=o.view_of,
                    temporary=len(outputs) in temporary_outputs and o.mode in (OutMode.ALLOCATED, OutMode.VIEW),
                    # what `allocate` produces: the declared layout (a view that copies
                    # instead, like Reshape, declares C)
                    layout=o.layout if o.layout.fully_packed(o.repr.shape) else None,
                    declared=o.layout if o.mode is OutMode.VIEW else None,
                )
            )
    llvm_emit(node.op, node, cx, inputs, outputs)
    if not b.terminated:
        for k, out in enumerate(outputs):
            if isinstance(out, SlotOutput | ScalarOutput) and not out.done:
                raise ValueError(f"{node.op}: output {k} was never set")
        b.ret(b.const_int(m.i32, 0))
    return NodeFunction(fn, sig, cx.ws_static)


def build_abi_wrapper(module, nf: NodeFunction, name: str, check_layouts: bool = False):
    """`int32 name(PtCtx*, PtArray* args, const void* env, void* ws)` around `nf`.

    `check_layouts`: verify that the arrays passed have the layouts the signature claims
    (for testing the layout inference; costs a few compares per call)."""
    m = ir()
    sig = nf.signature
    # (outputs without a value have no argument)
    present = [o for o in sig.outputs if o.mode is not OutMode.ABSENT]
    if sig.has_scalars:
        raise ValueError("functions with ScalarType inputs or outputs have no C ABI")
    pa = ptarray_type()
    fn = module.function(name, [m.ptr, m.ptr, m.ptr, m.ptr], m.i32, exported=True)
    b = fn.builder()
    ctx, args, env, ws = fn.params

    def arg(k):
        return b.gep_bytes(args, b.const_int(m.i64, k * PTARRAY_SIZE))

    def field(p, f):
        return b.load(m.ptr, b.field_ptr(pa, p, f))

    def read(k, r: TensorRepr):
        p = arg(k)
        data = field(p, 0)
        if r.ndim == 0:
            return data
        shape_p, strides_p = field(p, 1), field(p, 2)
        lengths = [b.load(m.i64, b.gep(m.i64, shape_p, b.const_int(m.i64, a))) for a in r.dynamic_axes]
        strides = [b.load(m.i64, b.gep(m.i64, strides_p, b.const_int(m.i64, a))) for a in r.stride_axes]
        return b.aggregate(
            r.ir_type,
            [
                data,
                b.aggregate(m.Type.array(m.i64, len(lengths)), lengths),
                b.aggregate(m.Type.array(m.i64, len(strides)), strides),
            ],
        )

    if check_layouts:
        from pytensor.link.llvm_v2.codegen.emit import EmitContext

        cx = EmitContext(b, ctx, env, ws)
        reprs = [*sig.inputs, *(o.repr for o in present if o.mode is OutMode.PROVIDED)]
        positions = [*range(len(sig.inputs)), *(len(sig.inputs) + k for k, o in enumerate(present) if o.mode is OutMode.PROVIDED)]
        for k, r in zip(positions, reprs, strict=True):
            if isinstance(r, TensorRepr) and r.ndim > 0 and len(r.stride_axes) < r.ndim:
                _check_layout(cx, arg(k), r, k)

    n_in = len(sig.inputs)
    call_args = [ctx, env, ws]
    call_args += [read(k, r) for k, r in enumerate(sig.inputs)]
    slots = []
    for k, o in enumerate(present):
        if o.mode is OutMode.PROVIDED:
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
        b.store(b.load(m.ptr, b.field_ptr(st, slot, 0)), b.field_ptr(pa, p, 0))
        shape_p, strides_p = field(p, 1), field(p, 2)
        lp, sp = b.field_ptr(st, slot, 1), b.field_ptr(st, slot, 2)
        for a in range(o.repr.ndim):
            i = b.const_int(m.i64, a)
            b.store(b.load(m.i64, b.gep(m.i64, lp, i)), b.gep(m.i64, shape_p, i))
            b.store(b.load(m.i64, b.gep(m.i64, sp, i)), b.gep(m.i64, strides_p, i))
        owner = b.load(m.ptr, b.field_ptr(st, slot, 3))
        if o.mode is OutMode.VIEW:
            # a view lies in the input's buffer, unless the callee allocated one
            mine = b.icmp(m.ICmp.NE, owner, b.null())
            owner = b.select(mine, owner, field(arg(o.view_of), 3))
        b.store(owner, b.field_ptr(pa, p, 3))
    b.ret(rc)
    return fn


def _check_layout(cx, p, r: TensorRepr, k):
    """Fail unless the PtArray at `p` has layout `r.layout` (numpy's rules: length-1 axes
    and empty arrays don't count)."""
    from pytensor.link.llvm_v2.codegen.emit import _and
    from pytensor.link.llvm_v2.codegen.types import itemsize

    b, m = cx.b, ir()
    pa = ptarray_type()
    shape_p = b.load(m.ptr, b.field_ptr(pa, p, 1))
    strides_p = b.load(m.ptr, b.field_ptr(pa, p, 2))
    shape = [b.load(m.i64, b.gep(m.i64, shape_p, b.const_int(m.i64, a))) for a in range(r.ndim)]
    actual = [b.load(m.i64, b.gep(m.i64, strides_p, b.const_int(m.i64, a))) for a in range(r.ndim)]
    expected = r.layout.stride_values(b, itemsize(r.dtype), shape, dict(enumerate(actual)))
    ok = b.const_bool(True)
    empty = b.const_bool(False)
    for a in range(r.ndim):
        one = b.icmp(m.ICmp.EQ, shape[a], b.const_int(m.i64, 1))
        same = b.icmp(m.ICmp.EQ, actual[a], expected[a])
        ok = _and(b, ok, b.or_(one, same))
        empty = b.or_(empty, b.icmp(m.ICmp.EQ, shape[a], b.const_int(m.i64, 0)))
    cx.check(b.or_(ok, empty), f"layout check: argument {k} doesn't have layout {r.layout}")


@dataclass
class CompiledNode:
    compiled: object
    signature: Signature
    workspace: int

    @property
    def address(self):
        return self.compiled.address("abi")


def compile_node(node, *, fastmath=True, opt_level=3, cache=True, provided=None) -> CompiledNode:
    """Compile one node as a C ABI function (for tests and one-node units).
    `provided`: layouts of the memory the caller will provide (default: as declared)."""
    m = ir()
    module = m.Module(f"node_{type(node.op).__name__}", fastmath=fastmath)
    nf = build_node_function(module, node, "node", provided=provided)
    build_abi_wrapper(module, nf, "abi")
    compiled = module.compile(opt_level=opt_level, cache=cache)
    return CompiledNode(compiled, nf.signature, nf.workspace)


__all__ = [
    "CompiledNode",
    "NodeFunction",
    "OutMode",
    "Output",
    "Signature",
    "build_abi_wrapper",
    "build_node_function",
    "compile_node",
    "llvm_emit",
    "fresh_output",
    "known_layouts",
    "llvm_outputs",
    "node_outputs",
    "view_output",
    "node_signature",
    "pack",
]
