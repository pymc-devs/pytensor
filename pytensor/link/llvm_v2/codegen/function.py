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
from pytensor.link.llvm_v2.codegen.types import (
    PTARRAY_SIZE,
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


@dataclass(frozen=True)
class Output:
    repr: TensorRepr | ScalarRepr
    mode: OutMode
    #: the input a VIEW output lies in
    view_of: int | None = None

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
        return [m.ptr, m.ptr, m.ptr, *(r.ir_type for r in self.inputs), *(o.ir_type for o in self.outputs)]

    @property
    def has_scalars(self):
        return any(isinstance(r, ScalarRepr) for r in self.inputs) or any(
            isinstance(o.repr, ScalarRepr) for o in self.outputs
        )


@singledispatch
def llvm_output_modes(op, node) -> list[tuple[OutMode, int | None]]:
    """How each output of `node` comes to be. The default follows the types and the
    op's view_map: views, scalars, provided memory for static shapes, else allocated."""
    view_map = getattr(op, "view_map", {}) or {}
    modes = []
    for k, out in enumerate(node.outputs):
        r = repr_of(out.type)
        if isinstance(r, ScalarRepr):
            modes.append((OutMode.SCALAR, None))
        elif k in view_map:
            modes.append((OutMode.VIEW, view_map[k][0]))
        elif r.static:
            modes.append((OutMode.PROVIDED, None))
        else:
            modes.append((OutMode.ALLOCATED, None))
    return modes


def skipped_inputs(node, folded=frozenset()):
    """Input positions that aren't arguments: folded constants and None inputs."""
    from pytensor.link.llvm_v2.codegen.types import is_absent

    return frozenset(folded) | {k for k, i in enumerate(node.inputs) if is_absent(i.type)}


def node_signature(node, folded=frozenset(), in_layouts=None, out_layouts=None) -> Signature:
    """`folded`: positions of inputs that are constants folded into the code (not
    arguments). None inputs aren't arguments either.

    `in_layouts`: the `Layout` of each input; `out_layouts`: of the memory provided for
    each PROVIDED output (None: nothing known, any strides)."""
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

    if out_layouts is None:
        out_layouts = [None] * len(node.outputs)
    outputs = tuple(
        Output(repr_of(o.type, lay if mode is OutMode.PROVIDED else None), mode, arg_index(view_of))
        for o, (mode, view_of), lay in zip(node.outputs, llvm_output_modes(node.op, node), out_layouts, strict=True)
    )
    return Signature(inputs, outputs)


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
    module, node, name: str, temporary_outputs=frozenset(), folded=None, in_layouts=None, out_layouts=None
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
    sig = node_signature(node, frozenset(folded), in_layouts, out_layouts)
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
        reprs = [*sig.inputs, *(o.repr for o in sig.outputs if o.mode is OutMode.PROVIDED)]
        positions = [*range(len(sig.inputs)), *(len(sig.inputs) + k for k, o in enumerate(sig.outputs) if o.mode is OutMode.PROVIDED)]
        for k, r in zip(positions, reprs, strict=True):
            if isinstance(r, TensorRepr) and r.ndim > 0 and len(r.stride_axes) < r.ndim:
                _check_layout(cx, arg(k), r, k)

    n_in = len(sig.inputs)
    call_args = [ctx, env, ws]
    call_args += [read(k, r) for k, r in enumerate(sig.inputs)]
    slots = []
    for k, o in enumerate(sig.outputs):
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


def compile_node(node, *, fastmath=True, opt_level=3, cache=True) -> CompiledNode:
    """Compile one node as a C ABI function (for tests and one-node units)."""
    m = ir()
    module = m.Module(f"node_{type(node.op).__name__}", fastmath=fastmath)
    nf = build_node_function(module, node, "node")
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
    "llvm_output_modes",
    "node_signature",
    "pack",
]
