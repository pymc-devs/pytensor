"""The kernel ABI around a node's function (or a unit's): `int32 kernel(PtCtx*, PtArray*
args, int32 n_inputs, int32 n_outputs, const void* config)`, what Rust code, the
runtime and other generated code call."""

from __future__ import annotations

from dataclasses import dataclass

from pytensor.link.llvm.codegen.emit import Codegen
from pytensor.link.llvm.codegen.node import NodeFunction, build_node_function
from pytensor.link.llvm.codegen.signature import OutputMode, Signature
from pytensor.link.llvm.codegen.types import (
    PTARRAY_SIZE,
    PtArray,
    Slot,
    TensorRepr,
    dtype_code,
    ir,
    itemsize,
    kernel_params,
    ptarray_type,
)


def build_kernel_wrapper(
    module, nf: NodeFunction, name: str, check_layouts: bool = False
):
    """`int32 name(PtCtx*, PtArray* args, int32 n_inputs, int32 n_outputs, const void*
    config)` around `nf`: the kernel ABI.

    The generated code knows its arguments from the signature, and doesn't look at
    their number, or at the `ndim` and `dtype` of the `PtArray`s.

    `check_layouts`: verify what the caller promises and can be looked at: the number
    of arguments, their `ndim` and `dtype`, and that the arrays passed have the layouts
    the signature claims (for testing the planner and the layout inference; costs a few
    compares per call)."""
    sig = nf.signature
    # (outputs without a value have no argument)
    present = [o for o in sig.outputs if o.mode is not OutputMode.ABSENT]
    if sig.has_scalars:
        raise ValueError(
            "functions with ScalarType inputs or outputs have no kernel ABI"
        )
    pa = ptarray_type()
    fn = module.function(name, kernel_params(), ir.i32, exported=True)
    b = fn.builder()
    ctx, args, n_inputs, n_outputs, config = fn.params

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
        cg = Codegen(b, ctx, config)
        cg.check(
            b.and_(
                b.eq(n_inputs, b.i32(len(sig.inputs))),
                b.eq(n_outputs, b.i32(len(present))),
            ),
            f"argument check: the kernel takes {len(sig.inputs)} inputs and "
            f"{len(present)} outputs",
        )
        for k, r in enumerate([*sig.inputs, *(o.repr for o in present)]):
            p = arg(k)
            ok = b.and_(
                b.eq(b.load(ir.i32, b.field_ptr(pa, p, PtArray.NDIM)), b.i32(r.ndim)),
                b.eq(
                    b.load(ir.i32, b.field_ptr(pa, p, PtArray.DTYPE)),
                    b.i32(dtype_code(r.dtype)),
                ),
            )
            cg.check(
                ok,
                f"argument check: argument {k} isn't a {r.ndim}-d array of {r.dtype}",
            )
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
    compiled: ir.CompiledModule
    signature: Signature

    @property
    def address(self):
        return self.compiled.address("kernel")


def compile_node(node, *, fastmath=True, opt_level=3, provided=None) -> CompiledNode:
    """Compile one node as a kernel (for tests and one-node units).
    `provided`: layouts of the memory the caller will provide (default: as declared)."""
    module = ir.Module(f"node_{type(node.op).__name__}", fastmath=fastmath)
    nf = build_node_function(module, node, "node", provided=provided)
    build_kernel_wrapper(module, nf, "kernel")
    compiled = module.compile(opt_level=opt_level)
    return CompiledNode(compiled, nf.signature)


__all__ = ["CompiledNode", "build_kernel_wrapper", "compile_node"]
