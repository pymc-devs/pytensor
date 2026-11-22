"""The kernel ABI around a node's function (or a unit's): `int32 kernel(PtCtx*, PtArray*
args, int32 n_inputs, int32 n_outputs, const void* config)`, what Rust code, the
runtime and other generated code call."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pytensor.graph.basic import Apply
from pytensor.link.llvm.analysis.layout import Layout
from pytensor.link.llvm.codegen.array import Array, Slot
from pytensor.link.llvm.codegen.context import Codegen
from pytensor.link.llvm.codegen.kernel_module import (
    KERNEL,
    CodegenOptions,
    KernelModule,
    link,
)
from pytensor.link.llvm.codegen.node import NodeFunction, build_node_function
from pytensor.link.llvm.codegen.signature import Output, Passing, Signature
from pytensor.link.llvm.codegen.types import (
    PTARRAY_SIZE,
    PtArray,
    TensorRepr,
    dtype_code,
    ir,
    kernel_params,
    ptarray_type,
)


def build_kernel_wrapper(
    unit: KernelModule,
    node_function: NodeFunction,
    name: str = KERNEL,
) -> ir.Function:
    """`int32 name(PtCtx*, PtArray* args, int32 n_inputs, int32 n_outputs, const void*
    config)` around `node_function`: the kernel ABI.

    The generated code knows its arguments from the signature, and doesn't look at
    their number, or at the `ndim` and `dtype` of the `PtArray`s.

    With `check_layouts` in `unit`'s options it verifies what the caller promises and
    can be looked at: the number of arguments, their `ndim` and `dtype`, and that the
    arrays passed have the layouts the signature claims (for testing the planner and
    the layout inference; costs a few compares per call)."""
    signature = node_function.signature
    # (outputs without a value have no argument)
    present = [output for output in signature.outputs if not output.is_absent]
    if signature.has_scalars:
        raise ValueError(
            "functions with ScalarType inputs or outputs have no kernel ABI"
        )
    # (without scalars, every input argument is a tensor)
    input_tensors = [
        representation
        for representation in signature.inputs
        if isinstance(representation, TensorRepr)
    ]
    ptarray = ptarray_type()
    fn = unit.module.function(name, kernel_params(), ir.i32, exported=True)
    check_layouts = unit.options.check_layouts
    b = fn.builder()
    ctx, args, n_inputs, n_outputs, config = fn.params

    def arg(k: int) -> ir.Value:
        return b.gep_bytes(args, k * PTARRAY_SIZE)

    def field(p: ir.Value, f: int) -> ir.Value:
        return b.load(ir.ptr, b.field_ptr(ptarray, p, f))

    def read(k: int, representation: TensorRepr) -> ir.Value:
        p = arg(k)
        data = field(p, PtArray.DATA)
        if representation.ndim == 0:
            return data
        shape_p, strides_p = field(p, PtArray.SHAPE), field(p, PtArray.STRIDES)
        lengths = [
            b.load(ir.i64, b.gep(ir.i64, shape_p, a))
            for a in representation.dynamic_axes
        ]
        strides = [
            b.load(ir.i64, b.gep(ir.i64, strides_p, a))
            for a in representation.stride_axes
        ]
        return b.aggregate(
            representation.ir_type,
            [
                data,
                b.aggregate(ir.Type.array(ir.i64, len(lengths)), lengths),
                b.aggregate(ir.Type.array(ir.i64, len(strides)), strides),
            ],
        )

    if check_layouts:
        cg = Codegen(b, ctx, config, unit, name=name)
        cg.check(
            b.and_(
                b.eq(n_inputs, b.i32(len(signature.inputs))),
                b.eq(n_outputs, b.i32(len(present))),
            ),
            f"argument check: the kernel takes {len(signature.inputs)} inputs and "
            f"{len(present)} outputs",
        )
        for k, representation in enumerate(
            [*input_tensors, *(output.tensor for output in present)]
        ):
            p = arg(k)
            ok = b.and_(
                b.eq(
                    b.load(ir.i32, b.field_ptr(ptarray, p, PtArray.NDIM)),
                    b.i32(representation.ndim),
                ),
                b.eq(
                    b.load(ir.i32, b.field_ptr(ptarray, p, PtArray.DTYPE)),
                    b.i32(dtype_code(representation.scalar_type)),
                ),
            )
            cg.check(
                ok,
                f"argument check: argument {k} isn't a {representation.ndim}-d array of {representation.scalar_type}",
            )
        reprs = [
            *input_tensors,
            *(
                output.tensor
                for output in present
                if output.passing is Passing.CALLER_BUFFER
            ),
        ]
        positions = [
            *range(len(signature.inputs)),
            *(
                len(signature.inputs) + k
                for k, output in enumerate(present)
                if output.passing is Passing.CALLER_BUFFER
            ),
        ]
        for k, representation in zip(positions, reprs, strict=True):
            ndim = representation.ndim
            if ndim > 0 and len(representation.stride_axes) < ndim:
                _check_layout(cg, arg(k), representation, k)

    n_in = len(signature.inputs)
    call_args: list[ir.Value] = [ctx, config]
    call_args += [
        read(k, representation) for k, representation in enumerate(input_tensors)
    ]
    slots: list[tuple[int, Output, Slot]] = []
    for k, output in enumerate(present):
        if output.passing is Passing.CALLER_BUFFER:
            call_args.append(read(n_in + k, output.tensor))
        else:
            slot = Slot.alloca(b, output.tensor.ndim)
            call_args.append(slot.pointer)
            slots.append((k, output, slot))
    rc = b.call(node_function.function, call_args)
    # copy the slots back, also on failure (the caller releases owners)
    for k, output, slot in slots:
        p = arg(n_in + k)
        b.store(
            slot.data(b),
            b.field_ptr(ptarray, p, PtArray.DATA),
        )
        shape_p, strides_p = field(p, PtArray.SHAPE), field(p, PtArray.STRIDES)
        lp, sp = slot.field(b, Slot.LENGTHS), slot.field(b, Slot.STRIDES)
        for a in range(output.tensor.ndim):
            b.store(b.load(ir.i64, b.gep(ir.i64, lp, a)), b.gep(ir.i64, shape_p, a))
            b.store(b.load(ir.i64, b.gep(ir.i64, sp, a)), b.gep(ir.i64, strides_p, a))
        owner = slot.owner(b)
        alias_arguments = signature.alias_arguments(output)
        if len(alias_arguments) == 1:
            # a view lies in the input's buffer, unless the kernel allocated one. (With
            # several inputs it may alias, a view's owner stays null: the caller keeps
            # all of them alive.)
            mine = b.ne(owner, b.null())
            input_owner = field(arg(alias_arguments[0]), PtArray.OWNER)
            owner = b.select(mine, owner, input_owner)
        b.store(owner, b.field_ptr(ptarray, p, PtArray.OWNER))
    assert rc is not None  # (it returns an i32)
    b.ret(rc)
    return fn


def _check_layout(
    cg: Codegen, pointer: ir.Value, representation: TensorRepr, k: int
) -> None:
    """Fail unless the PtArray at `pointer` has layout `representation.layout`."""
    b = cg.b
    ptarray = ptarray_type()
    shape_pointer = b.load(ir.ptr, b.field_ptr(ptarray, pointer, PtArray.SHAPE))
    strides_pointer = b.load(ir.ptr, b.field_ptr(ptarray, pointer, PtArray.STRIDES))
    axes = range(representation.ndim)
    shape = [b.load(ir.i64, b.gep(ir.i64, shape_pointer, a)) for a in axes]
    strides = [b.load(ir.i64, b.gep(ir.i64, strides_pointer, a)) for a in axes]
    data = b.load(ir.ptr, b.field_ptr(ptarray, pointer, PtArray.DATA))
    array = Array(
        representation.scalar_type, data, shape, strides, representation.layout
    )
    array.check_strides(cg, representation.layout, f"argument {k}")


@dataclass
class CompiledNode:
    """A node compiled as a kernel (`compile_node`)."""

    unit: KernelModule
    signature: Signature

    @property
    def compiled(self) -> ir.CompiledModule:
        assert self.unit.compiled is not None
        return self.unit.compiled

    @property
    def address(self) -> int:
        return self.unit.address

    @property
    def config(self) -> int:
        return self.unit.config


def compile_node(
    node: Apply,
    *,
    fastmath: bool = True,
    opt_level: int = 3,
    provided: Sequence[Layout | None] | None = None,
) -> CompiledNode:
    """Compile one node as a kernel (for tests and one-node units).
    `provided`: layouts of the memory the caller will provide (default: as declared)."""
    unit = KernelModule.new(
        f"node_{type(node.op).__name__}", CodegenOptions(fastmath=fastmath)
    )
    node_function = build_node_function(unit, node, "node", provided=provided)
    build_kernel_wrapper(unit, node_function)
    link([unit], opt_level)
    return CompiledNode(unit, node_function.signature)


__all__ = [
    "CompiledNode",
    "build_kernel_wrapper",
    "compile_node",
]
