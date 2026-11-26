"""Ops that only change how memory is viewed, copies, and scalar/tensor conversion."""

from __future__ import annotations

import pytensor.scalar as ps
from pytensor.compile.ops import DeepCopyOp, TypeCastingOp
from pytensor.link.llvm.analysis.layout import PACKED, Layout, Stride
from pytensor.link.llvm.codegen.array import Array
from pytensor.link.llvm.codegen.interface import llvm_emit, llvm_layout
from pytensor.link.llvm.codegen.num import cast
from pytensor.link.llvm.codegen.types import element_type
from pytensor.tensor.basic import ScalarFromTensor, TensorFromScalar
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.shape import SpecifyShape


@llvm_layout.register(DimShuffle)
def _(op, node, input_layouts) -> list[Layout | None]:
    """A view with the input's axes reordered, and new axes of length 1 added. The axes
    keep their strides; a new axis is packed."""
    x = input_layouts[0]
    out_of = {}  # input axis -> output axis
    strides: list[Stride] = []
    for j, entry in enumerate(op.new_order):
        if entry == "x":
            strides.append(PACKED)
        else:
            out_of[int(entry)] = j
            strides.append(x.strides[int(entry)])
    order = [j for j, entry in enumerate(op.new_order) if entry == "x"] + [
        out_of[a] for a in x.order if a in out_of
    ]
    return [Layout.make(node.outputs[0].type.shape, order, strides)]


@llvm_layout.register(TypeCastingOp)
@llvm_layout.register(SpecifyShape)
def _(op, node, input_layouts) -> list[Layout | None]:
    """A view that is its input, for the output's possibly more static shape."""
    x = input_layouts[0]
    # A cast of a value without data, like the xtensor rng casts, has no layout.
    return [None if x is None else x.for_shape(node.outputs[0].type.shape)]


@llvm_emit.register(DimShuffle)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    source = inputs[0]
    input_shape = node.inputs[0].type.shape
    for axis in op.drop:
        if input_shape[axis] is None:
            cg.check(
                b.eq(source.shape[axis], b.i64(1)),
                f"DimShuffle: cannot drop axis {axis} of length other than 1",
            )
    shape, strides = [], []
    for entry in op.new_order:
        if entry == "x":
            shape.append(b.i64(1))
            strides.append(b.i64(0))
        else:
            shape.append(source.shape[entry])
            strides.append(source.strides[entry])
    outputs[0].set_view(source.view(source.data, shape, strides))


@llvm_emit.register(TypeCastingOp)
def _(op, node, cg, inputs, outputs):
    if outputs[0] is not None:
        outputs[0].set_view(inputs[0])


@llvm_emit.register(SpecifyShape)
def _(op, node, cg, inputs, outputs):
    """Check the given lengths (None inputs: unconstrained axes), then a view."""
    b = cg.b
    source = inputs[0]
    for axis, (length, var) in enumerate(zip(inputs[1:], node.inputs[1:], strict=True)):
        if length is None:
            continue
        value = length.load(b).value if isinstance(length, Array) else length
        expected = cast(b, value, element_type(var.type), ps.int64)
        cg.check(
            b.eq(source.shape[axis], expected),
            f"SpecifyShape: axis {axis} has the wrong length",
            AssertionError,  # as `SpecifyShape.perform`
        )
    # The output's type may know lengths the input's doesn't.
    shape = [
        length if static is None else b.i64(static)
        for length, static in zip(source.shape, node.outputs[0].type.shape, strict=True)
    ]
    outputs[0].set_view(source.view(source.data, shape, source.strides))


@llvm_emit.register(DeepCopyOp)
def _(op, node, cg, inputs, outputs):
    # (nothing to copy of a value without data, like a generator: `compile_program`
    # returns a copy of it)
    if outputs[0] is not None:
        source = inputs[0]
        outputs[0].allocate(source.shape).copy_from(cg, source)


@llvm_emit.register(ScalarFromTensor)
def _(op, node, cg, inputs, outputs):
    outputs[0].set(inputs[0].load(cg.b).value)


@llvm_emit.register(TensorFromScalar)
def _(op, node, cg, inputs, outputs):
    outputs[0].allocate([]).store(cg.b, inputs[0])
