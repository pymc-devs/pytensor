"""Ops that only change how memory is viewed, copies, and scalar/tensor conversion."""

from __future__ import annotations

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.link.llvm.analysis.layout import dimshuffle_layout, same_layout
from pytensor.link.llvm.codegen.emit import Array, cast, copy
from pytensor.link.llvm.codegen.interface import llvm_emit, llvm_layout
from pytensor.tensor.basic import ScalarFromTensor, TensorFromScalar
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.shape import SpecifyShape


@llvm_layout.register(DimShuffle)
def _(op, node, input_layouts):
    return [dimshuffle_layout(op, node, input_layouts[0])]


@llvm_layout.register(ViewOp)
@llvm_layout.register(SpecifyShape)
def _(op, node, input_layouts):
    return [same_layout(node, input_layouts[0])]


@llvm_emit.register(DimShuffle)
def _(op, node, codegen, inputs, outputs):
    builder = codegen.b
    source = inputs[0]
    input_shape = node.inputs[0].type.shape
    for axis in op.drop:
        if input_shape[axis] is None:
            codegen.check(
                builder.eq(source.shape[axis], builder.i64(1)),
                f"DimShuffle: cannot drop axis {axis} of length other than 1",
            )
    shape, strides = [], []
    for entry in op.new_order:
        if entry == "x":
            shape.append(builder.i64(1))
            strides.append(builder.i64(0))
        else:
            shape.append(source.shape[entry])
            strides.append(source.strides[entry])
    outputs[0].view(source.view(source.data, shape, strides))


@llvm_emit.register(ViewOp)
def _(op, node, codegen, inputs, outputs):
    outputs[0].view(inputs[0])


@llvm_emit.register(SpecifyShape)
def _(op, node, codegen, inputs, outputs):
    """Check the given lengths (None inputs: unconstrained axes), then a view."""
    builder = codegen.b
    source = inputs[0]
    for axis, (length, var) in enumerate(zip(inputs[1:], node.inputs[1:], strict=True)):
        if length is None:
            continue
        value = length.load(builder).value if isinstance(length, Array) else length
        expected = cast(builder, value, var.type.dtype, "int64")
        codegen.check(
            builder.eq(source.shape[axis], expected),
            f"SpecifyShape: axis {axis} has the wrong length",
        )
    # The output's type may know lengths the input's doesn't.
    shape = [
        length if static is None else builder.i64(static)
        for length, static in zip(source.shape, node.outputs[0].type.shape, strict=True)
    ]
    outputs[0].view(source.view(source.data, shape, source.strides))


@llvm_emit.register(DeepCopyOp)
def _(op, node, codegen, inputs, outputs):
    source = inputs[0]
    copy(codegen, source, outputs[0].allocate(source.shape))


@llvm_emit.register(ScalarFromTensor)
def _(op, node, codegen, inputs, outputs):
    outputs[0].set(inputs[0].load(codegen.b).value)


@llvm_emit.register(TensorFromScalar)
def _(op, node, codegen, inputs, outputs):
    outputs[0].allocate([]).store(codegen.b, inputs[0])
