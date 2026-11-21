"""Ops that only change how memory is viewed, copies, and scalar/tensor conversion."""

from __future__ import annotations

from dataclasses import dataclass

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.graph.basic import Apply
from pytensor.link.llvm.analysis.layout import Layout, dimshuffle_layout, same_layout
from pytensor.link.llvm.codegen.emit import Array, Codegen, cast, copy
from pytensor.link.llvm.codegen.interface import (
    InputView,
    emit_input_views,
    llvm_emit,
    llvm_outputs,
)
from pytensor.tensor.basic import ScalarFromTensor, TensorFromScalar
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.shape import SpecifyShape


@dataclass(frozen=True)
class DimShuffleView:
    """The input with its axes reordered, length-1 axes dropped or added."""

    op: DimShuffle
    node: Apply

    def layout(self, source: Layout) -> Layout:
        return dimshuffle_layout(self.op, self.node, source)

    def apply(self, codegen: Codegen, inputs: list) -> Array:
        builder = codegen.b
        source: Array = inputs[0]
        input_shape = self.node.inputs[0].type.shape
        for axis in self.op.drop:
            if input_shape[axis] is None:
                codegen.check(
                    builder.eq(source.shape[axis], builder.i64(1)),
                    f"DimShuffle: cannot drop axis {axis} of length other than 1",
                )
        shape, strides = [], []
        for entry in self.op.new_order:
            if entry == "x":
                shape.append(builder.i64(1))
                strides.append(builder.i64(0))
            else:
                shape.append(source.shape[entry])
                strides.append(source.strides[entry])
        return source.view(source.data, shape, strides)


@dataclass(frozen=True)
class SameView:
    """The input itself (ViewOp, SpecifyShape after its checks)."""

    node: Apply

    def layout(self, source: Layout) -> Layout:
        return same_layout(self.node, source)

    def apply(self, codegen: Codegen, inputs: list) -> Array:
        source: Array = inputs[0]
        # The output's type may know lengths the input's doesn't.
        builder = codegen.b
        shape = [
            length if static is None else builder.i64(static)
            for length, static in zip(
                source.shape, self.node.outputs[0].type.shape, strict=True
            )
        ]
        return source.view(source.data, shape, source.strides)


@llvm_outputs.register(DimShuffle)
def _(op, node, input_layouts):
    return [InputView(0, DimShuffleView(op, node))]


@llvm_outputs.register(ViewOp)
@llvm_outputs.register(SpecifyShape)
def _(op, node, input_layouts):
    return [InputView(0, SameView(node))]


llvm_emit.register(DimShuffle, emit_input_views)
llvm_emit.register(ViewOp, emit_input_views)


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
    outputs[0].emit(codegen, inputs)


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
