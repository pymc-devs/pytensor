"""Ops that only change how memory is viewed, copies, and scalar/tensor conversion."""

from __future__ import annotations

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.link.llvm_v2.codegen.elemwise import output_array
from pytensor.link.llvm_v2.codegen.emit import Array, cast, copy, load, store
from pytensor.link.llvm_v2.codegen.function import llvm_emit
from pytensor.link.llvm_v2.codegen.types import ir
from pytensor.link.llvm_v2.layout import llvm_view_layout
from pytensor.tensor.basic import ScalarFromTensor, TensorFromScalar
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.shape import SpecifyShape


@llvm_emit.register(DimShuffle)
def _(op, node, cx, inputs, outputs):
    b, m = cx.b, ir()
    x = inputs[0]
    shape, strides = [], []
    for o in op.new_order:
        if o == "x":
            shape.append(b.const_int(m.i64, 1))
            strides.append(b.const_int(m.i64, 0))
        else:
            shape.append(x.shape[o])
            strides.append(x.strides[o])
    # the view's layout follows from the input's (the same rule the planner uses)
    layout = llvm_view_layout(op, node, [x.layout])[0]
    outputs[0].set_view(Array(x.dtype, x.data, shape, strides, layout))


@llvm_emit.register(ViewOp)
def _(op, node, cx, inputs, outputs):
    outputs[0].set_view(inputs[0])


@llvm_emit.register(SpecifyShape)
def _(op, node, cx, inputs, outputs):
    """Check the given lengths (None inputs: unconstrained axes), then a view."""
    b, m = cx.b, ir()
    x = inputs[0]
    for a, (n, var) in enumerate(zip(inputs[1:], node.inputs[1:], strict=True)):
        if n is None:
            continue
        want = cast(b, load(b, n.dtype, n.data) if isinstance(n, Array) else n, var.type.dtype, "int64")
        cx.check(b.icmp(m.ICmp.EQ, x.shape[a], want), f"SpecifyShape: axis {a} has the wrong length")
    outputs[0].set_view(x)


@llvm_emit.register(DeepCopyOp)
def _(op, node, cx, inputs, outputs):
    x = inputs[0]
    copy(cx, x, output_array(outputs[0], x.shape))


@llvm_emit.register(ScalarFromTensor)
def _(op, node, cx, inputs, outputs):
    x = inputs[0]
    outputs[0].set(cx.b, load(cx.b, x.dtype, x.data))


@llvm_emit.register(TensorFromScalar)
def _(op, node, cx, inputs, outputs):
    out = output_array(outputs[0], [])
    store(cx.b, out.dtype, inputs[0], out.data)

