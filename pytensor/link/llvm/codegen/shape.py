"""Ops that only change how memory is viewed, copies, and scalar/tensor conversion."""

from __future__ import annotations

from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.link.llvm.codegen.emit import Array, cast, copy
from pytensor.link.llvm.codegen.function import llvm_emit, llvm_layout
from pytensor.link.llvm.layout import dimshuffle_layout, same_layout
from pytensor.tensor.basic import ScalarFromTensor, TensorFromScalar
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.shape import SpecifyShape


@llvm_layout.register(DimShuffle)
def _(op, node, in_layouts):
    return [dimshuffle_layout(op, node, in_layouts[0])]


@llvm_layout.register(ViewOp)
@llvm_layout.register(SpecifyShape)
def _(op, node, in_layouts):
    return [same_layout(node, in_layouts[0])]


@llvm_emit.register(DimShuffle)
def _(op, node, cg, inputs, outputs):
    b = cg.b
    x = inputs[0]
    for a in (a for a in op.drop if node.inputs[0].type.shape[a] is None):
        cg.check(
            b.eq(x.shape[a], b.i64(1)),
            f"DimShuffle: cannot drop axis {a} of length other than 1",
        )
    shape, strides = [], []
    for o in op.new_order:
        if o == "x":
            shape.append(b.i64(1))
            strides.append(b.i64(0))
        else:
            shape.append(x.shape[o])
            strides.append(x.strides[o])
    # the view's layout follows from the input's (the same rule `llvm_layout` uses)
    outputs[0].set_view(
        x.view(x.data, shape, strides, dimshuffle_layout(op, node, x.layout))
    )


@llvm_emit.register(ViewOp)
def _(op, node, cg, inputs, outputs):
    outputs[0].set_view(inputs[0])


@llvm_emit.register(SpecifyShape)
def _(op, node, cg, inputs, outputs):
    """Check the given lengths (None inputs: unconstrained axes), then a view."""
    b = cg.b
    x = inputs[0]
    for a, (n, var) in enumerate(zip(inputs[1:], node.inputs[1:], strict=True)):
        if n is None:
            continue
        want = cast(
            b,
            n.load(b).value if isinstance(n, Array) else n,
            var.type.dtype,
            "int64",
        )
        cg.check(b.eq(x.shape[a], want), f"SpecifyShape: axis {a} has the wrong length")
    # the output's type may know lengths the input's doesn't: its layout is declared for those
    shape = [
        n if s is None else b.i64(s)
        for n, s in zip(x.shape, node.outputs[0].type.shape, strict=True)
    ]
    outputs[0].set_view(x.view(x.data, shape, x.strides, same_layout(node, x.layout)))


@llvm_emit.register(DeepCopyOp)
def _(op, node, cg, inputs, outputs):
    x = inputs[0]
    copy(cg, x, outputs[0].allocate(x.shape))


@llvm_emit.register(ScalarFromTensor)
def _(op, node, cg, inputs, outputs):
    x = inputs[0]
    outputs[0].set(cg.b, x.load(cg.b).value)


@llvm_emit.register(TensorFromScalar)
def _(op, node, cg, inputs, outputs):
    outputs[0].allocate([]).store(cg.b, inputs[0])
