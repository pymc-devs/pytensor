from functools import partial

from egglog import String, birewrite, eq, i64, rewrite, rule

from pytensor.sandbox.scrambled.basic import (
    Int,
    IntTuple,
    ScalarOp,
    Tensor,
    TensorTuple,
    egraph,
)
from pytensor.sandbox.scrambled.op import (
    Add,
    Alloc,
    BroadcastShapes,
    BroadcastShapesOfTensors,
    Div,
    Elemwise,
    ExpandDims,
    Mul,
    Reduce,
    Shape,
    Shape_i,
    Squeeze,
    Sub,
    Sum,
)


@egraph.register
def inline_tensor_shape(
    name: String,
    static_sh: IntTuple,
    value: Int,
):
    yield rewrite(Shape(Tensor.constant(value, static_sh))).to(
        TensorTuple.from_int_tuple(static_sh)
    )
    x = Shape(Tensor(name, static_sh))
    yield rewrite(x).to(TensorTuple.from_tensor_shape(x, static_sh, 0))


@egraph.register
def shape_i_rules(
    x: Tensor,
    i: Int,
):
    yield rewrite(Shape(x)[i]).to(Shape_i(x, i))


def reduce_op_axis_variants(
    op,
    x: Tensor,
    axis: IntTuple,
    i: i64,
    j: i64,
):
    """Rules for tuple axis Ops, where the leading sorted axis is applied last.

    This is the case for most reduction Ops.
    E.g, Sum(axis=(0, 1))(x) == Sum(axis=0)(Sum(axis=1)(x)
    """

    # Split into consecutive axis applications
    yield rewrite(op(axis=(i,) + axis)(x)).to(op(axis=(i,))(op(axis=axis)(x)))
    yield rewrite(op(axis=IntTuple.empty())(x)).to(x)
    # Swap consecutive axis applications
    yield rewrite(op(axis=(i,))(op(axis=(j,))(x))).to(
        op(axis=(j - 1,))(op(axis=(i,))(x)),
        i < j,
    )

    yield rewrite(op(axis=(i,))(op(axis=(j,))(x))).to(
        op(axis=(j,))(op(axis=(i + 1,))(x)),
        i >= j,
    )

    # Merge from consecutive axis applications
    yield rewrite(op(axis=(i,))(op(axis=(j,))(x))).to(
        op(axis=(i, j))(x),
        i < j,
    )

    yield rewrite(op(axis=(i,))(op(axis=(j,) + axis)(x))).to(
        op(axis=(i,) + (j + axis))(x),
        i < j,
    )


@egraph.register
def squeeze_rules(
    x: Tensor,
    axis: IntTuple,
    i: i64,
    j: i64,
):
    yield from reduce_op_axis_variants(Squeeze, x, axis, i, j)

    # Squeeze.shape
    yield rewrite(Shape(Squeeze(axis=(i,))(x))).to(Shape(x).pop(i))


@egraph.register
def expand_dims_rules(
    x: Tensor,
    axis: IntTuple,
    i: i64,
    j: i64,
):
    # Useless
    yield rewrite(ExpandDims(IntTuple.empty())(x)).to(x)

    # Split into consecutive axis applications
    yield rewrite(ExpandDims(axis=(i,) + axis)(x)).to(
        ExpandDims(axis=axis)(ExpandDims(axis=(i,))(x))
    )

    # Swap consecutive axis applications
    yield rewrite(ExpandDims(axis=(i,))(ExpandDims(axis=(j,))(x))).to(
        ExpandDims(axis=(j + 1,))(ExpandDims(axis=(i,))(x)),
        i <= j,
    )

    yield rewrite(ExpandDims(axis=(i,))(ExpandDims(axis=(j,))(x))).to(
        ExpandDims(axis=(j,))(ExpandDims(axis=(i - 1,))(x)),
        i > j,
    )

    # Merge from consecutive axis applications
    yield rewrite(ExpandDims(axis=(i,))(ExpandDims(axis=(j,))(x))).to(
        ExpandDims(axis=(j, i))(x),
        i > j,
    )

    yield rewrite(ExpandDims(axis=(i,) + axis)(ExpandDims(axis=(j,))(x))).to(
        ExpandDims(axis=(j,) + (i + axis))(x),
        i > j,
    )

    # ExpandDims.shape
    yield rewrite(Shape(ExpandDims(axis=(i,))(x))).to(Shape(x).insert(i, 1))


@egraph.register
def useless_expand_squeeze(
    x: Tensor,
    i: i64,
):
    yield rewrite(ExpandDims((i,))(Squeeze((i,))(x))).to(x)
    yield rewrite(Squeeze((i,))(ExpandDims((i,))(x))).to(x)


@egraph.register
def alloc_rules(
    x: Tensor,
    y: Tensor,
    sh: TensorTuple,
    sh2: TensorTuple,
    static_sh: IntTuple,
    v: Int,
):
    # Alloc.shape
    yield rewrite(Shape(Alloc(x, sh))).to(sh)

    # Introduce x.shape
    yield rule(eq(y).to(Alloc(x, sh))).then(Shape(x))

    # Useless Alloc
    yield rewrite(Alloc(x, sh)).to(x, eq(Shape(x)).to(sh))

    # Rewrite constants as Allocs. Constant have static shape type, while Alloc have symbolic types
    yield rewrite(Tensor.constant(v, static_sh)).to(
        Alloc(Tensor.constant(v), TensorTuple.from_int_tuple(static_sh))
    )

    # A more safe rewrite is Alloc(x, sh ^ sh2)
    yield rewrite(Alloc(Alloc(x, sh), sh2)).to(Alloc(x, sh2))


@egraph.register
def expand_squeeze_of_alloc(
    x: Tensor,
    sh: TensorTuple,
):
    yield rewrite(ExpandDims(axis=0)(Alloc(x, sh))).to(Alloc(x, (1,) + sh))
    yield rewrite(Squeeze(axis=0)(Alloc(x, (1,) + sh))).to(Alloc(x, sh))

    # Could be anything with shape[0]==1, but then would need to introduce a Squeeze
    yield rewrite(Alloc(ExpandDims(axis=0)(x), sh)).to(Alloc(x, sh))


@egraph.register
def broadcast_shapes_rules(
    s1: TensorTuple,
    s2: TensorTuple,
    s3: TensorTuple,
    d: Tensor,
):
    yield rewrite(BroadcastShapes(s1, s2)).to(s1 ^ s2)

    # Commutativity
    yield rewrite(s1 ^ s2).to(s2 ^ s1)

    # Introduce shape lengths
    yield rule(eq(s3).to(s1 ^ s2)).then(
        s1.length() > s2.length(),
        # s2.length() > 0,
    )

    # Pad to same length
    yield rewrite(s1 ^ s2).to(
        s1 ^ (TensorTuple(1) + s2),
        eq(s1.length() > s2.length()).to(Int(1)),
        # eq(s2.length() > 0).to(Int(1)),
    )

    # Apply broadcast shapes logic
    yield rewrite(s1 ^ s1).to(s1)
    yield rewrite(s1 ^ TensorTuple.empty()).to(s1)
    yield rewrite(TensorTuple(d) ^ TensorTuple(1)).to(TensorTuple(d))
    yield rewrite(((d,) + s1) ^ ((d,) + s2)).to(
        ((d,) + (s1 ^ s2)), eq(s1.length()).to(s2.length())
    )
    yield rewrite(((d,) + s1) ^ ((1,) + s2)).to(
        ((d,) + (s1 ^ s2)), eq(s1.length()).to(s2.length())
    )


@egraph.register
def broadcast_shapes_of_tensors_rules(
    x: Tensor,
    tensors: TensorTuple,
):
    yield rewrite(BroadcastShapesOfTensors(TensorTuple.empty())).to(TensorTuple.empty())
    yield rewrite(BroadcastShapesOfTensors(TensorTuple(x))).to(Shape(x))
    yield rewrite(BroadcastShapesOfTensors(TensorTuple(x) + tensors)).to(
        Shape(x) ^ BroadcastShapesOfTensors(tensors)
    )


@egraph.register
def elemwise_rules(
    op: ScalarOp,
    x: Tensor,
    tensors: TensorTuple,
    tensors2: TensorTuple,
    sh: TensorTuple,
):
    elemwise_op = Elemwise(op)
    yield rewrite(Shape(elemwise_op(tensors))).to(BroadcastShapesOfTensors(tensors))

    dims_needed = sh.length() - Shape(x).length()
    expanded_x = ExpandDims(IntTuple.from_range(0, dims_needed))(x)

    yield rewrite(elemwise_op(tensors + (Alloc(x, sh),))).to(
        Alloc(
            elemwise_op(tensors + (expanded_x,)),
            sh ^ BroadcastShapesOfTensors(tensors),
        )
    )

    yield rewrite(elemwise_op((Alloc(x, sh),) + tensors2)).to(
        Alloc(
            elemwise_op((expanded_x,) + tensors2),
            sh ^ BroadcastShapesOfTensors(tensors2),
        )
    )

    yield rewrite(elemwise_op(tensors + (Alloc(x, sh),) + tensors2)).to(
        Alloc(
            elemwise_op(tensors + (expanded_x,) + tensors2),
            sh ^ BroadcastShapesOfTensors(tensors) ^ BroadcastShapesOfTensors(tensors2),
        )
    )


@egraph.register
def elemwise_arithmetic_rules(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    sh: TensorTuple,
):
    yield rewrite(x + y).to(Add((x, y)))
    yield rewrite(x - y).to(Sub((x, y)))
    yield rewrite(x * y).to(Mul((x, y)))

    zero = Alloc(Tensor.constant(0), sh)
    one = Alloc(Tensor.constant(1), sh)

    yield rewrite(Add((zero, x))).to(Alloc(x, sh ^ Shape(x)))
    yield rewrite(Sub((x, zero))).to(Alloc(x, Shape(x) ^ sh))
    yield rewrite(Sub((x, x))).to(Alloc(Tensor.constant(0), Shape(x)))
    yield rewrite(Add((x, y))).to(Add((y, x)))
    yield rewrite(Add((Add((x, y)), z))).to(Add((x, Add((y, z)))))

    yield rewrite(Mul((one, x))).to(Alloc(x, sh ^ Shape(x)))

    yield rewrite(Mul((zero, x))).to(Alloc(Tensor.constant(0), sh ^ Shape(x)))
    yield rewrite(Mul((x, y))).to(Mul((y, x)))
    # FIXME: This blows up when there's a mul by zero :D
    # yield rewrite(Mul((Mul((x, y)), z))).to(Mul((x, Mul((y, z)))))

    yield birewrite(Mul((z, Add((x, y))))).to(Add((Mul((x, z)), Mul((y, z)))))


@egraph.register
def reduce_rules(
    x: Tensor,
    y: Tensor,
    i: i64,
    j: i64,
    axis: IntTuple,
    op: ScalarOp,
):
    any_reduce_op = partial(Reduce, op)

    yield from reduce_op_axis_variants(any_reduce_op, x, axis, i, j)

    yield rewrite(Shape(any_reduce_op(axis=(i,))(x))).to(Shape(x).pop(i))

    # Introduce shape[i] needed for removing useless reductions
    yield rule(eq(y).to(any_reduce_op(axis=(i,))(x))).then(Shape(x)[i])

    # Remove useless reductions
    yield rewrite(any_reduce_op(axis=(i,))(x)).to(
        Squeeze(axis=(i,))(x), eq(Shape(x)[i]).to(Tensor.constant(1))
    )


@egraph.register
def sum_rules(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    i: i64,
):
    # Introduce shape[i] needed for factoring out multiplication/division out of sum
    yield rule(eq(z).to(Sum(axis=(i,))(Mul((x, y))))).then(Shape(y)[i])

    # Factor multiplication/division out of sum
    for elemwise_op in (Mul, Div):
        yield rewrite(Sum(axis=(i,))(elemwise_op((x, y)))).to(
            elemwise_op(
                (
                    Sum(axis=(i,))(x),
                    Squeeze(axis=(i,))(y),
                )
            ),
            eq(Shape(y)[i]).to(Tensor.constant(1)),
        )
