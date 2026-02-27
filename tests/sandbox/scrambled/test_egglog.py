import numpy as np
from egglog import convert, eq

import pytensor.tensor as pt
from pytensor.graph.basic import equal_computations
from pytensor.sandbox.scrambled.basic import (
    Int,
    IntTuple,
    ScalarOp,
    Tensor,
    TensorTuple,
    egg_rewrite,
    egraph,
)
from pytensor.sandbox.scrambled.op import (
    Alloc,
    BroadcastShapes,
    Div,
    Elemwise,
    ExpandDims,
    Mul,
    Shape,
    Squeeze,
    Sub,
    Sum,
)


def assert_equivalent(source_expr, target_expr, epochs=20):
    with egraph:
        egraph.register(source_expr)
        egraph.run(epochs)
        egraph.check(eq(source_expr).to(target_expr))


def test_int_tuple():
    assert_equivalent(IntTuple(5).insert(0, 99).pop(0), IntTuple(5))
    assert_equivalent(IntTuple(5).insert(0, 99).pop(1), IntTuple(99))
    assert_equivalent(IntTuple(5).insert(1, 99).pop(1), IntTuple(5))
    assert_equivalent(IntTuple(5).insert(1, 99).pop(0), IntTuple(99))
    assert_equivalent(IntTuple(5).pop(0), IntTuple.empty())
    assert_equivalent(IntTuple(5).insert(0, 5).pop(0).pop(0), IntTuple.empty())
    assert_equivalent(
        IntTuple(5).insert(0, 5).pop(0).pop(0).insert(0, 99), IntTuple(99)
    )

    assert_equivalent(IntTuple.empty().length(), Int(0))
    assert_equivalent(IntTuple(0).length(), Int(1))
    assert_equivalent((IntTuple(0) + IntTuple.empty()).length(), Int(1))
    assert_equivalent(convert((1, 2, 3), IntTuple).length(), Int(3))

    assert_equivalent(IntTuple.from_range(0, 0), IntTuple.empty())
    assert_equivalent(IntTuple.from_range(0, 3), convert((0, 1, 2), IntTuple))
    assert_equivalent(IntTuple.from_range(1, 3), convert((1, 2), IntTuple))


def test_broadcast_shapes():
    assert_equivalent(TensorTuple.empty() ^ TensorTuple(5), TensorTuple(5))
    assert_equivalent(
        BroadcastShapes((1, 4, 1, 1), BroadcastShapes((4, 1, 1, 4), (4, 1))),
        (4, 4, 4, 4),
        epochs=30,
    )


def test_alloc():
    x = Tensor("x", (5, 5))
    y = Alloc(x, (5, 5))
    assert_equivalent(y, x)

    x = Tensor("x", (5, 1))
    y = Alloc(Alloc(x, (5, 5)), (5, 5))
    assert_equivalent(y, Alloc(x, (5, 5)))

    x = Tensor("x")
    y = ExpandDims(axis=(0, 1))(Alloc(x, (5,)))
    assert_equivalent(y, Alloc(x, (1, 1, 5)))

    x = Tensor("x")
    y = Squeeze(axis=(0, 1))(Alloc(x, (1, 1, 5)))
    assert_equivalent(y, Alloc(x, 5))

    x = Tensor("x")
    y = Alloc(Alloc(x, (5,)), (3, 5))
    assert_equivalent(y, Alloc(x, (3, 5)))


def test_elemwise_of_alloc():
    x = Tensor("x")
    y = Tensor("y", shape=(2, 5))
    source = Div((Alloc(x, (2, 5)), y))
    target = Div((ExpandDims((0, 1))(x), y))
    assert_equivalent(source, target)

    x = Tensor("x")
    y = Tensor("y", shape=(2, 5))
    source = Div((y, Alloc(x, (2, 5))))
    target = Div((y, ExpandDims((0, 1))(x)))
    assert_equivalent(source, target)

    Tri = egraph.constant("Tri", ScalarOp)
    x = Tensor("x")
    y = Tensor("y", shape=(2, 5))
    source = Elemwise(Tri)((y, Alloc(x, (2, 5)), y))
    target = Elemwise(Tri)((y, ExpandDims((0, 1))(x), y))
    assert_equivalent(source, target)

    x = Tensor("x", shape=(2,))
    y = Tensor("y", shape=(2, 5))
    x_exp = ExpandDims(1)(x)  # (2, 1)
    source = Div((Alloc(x_exp, (2, 5)), y))
    target = Alloc(Div((ExpandDims(1)(x), y)), (2, 5))
    assert_equivalent(source, target)

    x = Tensor("x", shape=(2,))
    y = Tensor("y", shape=(2, 1))
    x_exp = ExpandDims(1)(x)  # (2, 1)
    source = Div((Alloc(x_exp, (2, 5)), y))
    target = Alloc(Div((ExpandDims(1)(x), y)), (2, 5))
    assert_equivalent(source, target)


def test_arithmetic():
    x = Tensor("x", (5,))
    y = Tensor("y", (5,))
    assert_equivalent(x - x + y, y)

    x = Tensor("x", (1,))
    y = Tensor("y", (5,))
    assert_equivalent(y + (x - x), y)

    x = Tensor("x", (3, 4, 1))
    y = Tensor("y", (5,))
    assert_equivalent(x - x + ExpandDims((0, 1))(y), Alloc(y, (3, 4, 5)))

    x = Tensor("x", (2, 1))
    y = Tensor("y")
    five = Tensor.constant(5)
    ones = Tensor.constant(1, shape=(5,))
    source = Mul(
        (
            Sub(
                (
                    Mul((ExpandDims(axis=(0, 1))(five), x)),
                    Mul((ExpandDims(axis=(0, 1))(five), x)),
                )
            ),
            ExpandDims(axis=0)(Mul((ones, ExpandDims(axis=0)(y)))),
        )
    )
    target = Alloc(Tensor.constant(0), (2, 5))
    assert_equivalent(source, target)


def test_reduce_shape():
    x = Tensor("x", (1, 3))
    y = Tensor("y", (5, 1))
    assert_equivalent(Shape(Sum(axis=0)(x * y)), TensorTuple(3))
    assert_equivalent(Shape(Sum(axis=1)(x * y)), TensorTuple(5))
    assert_equivalent(Shape(Sum(axis=(0, 1))(x * y)), TensorTuple.empty())


def test_reduce_useless():
    x = Tensor("x", (5,))
    source = Sum(axis=(0, 2))(ExpandDims(axis=(0, 2))(x))
    assert_equivalent(source, x)

    source = Sum(axis=0)(Sum(axis=0)(ExpandDims(axis=1)(ExpandDims(axis=(0))(x))))
    assert_equivalent(source, x)

    x = Tensor("x", (5, 3))
    source = Sum(IntTuple.empty())(x)
    assert_equivalent(source, x)

    x = Tensor("x", (5, 3))
    source = Sum(axis=(1) + IntTuple.empty())(x)
    target = Sum(axis=1)(x)
    # This would create an ever-expanding graph
    assert_equivalent(source, target, epochs=1000)


def test_reverse_distributivity():
    x = Tensor("x", (7, 5, 3))
    y = Tensor("y", (5,))

    source = Sum(axis=(0, 2))(Mul((x, ExpandDims(axis=(0, 2))(y))))
    target = Mul((Sum(axis=(0, 2))(x), y))

    assert_equivalent(source, target)


def test_egg_rewrite():
    x = pt.scalar("x", dtype="int64")
    y = pt.matrix("y", dtype="int64")
    assert egg_rewrite(x) == (x,)
    assert egg_rewrite(x, y) == (x, y)
    assert equal_computations(egg_rewrite(x + y), (x + y,))

    z = pt.sum(y * (x * np.asarray(1, dtype="int64")), axis=0)
    res = egg_rewrite(z)
    try:
        assert equal_computations(res, (x * y.sum(0),))
    except AssertionError:
        assert equal_computations(res, (y.sum(0) * x,))

    z = pt.sum(y * (x * np.asarray(0, dtype="int64")), axis=0)
    assert equal_computations(
        egg_rewrite(z),
        (pt.alloc(np.array(0, dtype=int), y.shape[1]),),
    )
