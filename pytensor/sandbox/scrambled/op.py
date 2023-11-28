from functools import partial

from pytensor.sandbox.scrambled.basic import (
    Int,
    IntTuple,
    ScalarOp,
    Tensor,
    TensorTuple,
    UnaryInOp,
    VariadicInOp,
    egraph,
)


@egraph.function(cost=20)
def Reduce(scalar_op: ScalarOp, axis: IntTuple) -> UnaryInOp:
    ...


@egraph.function(cost=10)
def Elemwise(scalar_op: ScalarOp) -> VariadicInOp:
    ...


@egraph.function(cost=1)
def Squeeze(axis: IntTuple) -> UnaryInOp:
    ...


@egraph.function(cost=1)
def ExpandDims(axis: IntTuple) -> UnaryInOp:
    ...


@egraph.function
def BroadcastShapes(s1: TensorTuple, s2: TensorTuple) -> TensorTuple:
    ...


@egraph.function
def BroadcastShapesOfTensors(tensors: TensorTuple) -> TensorTuple:
    ...


@egraph.function
def Shape_i(x: Tensor, i: Int) -> Tensor:
    ...


@egraph.function(cost=100)
def Shape(x: Tensor) -> TensorTuple:
    ...


@egraph.function(cost=5)
def Alloc(x: Tensor, shape: TensorTuple) -> Tensor:
    ...


ScalarAdd = egraph.constant("Add", ScalarOp)
ScalarSub = egraph.constant("Sub", ScalarOp)
ScalarMul = egraph.constant("Mul", ScalarOp)
ScalarDiv = egraph.constant("Div", ScalarOp)

Add = Elemwise(ScalarAdd)
Sub = Elemwise(ScalarSub)
Mul = Elemwise(ScalarMul)
Div = Elemwise(ScalarDiv)

Sum = partial(Reduce, ScalarAdd)
