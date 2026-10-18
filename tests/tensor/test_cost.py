import sys

import pytensor.tensor as pt
from pytensor.graph.basic import Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.cost import ShapeSymbols, dims_of
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.shape import Shape_i


def _shape_feature(*outputs):
    fg = FunctionGraph(outputs=list(outputs), clone=False)
    shape_feature = ShapeFeature()
    fg.attach_feature(shape_feature)
    return shape_feature


class TestDimsOf:
    def test_static_dims_are_ints(self):
        x = pt.matrix("x", shape=(2, 3))
        assert dims_of(x, _shape_feature(x), ShapeSymbols()) == (2, 3)

    def test_equal_dims_share_a_symbol(self):
        x = pt.matrix("x")
        y = pt.exp(x)
        shape_feature, symbols = _shape_feature(y), ShapeSymbols()
        x_dims = dims_of(x, shape_feature, symbols)
        y_dims = dims_of(y, shape_feature, symbols)
        assert all(a is b for a, b in zip(x_dims, y_dims, strict=True))

    def test_rebuilt_expressions_share_a_symbol(self):
        x = pt.matrix("x")
        symbols = ShapeSymbols()
        n = symbols.dim(Shape_i(0)(x))
        assert symbols.dim(Shape_i(0)(x)) is n
        assert symbols.dim(Shape_i(1)(x)) is not n
        # A cast need not preserve the value, so it is a different symbol.
        assert symbols.dim(pt.cast(Shape_i(0)(x), "int32")) is not n

    def test_dim_deeper_than_the_recursion_limit(self):
        v = pt.vector("v", dtype="float64")
        out = v
        for _ in range(sys.getrecursionlimit() + 100):
            out = pt.concatenate([out, v])
        assert isinstance(
            dims_of(out, _shape_feature(out), ShapeSymbols())[0], Variable
        )

    def test_constant_dim_is_an_int(self):
        assert ShapeSymbols().dim(pt.cast(pt.as_tensor(5), "int32")) == 5
