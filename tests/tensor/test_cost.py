import sys

import pytensor.scalar as ps
import pytensor.tensor as pt
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.cost import Bound, Cost, CostExpr, _polynomial_le, provably_cheaper
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.printing import pprint
from pytensor.tensor.cost import ShapeSymbols, dims_of, graph_cost, node_cost
from pytensor.tensor.elemwise import Elemwise
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


class _OpWithoutRule(Op):
    def make_node(self, x):
        x = pt.as_tensor(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError


def _node_cost(out):
    return node_cost(out.owner, _shape_feature(out), ShapeSymbols())


def _exact(flops, bytes_):
    return Cost(Bound.exact(flops), Bound.exact(bytes_))


class TestElemwiseAndDot:
    def test_dot_counts_two_flops_per_multiply_add(self):
        out = pt.dot(
            pt.matrix("x", shape=(5, 3), dtype="float64"),
            pt.matrix("y", shape=(3, 4), dtype="float64"),
        )
        expected_bytes = 8 * (5 * 3 + 3 * 4 + 5 * 4)
        assert _node_cost(out) == _exact(
            CostExpr.from_dim_product([2 * 5 * 4 * 3]),
            CostExpr.from_dim_product([expected_bytes]),
        )

    def test_symbolic_dot(self):
        x, y = pt.matrix("x"), pt.matrix("y")
        out = pt.dot(x, y)
        shape_feature, symbols = _shape_feature(out), ShapeSymbols()
        m, p = dims_of(x, shape_feature, symbols)
        _, n = dims_of(y, shape_feature, symbols)
        cost = node_cost(out.owner, shape_feature, symbols)
        assert cost.flops == Bound.exact(CostExpr.from_dim_product([2, m, n, p]))

    def test_elemwise_counts_its_broadcast_output(self):
        out = pt.tensor("x", shape=(2, 1, 7), dtype="float64") + pt.tensor(
            "y", shape=(1, 3, 7), dtype="float64"
        )
        expected_bytes = 8 * (2 * 7 + 3 * 7 + 2 * 3 * 7)
        assert _node_cost(out) == _exact(
            CostExpr.from_dim_product([2 * 3 * 7]),
            CostExpr.from_dim_product([expected_bytes]),
        )

    def test_composite_counts_each_inner_float_operation(self):
        a = ps.float64("a")
        composite = ps.Composite([a], [ps.exp(a) + a])
        out = Elemwise(composite)(pt.vector("x", shape=(10,)))
        assert _node_cost(out).flops == Bound.exact(CostExpr.from_dim_product([20]))

        # the integer add is free; the cast to float and the multiply count
        i = ps.int64("i")
        mixed = ps.Composite([a, i], [a * ps.cast(i + i, "float64")])
        out = Elemwise(mixed)(pt.vector("x", shape=(10,)), pt.lvector("j", shape=(10,)))
        assert _node_cost(out).flops == Bound.exact(CostExpr.from_dim_product([20]))

    def test_variadic_add_counts_one_flop_per_extra_input(self):
        x = pt.vector("x", shape=(10,), dtype="float64")
        out = pt.add(x, x, x, x)
        assert _node_cost(out).flops == Bound.exact(CostExpr.from_dim_product([30]))

    def test_inplace_output_counts_like_any_other(self):
        x = pt.vector("x", shape=(10,), dtype="float64")
        inplace = Elemwise(ps.exp, inplace_pattern={0: 0})(x)
        assert inplace.owner.op.destroy_map
        assert _node_cost(inplace) == _node_cost(pt.exp(x))

    def test_flops_need_a_float_input_or_output(self):
        integer_sum = pt.lvector("i", shape=(10,)) + pt.lvector("j", shape=(10,))
        float_comparison = pt.vector("x", shape=(10,), dtype="float64") > 0
        assert _node_cost(integer_sum).flops == Bound.exact(CostExpr.zero())
        assert _node_cost(float_comparison).flops == Bound.exact(
            CostExpr.from_dim_product([10])
        )


class TestGraphCost:
    def test_shared_node_counted_once(self):
        x = pt.matrix("x", shape=(4, 3), dtype="float64")
        y = pt.matrix("y", shape=(3, 5), dtype="float64")
        shared_dot = pt.dot(x, y)
        shared = shared_dot + pt.exp(shared_dot)
        duplicated = pt.dot(x, y) + pt.exp(pt.dot(x, y))

        dot_flops, elemwise_flops = 2 * 4 * 5 * 3, 4 * 5
        shared_cost = graph_cost([shared])
        assert shared_cost.flops == Bound.exact(
            CostExpr.from_dim_product([dot_flops + 2 * elemwise_flops])
        )
        assert graph_cost([duplicated]).flops == Bound.exact(
            CostExpr.from_dim_product([2 * dot_flops + 2 * elemwise_flops])
        )
        assert provably_cheaper(shared_cost, graph_cost([duplicated]))

    def test_unknown_node_leaves_the_upper_bound_open(self):
        x = pt.matrix("x", shape=(4, 3), dtype="float64")
        out = pt.exp(_OpWithoutRule()(x))
        assert graph_cost([out]).flops == Bound(CostExpr.from_dim_product([12]), None)

    def test_leaves_the_graphs_features_as_found(self):
        x = pt.matrix("x", shape=(4, 3), dtype="float64")
        expected = graph_cost([pt.exp(x)])

        bare = FunctionGraph(outputs=[pt.exp(x)], clone=False)
        assert graph_cost(bare) == expected
        assert not hasattr(bare, "shape_feature")

        shape_feature = ShapeFeature()
        with_feature = FunctionGraph(
            outputs=[pt.exp(x)], features=[shape_feature], clone=False
        )
        assert graph_cost(with_feature) == expected
        assert with_feature.shape_feature is shape_feature

    def test_shape_symbols_format_as_their_expressions(self):
        x = pt.matrix("x", dtype="float64")
        flops = graph_cost([pt.exp(x)]).flops.hi
        assert flops.format(label=pprint) == "Shape_i{0}(x)*Shape_i{1}(x)"

    def test_minted_symbols_allow_empty_dims(self):
        # m*k <= 2*m*k*p holds for p >= 1 but fails at p = 0
        x, y = pt.matrix("x"), pt.matrix("y")
        symbols = ShapeSymbols()
        exp_flops = graph_cost([pt.exp(x)], symbols=symbols).flops.hi
        dot_flops = graph_cost([pt.dot(x, y)], symbols=symbols).flops.hi
        assert _polynomial_le(exp_flops, dot_flops)
        assert not _polynomial_le(exp_flops, dot_flops, zero_able=symbols.symbols)

    def test_symbolic_graphs_compare_through_shared_symbols(self):
        x = pt.matrix("x")
        symbols = ShapeSymbols()
        once = graph_cost([pt.exp(x)], symbols=symbols)
        twice = graph_cost([pt.exp(pt.exp(x))], symbols=symbols)
        assert provably_cheaper(once, twice)
