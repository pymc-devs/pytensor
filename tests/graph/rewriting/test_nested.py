from pytensor.compile.builders import OpFromGraph
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import NodeRewriter
from pytensor.graph.rewriting.nested import WalkingNestedGraphRewriter
from pytensor.scan import scan
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.math import sum
from pytensor.tensor.type import matrix, scalar, vector


class TestWalkingNestedGraphRewriter:
    def apply_rewrites(self, fgraph):
        cnt = 0

        class MyRwriter(NodeRewriter):
            def transform(self, fgraph, node):
                nonlocal cnt
                if isinstance(node.op, Elemwise):
                    cnt += 1
                return node.outputs

        node_rewriter = MyRwriter()
        rewriter = WalkingNestedGraphRewriter(node_rewriter)
        rewriter.apply(fgraph)
        return cnt

    def test_rewrite_in_scan(self):
        def scan_step(x_0):
            x = x_0 + 1
            return x

        x_0 = vector("x_0")
        result, _ = scan(
            scan_step,
            outputs_info=None,
            sequences=x_0,
        )
        x = sum(result) + 1
        graph = FunctionGraph([x_0], [x], clone=False)

        rewrites_cnt = self.apply_rewrites(graph)

        # one replacemnt in the scan inner grap and one in outer graph
        assert rewrites_cnt == 2

    def test_rewrite_in_nested_scan(self):
        def inner_scan_step(x_0):
            x = x_0 + 1
            return x

        def outer_scan_step(x_0):
            x, _ = scan(
                fn=inner_scan_step,
                sequences=x_0,
                outputs_info=None,
            )
            x = x + 1
            return x

        x_0 = matrix("x_0")
        result, _ = scan(
            fn=outer_scan_step,
            sequences=x_0,
            outputs_info=None,
        )

        graph = FunctionGraph([x_0], [result], clone=False)
        rewrites_cnt = self.apply_rewrites(graph)
        # one replacemnt in the inner scan and one in outer scan
        assert rewrites_cnt == 2

    def test_rewrite_op_from_graph(self):
        x, y, z = scalar("x"), scalar("y"), scalar("z")
        e = x + y * z
        op = OpFromGraph([x, y, z], [e])
        e2 = op(x, y, z) + op(z, y, x)
        graph = FunctionGraph([x, y, z], [e2], clone=False)

        rewrites_cnt = self.apply_rewrites(graph)
        # two rewrites in each OpFromGraph inner graphs and one in outer graph
        assert rewrites_cnt == 5

    def test_rewrite_nested_op_from_graph(self):
        x, y, z = scalar("x"), scalar("y"), scalar("z")
        e = x + y
        op = OpFromGraph([x, y], [e])
        e2 = op(x, y) * op(x, y)
        op2 = OpFromGraph([x, y], [e2])
        e3 = op2(x, y) + z
        graph = FunctionGraph([x, y, z], [e3], clone=False)

        rewrites_cnt = self.apply_rewrites(graph)
        # two rewrites in inner most OpFromGraph, one in second OpFromGraph, and one in outer graph
        assert rewrites_cnt == 4
