from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import graph_rewriter
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.type import vectors


def test_rewrite_graph():
    x, y = vectors("xy")

    @graph_rewriter
    def custom_rewrite(fgraph):
        fgraph.replace(x, y, import_missing=True)

    x_rewritten = rewrite_graph(x, custom_rewrite=custom_rewrite)

    assert x_rewritten is y

    x_rewritten = rewrite_graph(
        FunctionGraph(outputs=[x], clone=False), custom_rewrite=custom_rewrite
    )

    assert x_rewritten.outputs[0] is y
