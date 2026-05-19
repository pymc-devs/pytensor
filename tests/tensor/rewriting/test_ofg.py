import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import config
from pytensor.compile.builders import OpFromGraph
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import dfs_rewriter
from pytensor.tensor.rewriting.ofg import inline_ofg_expansion


@pytest.mark.skipif(
    config.mode == "FAST_COMPILE",
    reason="Rewrite is not applied in FAST_COMPILE mode",
)
def test_alloc_diag_inlined():
    x = pt.tensor("x", shape=(None,))

    z = pt.diag(x)
    assert isinstance(z.owner.op, OpFromGraph)

    f = pytensor.function([x], z)
    nodes = f.maker.fgraph.apply_nodes

    assert not any(isinstance(node.op, OpFromGraph) for node in nodes)


def test_expansion_no_cloning():
    x = pt.scalar("x")
    y = pt.exp(x)

    inner_y = y.type()
    ofg = OpFromGraph([inner_y], [pt.cos(inner_y)], inline=True)
    z = ofg(y)

    fg = FunctionGraph(outputs=[y, z])
    assert len(fg.toposort()) == 2

    dfs_rewriter(inline_ofg_expansion).rewrite(fg)
    assert len(fg.toposort()) == 2, len(fg.toposort())
