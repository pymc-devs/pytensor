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


def test_inline_value_dependent_output_shape():
    """Two inline OpFromGraphs with the same inner structure but different baked
    output shapes must inline to their own shapes.

    Regression for #2202.
    """

    x = pt.vector("x", shape=(6,))
    shp_a = pt.stack([2, 3])
    shp_b = pt.stack([3, 2])
    out_a = OpFromGraph([x, shp_a], [pt.reshape(x, shp_a)], inline=True)(x, shp_a)
    out_b = OpFromGraph([x, shp_b], [pt.reshape(x, shp_b)], inline=True)(x, shp_b)
    assert out_a.type.shape == (2, 3)
    assert out_b.type.shape == (3, 2)

    fg = FunctionGraph([x], [out_a, out_b])
    dfs_rewriter(inline_ofg_expansion).rewrite(fg)

    assert not any(isinstance(n.op, OpFromGraph) for n in fg.apply_nodes)
    assert fg.outputs[0].type.shape == (2, 3)
    assert fg.outputs[1].type.shape == (3, 2)
