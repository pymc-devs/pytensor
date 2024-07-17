import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph


def test_alloc_diag_inlined():
    x = pt.tensor("x", shape=(None,))

    # pt.diag calls AllocDiag, which inherits from OpFromGrpah
    z = pt.diag(x)
    f = pytensor.function([x], z)

    nodes = f.maker.fgraph.apply_nodes

    assert not any(isinstance(node.op, OpFromGraph) for node in nodes)
