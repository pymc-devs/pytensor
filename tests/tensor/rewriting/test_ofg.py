import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph


def test_inherited_ofg_class_inlined():
    x = pt.tensor("x", shape=(None,))

    # pt.diag calls AllocDiag, which inherits from OpFromGrpah
    z = pt.diag(x)

    f = pytensor.function([x], z)
    pytensor.dprint(f)

    nodes = f.maker.fgraph.apply_nodes

    assert not any(isinstance(node.op, OpFromGraph) for node in nodes)


def test_several_ofg_inlined():
    x = pt.tensor("x", shape=(None,))
    y = pt.diag(x)

    # pt.linalg.kron also inherits from OpFromGraph
    z = pt.linalg.kron(y, pt.eye(2))

    f = pytensor.function([x], z)
    pytensor.dprint(f)

    nodes = f.maker.fgraph.apply_nodes

    assert not any(isinstance(node.op, OpFromGraph) for node in nodes)
