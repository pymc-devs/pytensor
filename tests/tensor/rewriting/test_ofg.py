import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph


def test_OpFromGraph_inlined():
    x = pt.tensor("x", shape=(None,))
    z = x**2
    fx = OpFromGraph([x], [z], inline=False)(x)
    f2 = pytensor.function([x], fx)

    nodes = f2.maker.fgraph.apply_nodes

    assert any(isinstance(node.op, OpFromGraph) for node in nodes)
    assert all(node.op.is_inline for node in nodes if isinstance(node.op, OpFromGraph))


def test_inherited_ofg_class_inlined():
    x = pt.tensor("x", shape=(None,))

    # pt.diag calls AllocDiag, which inherits from OpFromGrpah
    z = pt.diag(x)

    f = pytensor.function([x], z)
    pytensor.dprint(f)

    nodes = f.maker.fgraph.apply_nodes

    assert any(isinstance(node.op, OpFromGraph) for node in nodes)
    assert all(node.op.is_inline for node in nodes if isinstance(node.op, OpFromGraph))


def test_several_ofg_inlined():
    x = pt.tensor("x", shape=(None,))
    y = pt.diag(x)

    # pt.linalg.kron also inherits from OpFromGraph
    z = pt.linalg.kron(y, pt.eye(2))

    f = pytensor.function([x], z)
    pytensor.dprint(f)

    nodes = f.maker.fgraph.apply_nodes

    assert any(isinstance(node.op, OpFromGraph) for node in nodes)
    assert all(node.op.is_inline for node in nodes if isinstance(node.op, OpFromGraph))


def test_ofg_not_inlined_in_JAX_mode():
    pytest.importorskip("jax")

    x = pt.tensor("x", shape=(None,))
    y = pt.diag(x)

    f = pytensor.function([x], y, mode="JAX")
    nodes = f.maker.fgraph.apply_nodes
    assert any(isinstance(node.op, OpFromGraph) for node in nodes)
    assert not any(
        node.op.is_inline for node in nodes if isinstance(node.op, OpFromGraph)
    )
