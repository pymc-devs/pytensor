import pytest

from pytensor.graph.replace import vectorize_node
from pytensor.tensor import tensor
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.nlinalg import MatrixInverse


torch = pytest.importorskip("torch")


def test_vectorize_blockwise():
    mat = tensor(shape=(None, None))
    tns = tensor(shape=(None, None, None))

    # Something that falls back to Blockwise
    node = MatrixInverse()(mat).owner
    vect_node = vectorize_node(node, tns)
    assert isinstance(vect_node.op, Blockwise) and isinstance(
        vect_node.op.core_op, MatrixInverse
    )
    assert vect_node.op.signature == ("(m,m)->(m,m)")
    assert vect_node.inputs[0] is tns

    # Useless blockwise
    tns4 = tensor(shape=(5, None, None, None))
    new_vect_node = vectorize_node(vect_node, tns4)
    assert new_vect_node.op is vect_node.op
    assert isinstance(new_vect_node.op, Blockwise) and isinstance(
        new_vect_node.op.core_op, MatrixInverse
    )
    assert new_vect_node.inputs[0] is tns4
