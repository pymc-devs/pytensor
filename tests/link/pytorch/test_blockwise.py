import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.blockwise import Blockwise


torch = pytest.importorskip("torch")
basic = pytest.importorskip("pytensor.link.pytorch.dispatch.basic")


class BatchedTestOp(Op):
    gufunc_signature = "(m,n),(n,p)->(m,p)"

    def __init__(self, final_shape):
        super().__init__()
        self.final_shape = final_shape
        self.call_shapes = []

    def make_node(self, *args):
        return Apply(self, list(args), [pt.matrix("_", shape=self.final_shape)])

    def perform(self, *_):
        raise RuntimeError("In perform")


@basic.pytorch_funcify.register(BatchedTestOp)
def evaluate_test_op(op, **_):
    def func(a, b):
        op.call_shapes.extend(map(torch.Tensor.size, [a, b]))
        return a @ b

    return func


def test_blockwise_broadcast():
    _x = np.random.rand(5, 1, 2, 3)
    _y = np.random.rand(3, 3, 2)

    x = pt.tensor4("x", shape=(5, 1, 2, 3))
    y = pt.tensor3("y", shape=(3, 3, 2))
    op = BatchedTestOp((2, 2))
    z = Blockwise(op)(x, y)

    f = pytensor.function([x, y], z, mode="PYTORCH")
    res = f(_x, _y)
    assert tuple(res.shape) == (5, 3, 2, 2)
    np.testing.assert_allclose(res, _x @ _y)
    assert op.call_shapes == [(2, 3), (3, 2)]
