import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
import pytensor.tensor.math as ptm
from pytensor.configdefaults import config
from pytensor.scalar.basic import ScalarOp, get_scalar_type
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.special import SoftmaxGrad, log_softmax, softmax
from pytensor.tensor.type import matrix, tensor, tensor3, vector
from tests.link.pytorch.test_basic import compare_pytorch_and_py


torch = pytest.importorskip("torch")


def test_pytorch_Dimshuffle():
    a_pt = matrix("a")

    x = a_pt.T

    compare_pytorch_and_py(
        [a_pt], [x], [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)]
    )

    x = a_pt.dimshuffle([0, 1, "x"])

    compare_pytorch_and_py(
        [a_pt], [x], [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)]
    )

    a_pt = tensor(dtype=config.floatX, shape=(None, 1))
    x = a_pt.dimshuffle((0,))

    compare_pytorch_and_py(
        [a_pt], [x], [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)]
    )


def test_multiple_input_output():
    x = vector("x")
    y = vector("y")
    out = pt.mul(x, y)

    compare_pytorch_and_py([x, y], [out], [[1.5], [2.5]])

    x = vector("x")
    y = vector("y")
    div = pt.int_div(x, y)
    pt_sum = pt.add(y, x)

    compare_pytorch_and_py([x, y], [div, pt_sum], [[1.5], [2.5]])


def test_pytorch_elemwise():
    x = pt.vector("x")
    out = pt.log(1 - x)

    compare_pytorch_and_py([x], [out], [[0.9, 0.9]])


@pytest.mark.parametrize("fn", [ptm.sum, ptm.prod, ptm.max, ptm.min])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, -1)])
def test_pytorch_careduce(fn, axis):
    a_pt = tensor3("a")
    test_value = np.array(
        [
            [
                [1, 1, 1, 1],
                [2, 2, 2, 2],
            ],
            [
                [3, 3, 3, 3],
                [
                    4,
                    4,
                    4,
                    4,
                ],
            ],
        ]
    ).astype(config.floatX)

    x = fn(a_pt, axis=axis)

    compare_pytorch_and_py([a_pt], [x], [test_value])


@pytest.mark.parametrize("fn", [ptm.any, ptm.all])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
def test_pytorch_any_all(fn, axis):
    a_pt = matrix("a")
    test_value = np.array([[True, False, True], [False, True, True]])

    x = fn(a_pt, axis=axis)

    compare_pytorch_and_py([a_pt], [x], [test_value])


@pytest.mark.parametrize("dtype", ["float64", "int64"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax(axis, dtype):
    x = matrix("x", dtype=dtype)
    out = softmax(x, axis=axis)
    test_input = np.arange(6, dtype=config.floatX).reshape(2, 3)

    if dtype == "int64":
        with pytest.raises(
            NotImplementedError,
            match="Pytorch Softmax is not currently implemented for non-float types\\.",
        ):
            compare_pytorch_and_py([x], [out], [test_input])
    else:
        compare_pytorch_and_py([x], [out], [test_input])


@pytest.mark.parametrize("dtype", ["float64", "int64"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_logsoftmax(axis, dtype):
    x = matrix("x", dtype=dtype)
    out = log_softmax(x, axis=axis)
    test_input = np.arange(6, dtype=config.floatX).reshape(2, 3)

    if dtype == "int64":
        with pytest.raises(
            NotImplementedError,
            match="Pytorch LogSoftmax is not currently implemented for non-float types\\.",
        ):
            compare_pytorch_and_py([x], [out], [test_input])
    else:
        compare_pytorch_and_py([x], [out], [test_input])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_softmax_grad(axis):
    dy = matrix("dy")
    dy_value = np.array([[1, 1, 1], [0, 0, 0]], dtype=config.floatX)
    sm = matrix("sm")
    sm_value = np.arange(6, dtype=config.floatX).reshape(2, 3)
    out = SoftmaxGrad(axis=axis)(dy, sm)
    compare_pytorch_and_py([dy, sm], [out], [dy_value, sm_value])


def test_cast():
    x = matrix("x", dtype="float32")
    out = pt.cast(x, "int32")
    _, [res] = compare_pytorch_and_py(
        [x], [out], [np.arange(6, dtype="float32").reshape(2, 3)]
    )
    assert res.dtype == np.int32


@pytest.mark.parametrize(
    "x_val, min_val, max_val",
    [
        (np.array([5.0], dtype=config.floatX), 0.0, 10.0),
        (np.array([-5.0], dtype=config.floatX), 0.0, 10.0),
    ],
)
def test_clip(x_val, min_val, max_val):
    x = pt.tensor("x", shape=x_val.shape, dtype=config.floatX)
    out = pt.clip(x, min_val, max_val)
    compare_pytorch_and_py([x], [out], [x_val])


def test_vmap_elemwise():
    from pytensor.link.pytorch.dispatch.basic import pytorch_funcify

    class TestOp(ScalarOp):
        def __init__(self):
            super().__init__(
                output_types_preference=lambda *_: [get_scalar_type("float32")]
            )
            self.call_shapes = []
            self.nin = 1

        def perform(self, *_):
            raise RuntimeError("In perform")

    @pytorch_funcify.register(TestOp)
    def relu(op, node, **kwargs):
        def relu(row):
            op.call_shapes.append(row.size())
            return torch.max(torch.zeros_like(row), row)

        return relu

    x = matrix("x", shape=(2, 3))
    op = TestOp()
    f = pytensor.function([x], Elemwise(op)(x), mode="PYTORCH")
    vals = torch.zeros(2, 3).normal_()
    np.testing.assert_allclose(f(vals), torch.relu(vals))
    assert op.call_shapes == [torch.Size([])], op.call_shapes
