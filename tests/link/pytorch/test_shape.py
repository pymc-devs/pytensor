import numpy as np

import pytensor.tensor as pt
from pytensor.configdefaults import config
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.shape import Shape, Shape_i, Unbroadcast, reshape
from pytensor.tensor.type import iscalar, vector
from tests.link.pytorch.test_basic import compare_pytorch_and_py


def test_pytorch_shape_ops():
    x_np = np.zeros((20, 3))
    x = Shape()(pt.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_pytorch_and_py(x_fg, [], must_be_device_array=False)

    x = Shape_i(1)(pt.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_pytorch_and_py(x_fg, [], must_be_device_array=False)


def test_pytorch_specify_shape():
    in_pt = pt.matrix("in")
    x = pt.specify_shape(in_pt, (4, None))
    x_fg = FunctionGraph([in_pt], [x])
    compare_pytorch_and_py(x_fg, [np.ones((4, 5)).astype(config.floatX)])

    # When used to assert two arrays have similar shapes
    in_pt = pt.matrix("in")
    shape_pt = pt.matrix("shape")
    x = pt.specify_shape(in_pt, shape_pt.shape)
    x_fg = FunctionGraph([in_pt, shape_pt], [x])
    compare_pytorch_and_py(
        x_fg,
        [np.ones((4, 5)).astype(config.floatX), np.ones((4, 5)).astype(config.floatX)],
    )


def test_pytorch_Reshape_constant():
    a = vector("a")
    x = reshape(a, (2, 2))
    x_fg = FunctionGraph([a], [x])
    compare_pytorch_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])


def test_pytorch_Reshape_dynamic():
    a = vector("a")
    shape_pt = iscalar("b")
    x = reshape(a, (shape_pt, shape_pt))
    x_fg = FunctionGraph([a, shape_pt], [x])
    compare_pytorch_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX), 2])


def test_pytorch_unbroadcast():
    x_np = np.zeros((20, 1, 1))
    x = Unbroadcast(0, 2)(pt.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_pytorch_and_py(x_fg, [])
