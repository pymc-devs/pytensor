import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.configdefaults import config
from pytensor.tensor.shape import Shape, Shape_i, reshape
from pytensor.tensor.type import iscalar, vector
from tests.link.mlx.test_basic import compare_mlx_and_py


def test_mlx_shape_ops():
    x_np = np.zeros((20, 3))
    x = Shape()(pt.as_tensor_variable(x_np))

    compare_mlx_and_py([], [x], [], must_be_device_array=False)

    x = Shape_i(1)(pt.as_tensor_variable(x_np))

    compare_mlx_and_py([], [x], [], must_be_device_array=False)


def test_mlx_specify_shape():
    in_pt = pt.matrix("in")
    x = pt.specify_shape(in_pt, (4, None))
    compare_mlx_and_py([in_pt], [x], [np.ones((4, 5)).astype(config.floatX)])

    # When used to assert two arrays have similar shapes
    in_pt = pt.matrix("in")
    shape_pt = pt.matrix("shape")
    x = pt.specify_shape(in_pt, shape_pt.shape)

    compare_mlx_and_py(
        [in_pt, shape_pt],
        [x],
        [np.ones((4, 5)).astype(config.floatX), np.ones((4, 5)).astype(config.floatX)],
    )


def test_mlx_Reshape_constant():
    a = vector("a")
    x = reshape(a, (2, 2))
    compare_mlx_and_py([a], [x], [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])


def test_mlx_Reshape_various_shapes():
    """Test reshape with various different shapes to ensure robustness."""
    # 1D to 2D
    a = vector("a")
    x = reshape(a, (2, 3))
    compare_mlx_and_py([a], [x], [np.arange(6, dtype=config.floatX)])

    # 2D to 1D
    b = pt.matrix("b")
    y = reshape(b, (6,))
    compare_mlx_and_py([b], [y], [np.arange(6, dtype=config.floatX).reshape(2, 3)])

    # 2D to 3D
    c = pt.matrix("c")
    z = reshape(c, (2, 2, 3))
    compare_mlx_and_py([c], [z], [np.arange(12, dtype=config.floatX).reshape(4, 3)])

    # 3D to 2D
    d = pt.tensor3("d")
    w = reshape(d, (3, 4))
    compare_mlx_and_py([d], [w], [np.arange(12, dtype=config.floatX).reshape(2, 2, 3)])


def test_mlx_Reshape_negative_one():
    """Test reshape with -1 dimension (infer dimension)."""
    a = vector("a")
    # Use -1 to infer the second dimension
    x = reshape(a, (2, -1))
    compare_mlx_and_py([a], [x], [np.arange(8, dtype=config.floatX)])

    # Use -1 to infer the first dimension
    y = reshape(a, (-1, 4))
    compare_mlx_and_py([a], [y], [np.arange(8, dtype=config.floatX)])


def test_mlx_Reshape_concrete_shape():
    """MLX should compile when a concrete value is passed for the `shape` parameter."""
    a = vector("a")
    x = reshape(a, a.shape)
    compare_mlx_and_py([a], [x], [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])

    x = reshape(a, (a.shape[0] // 2, a.shape[0] // 2))
    compare_mlx_and_py([a], [x], [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])


@pytest.mark.xfail(reason="`shape_pt` should be specified as a static argument")
def test_mlx_Reshape_shape_graph_input():
    a = vector("a")
    shape_pt = iscalar("b")
    x = reshape(a, (shape_pt, shape_pt))
    compare_mlx_and_py(
        [a, shape_pt], [x], [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX), 2]
    )


@pytest.mark.xfail(reason="ViewOp Op is not supported yet")
def test_mlx_compile_ops():
    x = DeepCopyOp()(pt.as_tensor_variable(1.1))
    compare_mlx_and_py([], [x], [])

    x_np = np.zeros((20, 1, 1))
    x = ViewOp()(pt.as_tensor_variable(x_np))

    compare_mlx_and_py([], [x], [])
