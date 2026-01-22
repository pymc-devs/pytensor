import numpy as np

import pytensor.tensor as pt
from pytensor.compile.ops import DeepCopyOp, ViewOp
from pytensor.configdefaults import config
from pytensor.tensor.shape import Shape, Shape_i, reshape
from pytensor.tensor.type import iscalar, vector
from tests.link.jax.test_basic import compare_jax_and_py


def test_jax_shape_ops():
    x_np = np.zeros((20, 3))
    x = Shape()(pt.as_tensor_variable(x_np))

    compare_jax_and_py([], [x], [], must_be_device_array=False)

    x = Shape_i(1)(pt.as_tensor_variable(x_np))

    compare_jax_and_py([], [x], [], must_be_device_array=False)


def test_jax_specify_shape():
    in_pt = pt.matrix("in")
    x = pt.specify_shape(in_pt, (4, None))
    compare_jax_and_py([in_pt], [x], [np.ones((4, 5)).astype(config.floatX)])

    # When used to assert two arrays have similar shapes
    in_pt = pt.matrix("in")
    shape_pt = pt.matrix("shape")
    x = pt.specify_shape(in_pt, shape_pt.shape)

    compare_jax_and_py(
        [in_pt, shape_pt],
        [x],
        [np.ones((4, 5)).astype(config.floatX), np.ones((4, 5)).astype(config.floatX)],
    )


def test_jax_Reshape_constant():
    a = vector("a")
    x = reshape(a, (2, 2))
    compare_jax_and_py([a], [x], [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])


def test_jax_Reshape_concrete_shape():
    """JAX should compile when a concrete value is passed for the `shape` parameter."""
    a = vector("a")
    x = reshape(a, a.shape)
    compare_jax_and_py([a], [x], [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])

    x = reshape(a, (a.shape[0] // 2, a.shape[0] // 2))
    compare_jax_and_py([a], [x], [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])


def test_jax_Reshape_shape_graph_input():
    a = vector("a")
    shape_pt = iscalar("b")
    x = reshape(a, (shape_pt, shape_pt))
    compare_jax_and_py(
        [a, shape_pt], [x], [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX), 2]
    )


def test_jax_compile_ops():
    x = DeepCopyOp()(pt.as_tensor_variable(1.1))
    compare_jax_and_py([], [x], [])

    x_np = np.zeros((20, 1, 1))
    x = ViewOp()(pt.as_tensor_variable(x_np))

    compare_jax_and_py([], [x], [])
