from functools import partial

import numpy as np
import pytest
from scipy.signal import convolve as scipy_convolve
from scipy.signal import convolve2d as scipy_convolve2d

from pytensor import config, function, grad
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.traversal import ancestors, io_toposort
from pytensor.tensor import matrix, tensor, vector
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.signal.conv import Convolve1d, convolve1d, convolve2d
from tests import unittest_tools as utt


@pytest.mark.parametrize("kernel_shape", [3, 5, 8], ids=lambda x: f"kernel_shape={x}")
@pytest.mark.parametrize("data_shape", [3, 5, 8], ids=lambda x: f"data_shape={x}")
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_convolve1d(mode, data_shape, kernel_shape):
    data = vector("data")
    kernel = vector("kernel")
    op = partial(convolve1d, mode=mode)

    rng = np.random.default_rng((26, kernel_shape, data_shape, sum(map(ord, mode))))
    data_val = rng.normal(size=data_shape).astype(data.dtype)
    kernel_val = rng.normal(size=kernel_shape).astype(kernel.dtype)

    fn = function([data, kernel], op(data, kernel))
    np.testing.assert_allclose(
        fn(data_val, kernel_val),
        scipy_convolve(data_val, kernel_val, mode=mode),
        rtol=1e-6 if config.floatX == "float32" else 1e-15,
    )
    utt.verify_grad(op=lambda x: op(x, kernel_val), pt=[data_val])


def test_convolve1d_batch():
    x = matrix("data")
    y = matrix("kernel")
    out = convolve1d(x, y)

    rng = np.random.default_rng(38)
    x_test = rng.normal(size=(2, 8)).astype(x.dtype)
    y_test = x_test[::-1]

    res = out.eval({x: x_test, y: y_test})
    # Second entry of x, y are just y, x respectively,
    # so res[0] and res[1] should be identical.
    rtol = 1e-6 if config.floatX == "float32" else 1e-15
    res_np = np.convolve(x_test[0], y_test[0])
    np.testing.assert_allclose(res[0], res_np, rtol=rtol)
    np.testing.assert_allclose(res[1], res_np, rtol=rtol)


def test_convolve1d_batch_same():
    x = matrix("data")
    y = matrix("kernel")
    out = convolve1d(x, y, mode="same")

    rng = np.random.default_rng(38)
    x_test = rng.normal(size=(2, 8)).astype(x.dtype)
    y_test = rng.normal(size=(2, 8)).astype(x.dtype)

    res = out.eval({x: x_test, y: y_test})
    assert res.shape == (2, 8)


@pytest.mark.parametrize("mode", ("full", "valid", "same"))
def test_convolve1d_batch_graph(mode):
    """Test that we don't have slow Blockwise Subtensors in graph of a batched convolve1d"""
    x = matrix("x")
    y = matrix("y")
    out = convolve1d(x, y, mode=mode)
    grads = grad(out.sum(), wrt=[x, y])
    final_grads = rewrite_graph(
        grads, include=("ShapeOpt", "canonicalize", "stabilize", "specialize")
    )

    blockwise_nodes = [
        var.owner
        for var in ancestors(final_grads)
        if var.owner is not None and isinstance(var.owner.op, Blockwise)
    ]
    # Check any Blockwise are just Conv1d
    assert all(isinstance(node.op.core_op, Convolve1d) for node in blockwise_nodes)


@pytest.mark.parametrize("static_shape", [False, True])
def test_convolve1d_valid_grad(static_shape):
    """Test we don't do a full convolve in the gradient of the smaller input to a valid convolve."""
    larger = vector("larger", shape=(128 if static_shape else None,))
    smaller = vector("smaller", shape=(64 if static_shape else None,))
    out = convolve1d(larger, smaller, mode="valid")
    grad_out = rewrite_graph(
        grad(out.sum(), wrt=smaller),
        include=(
            "ShapeOpt",
            "canonicalize",
            "stabilize",
            "local_useless_unbatched_blockwise",
        ),
    )
    [conv_node] = [
        node
        for node in io_toposort([larger, smaller], [grad_out])
        if isinstance(node.op, Convolve1d)
    ]
    full_mode = conv_node.inputs[-1]
    # If shape is static we get constant mode == "valid", otherwise it depends on the input shapes
    # ignoring E712 because np.True_ and np.False_ need to be compared with `==` to produce a valid boolean
    if static_shape:
        assert full_mode.eval() == False  # noqa: E712
    else:
        dtype = larger.dtype
        larger_test = np.zeros((128,), dtype=dtype)
        smaller_test = np.zeros((64,), dtype=dtype)
        assert full_mode.eval({larger: larger_test, smaller: smaller_test}) == False  # noqa: E712
        assert full_mode.eval({larger: smaller_test, smaller: larger_test}) == True  # noqa: E712


def convolve1d_grad_benchmarker(convolve_mode, mode, benchmark):
    # Use None core shape so PyTensor doesn't know which mode to use until runtime.
    larger = tensor("larger", shape=(8, None))
    smaller = tensor("smaller", shape=(8, None))
    grad_wrt_smaller = grad(
        convolve1d(larger, smaller, mode=convolve_mode).sum(), wrt=smaller
    )

    fn = function([larger, smaller], grad_wrt_smaller, trust_input=True, mode=mode)

    rng = np.random.default_rng([119, mode == "full"])
    test_larger = rng.normal(size=(8, 1024)).astype(larger.type.dtype)
    test_smaller = rng.normal(size=(8, 16)).astype(smaller.type.dtype)
    benchmark(fn, test_larger, test_smaller)


@pytest.mark.parametrize("convolve_mode", ["full", "valid"])
def test_convolve1d_grad_benchmark_c(convolve_mode, benchmark):
    convolve1d_grad_benchmarker(convolve_mode, "FAST_RUN", benchmark)


@pytest.mark.parametrize(
    "kernel_shape", [(3, 3), (5, 3), (5, 8)], ids=lambda x: f"kernel_shape={x}"
)
@pytest.mark.parametrize(
    "data_shape", [(3, 3), (5, 5), (8, 8)], ids=lambda x: f"data_shape={x}"
)
@pytest.mark.parametrize("mode", ["full", "valid", "same"])
@pytest.mark.parametrize("boundary", ["fill", "wrap", "symm"])
def test_convolve2d(kernel_shape, data_shape, mode, boundary):
    data = matrix("data")
    kernel = matrix("kernel")
    op = partial(convolve2d, mode=mode, boundary=boundary, fillvalue=0)

    rng = np.random.default_rng((26, kernel_shape, data_shape, sum(map(ord, mode))))
    data_val = rng.normal(size=data_shape).astype(data.dtype)
    kernel_val = rng.normal(size=kernel_shape).astype(kernel.dtype)

    fn = function([data, kernel], op(data, kernel))
    np.testing.assert_allclose(
        fn(data_val, kernel_val),
        scipy_convolve2d(
            data_val, kernel_val, mode=mode, boundary=boundary, fillvalue=0
        ),
        atol=1e-6 if config.floatX == "float32" else 1e-8,
    )

    utt.verify_grad(lambda k: op(data_val, k).sum(), [kernel_val], eps=1e-4)


# @pytest.mark.parametrize(
#     "data_shape, kernel_shape", [[(10, 1, 8, 8), (3, 1, 3, 3)], # 8x8 grayscale
#                                  [(1000, 1, 8, 8), (3, 1, 1, 3)], # same, but with 1000 images
#                                  [(10, 3, 64, 64), (10, 3, 8, 8)], # 64x64 RGB
#                                  [(1000, 3, 64, 64), (10, 3, 8, 8)], # same, but with 1000 images
#                                  [(3, 100, 100, 100), (250, 100, 50, 50)]], # Very large, deep hidden layer or something
#
#     ids=lambda x: f"data_shape={x[0]}, kernel_shape={x[1]}"
# )
# @pytest.mark.parametrize('func', ['new', 'theano'], ids=['new-impl', 'theano-impl'])
# def test_conv2d_nn_benchmark(data_shape, kernel_shape, func, benchmark):
#     import pytensor.tensor as pt
#     x = pt.tensor("x", shape=data_shape)
#     y = pt.tensor("y", shape=kernel_shape)
#
#     if func == 'new':
#         out = nn_conv2d(x, y)
#     else:
#         out = conv2d(input=x, filters=y, border_mode="valid")
#
#     rng = np.random.default_rng(38)
#     x_test = rng.normal(size=data_shape).astype(x.dtype)
#     y_test = rng.normal(size=kernel_shape).astype(y.dtype)
#
#     fn = function([x, y], out, trust_input=True)
#
#     benchmark(fn, x_test, y_test)
