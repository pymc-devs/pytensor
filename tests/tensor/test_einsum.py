from functools import partial
from string import ascii_lowercase

import numpy as np
import pytest

import pytensor
import pytensor.tensor as pt
from pytensor import Mode, config, function
from pytensor.graph import FunctionGraph
from pytensor.graph.op import HasInnerGraph
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.einsum import _delta, _general_dot, _iota, einsum
from pytensor.tensor.shape import Reshape


# Fail for unexpected warnings in this file
pytestmark = pytest.mark.filterwarnings("error")

floatX = pytensor.config.floatX
ATOL = RTOL = 1e-8 if floatX == "float64" else 1e-4


def assert_no_blockwise_in_graph(fgraph: FunctionGraph, core_op=None) -> None:
    for node in fgraph.apply_nodes:
        if isinstance(node.op, Blockwise):
            if core_op is None:
                raise AssertionError
            assert not isinstance(node.op.core_op, core_op)

        if isinstance(node.op, HasInnerGraph):
            # InnerGraph Ops can be rewritten without modifying the original fgraph
            if hasattr(node.op, "_fn"):
                inner_fgraph = node.op._fn.maker.fgraph
            else:
                inner_fgraph = node.op.fgraph
            assert_no_blockwise_in_graph(inner_fgraph, core_op=core_op)


def test_iota():
    mode = Mode(linker="py", optimizer=None)
    np.testing.assert_allclose(
        _iota((4, 8), 0).eval(mode=mode),
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3],
        ],
    )

    np.testing.assert_allclose(
        _iota((4, 8), 1).eval(mode=mode),
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
    )


def test_delta():
    mode = Mode(linker="py", optimizer=None)
    np.testing.assert_allclose(
        _delta((2, 2), (0, 1)).eval(mode=mode),
        [[1.0, 0.0], [0.0, 1.0]],
    )

    np.testing.assert_allclose(
        _delta((2, 2, 2), (0, 1)).eval(mode=mode),
        [[[1, 1], [0, 0]], [[0, 0], [1, 1]]],
    )


def test_general_dot():
    rng = np.random.default_rng(45)
    signature = "(l0,a0,a1,l1),(a1,r0,r1,a0)->(l0,l1,r0,r1)"
    tensordot_axes = [(-3, -2), (-1, -4)]

    # X has two batch dims
    # Y has one batch dim
    x = pt.tensor("x", shape=(5, 4, 2, 11, 13, 3))
    y = pt.tensor("y", shape=(4, 13, 5, 7, 11))
    out = _general_dot((x, y), tensordot_axes, [(0, 1), (0,)])

    fn = pytensor.function([x, y], out)
    # fn.dprint(print_type=True)
    if config.mode != "FAST_COMPILE":
        assert_no_blockwise_in_graph(fn.maker.fgraph, Reshape)

    np_batched_tensordot = np.vectorize(
        partial(np.tensordot, axes=tensordot_axes), signature=signature
    )
    x_test = rng.normal(size=x.type.shape).astype(floatX)
    y_test = rng.normal(size=y.type.shape).astype(floatX)
    np.testing.assert_allclose(
        fn(x_test, y_test), np_batched_tensordot(x_test, y_test), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("static_shape_known", [True, False])
@pytest.mark.parametrize(
    "signature",
    [
        "ij",
        "ji",
        "ii->i",
        "ii",
        "ij->",
        "ij->j",
        "ij->i",
        "ij,ij->ij",
        "ij,ji->ij",
        "ij,ji->ji",
        "ij,jk",
        "kj,ji",
        "ij,kj->ik",
        "ik,kj->ikj",
        "ij,kl->ijkl",
        "ij,jk,kl->il",
        "kl,ij,jk->il",
        "oij,imj,mjkn,lnk,plk->op",
    ],
)
def test_einsum_signatures(static_shape_known, signature):
    letters_to_dims = dict(zip("ijklmnop", [2, 3, 5, 7, 11, 13, 17, 19], strict=True))

    inputs = signature.split("->")[0].split(",")

    shapes = [tuple(letters_to_dims[letter] for letter in inp) for inp in inputs]
    if static_shape_known:
        static_shapes = shapes
    else:
        static_shapes = [[None] * len(shape) for shape in shapes]

    operands = [
        pt.tensor(name, shape=static_shape)
        for name, static_shape in zip(ascii_lowercase, static_shapes)
    ]
    out = pt.einsum(signature, *operands)
    assert out.owner.op.optimized == static_shape_known or len(operands) <= 2

    rng = np.random.default_rng(37)
    test_values = [rng.normal(size=shape).astype(floatX) for shape in shapes]
    np_out = np.einsum(signature, *test_values)

    fn = function(operands, out)
    pt_out = fn(*test_values)

    # print(); fn.dprint(print_type=True)

    if config.mode != "FAST_COMPILE":
        assert_no_blockwise_in_graph(fn.maker.fgraph)
    np.testing.assert_allclose(pt_out, np_out, atol=ATOL, rtol=RTOL)


def test_batch_dim():
    shapes = (
        (7, 3, 5),
        (5, 2),
    )
    x, y = (pt.tensor(name, shape=shape) for name, shape in zip("xy", shapes))
    out = pt.einsum("mij,jk->mik", x, y)

    assert out.type.shape == (7, 3, 2)


def test_einsum_conv():
    # Adapted example from https://medium.com/latinxinai/vectorized-convolution-operation-using-numpy-b122fd52fba3
    rng = np.random.default_rng(125)
    batch_size = 32
    channels = 3
    height = 8
    width = 8
    kernel_size = 2
    num_filters = 15
    conv_signature = "bchwkt,fckt->bfhw"
    windowed_input = rng.random(
        size=(batch_size, channels, height, width, kernel_size, kernel_size)
    ).astype(floatX)
    weights = rng.random(size=(num_filters, channels, kernel_size, kernel_size)).astype(
        floatX
    )
    result = einsum(conv_signature, windowed_input, weights).eval()

    assert result.shape == (32, 15, 8, 8)
    np.testing.assert_allclose(
        result,
        np.einsum("bchwkt,fckt->bfhw", windowed_input, weights),
        atol=ATOL,
        rtol=RTOL,
    )


def test_ellipsis():
    rng = np.random.default_rng(159)
    x = pt.tensor("x", shape=(3, 5, 7, 11))
    y = pt.tensor("y", shape=(3, 5, 11, 13))
    x_test = rng.normal(size=x.type.shape).astype(floatX)
    y_test = rng.normal(size=y.type.shape).astype(floatX)
    expected_out = np.matmul(x_test, y_test)

    with pytest.raises(ValueError):
        pt.einsum("mp,pn->mn", x, y)

    out = pt.einsum("...mp,...pn->...mn", x, y)
    np.testing.assert_allclose(
        out.eval({x: x_test, y: y_test}), expected_out, atol=ATOL, rtol=RTOL
    )

    # Put batch axes in the middle
    new_x = pt.moveaxis(x, -2, 0)
    new_y = pt.moveaxis(y, -2, 0)
    out = pt.einsum("m...p,p...n->m...n", new_x, new_y)
    np.testing.assert_allclose(
        out.eval({x: x_test, y: y_test}),
        expected_out.transpose(-2, 0, 1, -1),
        atol=ATOL,
        rtol=RTOL,
    )

    out = pt.einsum("m...p,p...n->mn", new_x, new_y)
    np.testing.assert_allclose(
        out.eval({x: x_test, y: y_test}), expected_out.sum((0, 1)), atol=ATOL, rtol=RTOL
    )


def test_broadcastable_dims():
    # Test that einsum handles broadcasting dims correctly. There are two points:
    # 1. Numpy einsum allows the same subscript for degenerate and full dimensions
    # There is some stale discussion on whether this should be a bug or not, but for now it is not:
    # https://github.com/numpy/numpy/issues/11548

    # 2. Using the same letter for dimensions that are and aren't broadcastable
    # can lead to suboptimal paths. We check we issue a warning for the following example:
    # https://github.com/dgasmith/opt_einsum/issues/220
    rng = np.random.default_rng(222)
    a = pt.tensor("a", shape=(32, 32, 32))
    b = pt.tensor("b", shape=(1000, 32))
    c = pt.tensor("c", shape=(1, 32))

    a_test = rng.normal(size=a.type.shape).astype(floatX)
    b_test = rng.normal(size=b.type.shape).astype(floatX)
    c_test = rng.normal(size=c.type.shape).astype(floatX)

    # Note b is used for both 1 and 32
    with pytest.warns(
        UserWarning, match="This can result in a suboptimal contraction path"
    ):
        suboptimal_out = pt.einsum("ijk,bj,bk->i", a, b, c)
    assert not [set(p) for p in suboptimal_out.owner.op.path] == [{0, 2}, {0, 1}]

    # If we use a distinct letter we get the optimal path
    optimal_out = pt.einsum("ijk,bj,ck->i", a, b, c)
    assert [set(p) for p in optimal_out.owner.op.path] == [{0, 2}, {0, 1}]

    suboptimal_eval = suboptimal_out.eval({a: a_test, b: b_test, c: c_test})
    optimal_eval = optimal_out.eval({a: a_test, b: b_test, c: c_test})
    np_eval = np.einsum("ijk,bj,bk->i", a_test, b_test, c_test)
    atol = 1e-12 if config.floatX == "float64" else 1e-2
    np.testing.assert_allclose(suboptimal_eval, np_eval, atol=atol)
    np.testing.assert_allclose(optimal_eval, np_eval, atol=atol)
