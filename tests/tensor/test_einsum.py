from functools import partial
from string import ascii_lowercase

import numpy as np
import pytest

import pytensor.tensor as pt
from pytensor import Mode, function
from pytensor.tensor.einsum import _delta, _general_dot, _iota, einsum


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


def test_general_dot():
    mode = Mode(linker="py", optimizer=None)
    rng = np.random.default_rng(45)

    signature = "(l0,a0,a1,l1),(a1,r0,r1,a0)->(l0,l1,r0,r1)"
    tensordot_axes = [(-3, -2), (-1, -4)]

    # X has two batch dims
    # Y has one batch dim
    x = pt.tensor("x", shape=(5, 4, 2, 11, 13, 3))
    y = pt.tensor("y", shape=(4, 13, 5, 7, 11))
    out = _general_dot((x, y), tensordot_axes, [(0, 1), (0,)])

    # FIXME: Not a satisfactory graph!
    # import pytensor
    # fn = pytensor.function([x, y], out)
    # print()
    # pytensor.dprint(fn, print_type=True)

    x_test = rng.normal(size=x.type.shape)
    y_test = rng.normal(size=y.type.shape)

    np_batched_tensordot = np.vectorize(
        partial(np.tensordot, axes=tensordot_axes), signature=signature
    )

    np.testing.assert_allclose(
        out.eval({x: x_test, y: y_test}, mode=mode),
        np_batched_tensordot(x_test, y_test),
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
def test_enisum_signatures(static_shape_known, signature):
    letters_to_dims = dict(zip("ijklmnop", [2, 3, 5, 7, 11, 13, 17, 19], strict=True))

    inputs = signature.split("->")[0].split(",")

    shapes = [tuple(letters_to_dims[letter] for letter in inp) for inp in inputs]
    if static_shape_known:
        static_shapes = shapes
    else:
        static_shapes = [[None] * len(shape) for shape in shapes]

    operands = [
        pt.tensor(name, shape=static_shape) for name, static_shape in zip(ascii_lowercase, static_shapes)
    ]
    out = pt.einsum(signature, *operands)
    assert out.owner.op.optimized == static_shape_known

    rng = np.random.default_rng(37)
    test_values = [rng.normal(size=shape) for shape in shapes]
    np_out = np.einsum(signature, *test_values)

    fn = function(operands, out)
    pt_out = fn(*test_values)

    # import pytensor
    # print(); pytensor.dprint(fn, print_type=True)

    # assert out.type.shape == np_out.shape  # Reshape operations lose static shape
    np.testing.assert_allclose(pt_out, np_out)


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
    )
    weights = rng.random(size=(num_filters, channels, kernel_size, kernel_size))
    result = einsum(conv_signature, windowed_input, weights).eval()

    assert result.shape == (32, 15, 8, 8)
    np.testing.assert_allclose(
        result,
        np.einsum("bchwkt,fckt->bfhw", windowed_input, weights),
    )


def test_ellipsis():
    rng = np.random.default_rng(159)
    x = pt.tensor("x", shape=(3, 5, 7, 11))
    y = pt.tensor("y", shape=(3, 5, 11, 13))
    x_test = rng.normal(size=x.type.shape)
    y_test = rng.normal(size=y.type.shape)
    expected_out = np.matmul(x_test, y_test)

    with pytest.raises(ValueError):
        pt.einsum("mp,pn->mn", x, y)

    out = pt.einsum("...mp,...pn->...mn", x, y)
    np.testing.assert_allclose(out.eval({x: x_test, y: y_test}), expected_out)

    # Put batch axes in the middle
    new_x = pt.moveaxis(x, -2, 0)
    new_y = pt.moveaxis(y, -2, 0)
    out = pt.einsum("m...p,p...n->m...n", new_x, new_y)
    np.testing.assert_allclose(
        out.eval({x: x_test, y: y_test}), expected_out.transpose(-2, 0, 1, -1)
    )

    out = pt.einsum("m...p,p...n->mn", new_x, new_y)
    np.testing.assert_allclose(
        out.eval({x: x_test, y: y_test}), expected_out.sum((0, 1))
    )
