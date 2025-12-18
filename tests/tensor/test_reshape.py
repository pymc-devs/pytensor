import numpy as np
import pytest

from pytensor import config, function
from pytensor import tensor as pt
from pytensor.graph import vectorize_graph
from pytensor.tensor.reshape import (
    join_dims,
    split_dims,
)


def test_join_dims():
    rng = np.random.default_rng()

    x = pt.tensor("x", shape=(2, 3, 4, 5))
    assert join_dims(x, axis=(0, 1)).type.shape == (6, 4, 5)
    assert join_dims(x, axis=(1, 2)).type.shape == (2, 12, 5)
    assert join_dims(x, axis=(-1, -2)).type.shape == (2, 3, 20)

    assert join_dims(x, axis=()).type.shape == (2, 3, 4, 5)
    assert join_dims(x, axis=(2,)).type.shape == (2, 3, 4, 5)

    with pytest.raises(
        ValueError,
        match=r"join_dims axis must be consecutive, got normalized axis: \(0, 2\)",
    ):
        _ = join_dims(x, axis=(0, 2)).type.shape == (8, 3, 5)

    x_joined = join_dims(x, axis=(1, 2))
    x_value = rng.normal(size=(2, 3, 4, 5)).astype(config.floatX)

    fn = function([x], x_joined, mode="FAST_COMPILE")

    x_joined_value = fn(x_value)
    np.testing.assert_allclose(x_joined_value, x_value.reshape(2, 12, 5))

    assert join_dims(x, axis=(1,)).eval({x: x_value}).shape == (2, 3, 4, 5)
    assert join_dims(x, axis=()).eval({x: x_value}).shape == (2, 3, 4, 5)

    x = pt.tensor("x", shape=(3, 5))
    x_joined = join_dims(x, axis=(0, 1))
    x_batched = pt.tensor("x_batched", shape=(10, 3, 5))
    x_joined_batched = vectorize_graph(x_joined, {x: x_batched})

    assert x_joined_batched.type.shape == (10, 15)

    x_batched_val = rng.normal(size=(10, 3, 5)).astype(config.floatX)
    assert x_joined_batched.eval({x_batched: x_batched_val}).shape == (10, 15)


@pytest.mark.parametrize(
    "axis, shape, expected_shape",
    [
        (0, pt.as_tensor([2, 3]), (2, 3, 4, 6)),
        (2, [2, 3], (6, 4, 2, 3)),
        (-1, 6, (6, 4, 6)),
    ],
    ids=["tensor", "list", "integer"],
)
def test_split_dims(axis, shape, expected_shape):
    rng = np.random.default_rng()

    x = pt.tensor("x", shape=(6, 4, 6))
    x_split = split_dims(x, axis=axis, shape=shape)
    assert x_split.type.shape == expected_shape

    x_split = split_dims(x, axis=axis, shape=shape)
    x_value = rng.normal(size=(6, 4, 6)).astype(config.floatX)

    fn = function([x], x_split, mode="FAST_COMPILE")

    x_split_value = fn(x_value)
    np.testing.assert_allclose(x_split_value, x_value.reshape(expected_shape))

    x = pt.tensor("x", shape=(10,))
    x_split = split_dims(x, shape=(5, 2), axis=0)
    x_batched = pt.tensor("x_batched", shape=(3, 10))
    x_split_batched = vectorize_graph(x_split, {x: x_batched})

    assert x_split_batched.type.shape == (3, 5, 2)

    x_batched_val = rng.normal(size=(3, 10)).astype(config.floatX)
    assert x_split_batched.eval({x_batched: x_batched_val}).shape == (3, 5, 2)


def test_split_size_zero_shape():
    x = pt.tensor("x", shape=(1, 4, 6))
    x_split = split_dims(x, axis=0, shape=pt.as_tensor(np.zeros((0,))))
    assert x_split.type.shape == (4, 6)

    x_value = np.empty((1, 4, 6), dtype=config.floatX)

    fn = function([x], x_split, mode="FAST_COMPILE")

    x_split_value = fn(x_value)
    np.testing.assert_allclose(x_split_value, x_value.squeeze(0))
