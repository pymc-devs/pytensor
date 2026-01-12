import numpy as np
import pytest

import pytensor
import tests.unittest_tools as utt
from pytensor import config, function
from pytensor import tensor as pt
from pytensor.graph import rewrite_graph, vectorize_graph
from pytensor.graph.op import io_connection_pattern
from pytensor.tensor.reshape import (
    _analyze_axes_list,
    join_dims,
    pack,
    split_dims,
    unpack,
)


def test_join_dims():
    rng = np.random.default_rng()

    x = pt.tensor("x", shape=(2, 3, 4, 5))
    assert join_dims(x).type.shape == (120,)
    assert join_dims(x, n_axes=1).type.shape == (2, 3, 4, 5)
    assert join_dims(x, n_axes=0).type.shape == (1, 2, 3, 4, 5)

    assert join_dims(x, n_axes=2).type.shape == (6, 4, 5)
    assert join_dims(x, start_axis=1, n_axes=2).type.shape == (2, 12, 5)
    assert join_dims(x, start_axis=-3, n_axes=2).type.shape == (2, 12, 5)
    assert join_dims(x, start_axis=2).type.shape == (2, 3, 20)

    with pytest.raises(
        IndexError,
        match=r"Axis 5 is out of bounds for array of dimension 4",
    ):
        join_dims(x, start_axis=5)

    with pytest.raises(
        ValueError,
        match=r"JoinDims was asked to join dimensions 0 to 5, but input x has only 4 dimensions.",
    ):
        join_dims(x, n_axes=5)

    x_value = rng.normal(size=(2, 3, 4, 5)).astype(config.floatX)
    np.testing.assert_allclose(
        join_dims(x, start_axis=1, n_axes=2).eval({x: x_value}),
        x_value.reshape(2, 12, 5),
    )
    assert join_dims(x, 1, n_axes=1).eval({x: x_value}).shape == (2, 3, 4, 5)
    assert join_dims(x, 1, n_axes=0).eval({x: x_value}).shape == (2, 1, 3, 4, 5)

    x = pt.tensor("x", shape=(3, 5))
    x_joined = join_dims(x)
    x_batched = pt.tensor("x_batched", shape=(10, 3, 5))
    x_joined_batched = vectorize_graph(x_joined, {x: x_batched})

    assert x_joined_batched.type.shape == (10, 15)

    x_batched_val = rng.normal(size=(10, 3, 5)).astype(config.floatX)
    assert x_joined_batched.eval({x_batched: x_batched_val}).shape == (10, 15)

    utt.verify_grad(lambda x: join_dims(x, start_axis=1, n_axes=2), [x_value])


@pytest.mark.parametrize(
    "axis, shape, expected_shape",
    [
        (0, pt.as_tensor([2, 3]), (2, 3, 4, 6)),
        (2, [2, 3], (6, 4, 2, 3)),
        (-1, pt.as_tensor(6), (6, 4, 6)),
        (-1, 6, (6, 4, 6)),
    ],
    ids=["tensor list", "integer list", "tensor", "integer"],
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

    utt.verify_grad(lambda x: split_dims(x, shape=shape, axis=axis), [x_value])

    x = pt.tensor("x", shape=(10,))
    x_split = split_dims(x, shape=(5, 2), axis=0)
    x_batched = pt.tensor("x_batched", shape=(3, 10))
    x_split_batched = vectorize_graph(x_split, {x: x_batched})

    assert x_split_batched.type.shape == (3, 5, 2)

    x_batched_val = rng.normal(size=(3, 10)).astype(config.floatX)
    assert x_split_batched.eval({x_batched: x_batched_val}).shape == (3, 5, 2)


def test_split_size_zero_shape():
    x = pt.tensor("x", shape=(1, 4, 6))
    x_split = split_dims(x, axis=0, shape=pt.as_tensor(np.zeros((0,), dtype="int32")))
    assert x_split.type.shape == (4, 6)

    x_value = np.empty((1, 4, 6), dtype=config.floatX)

    fn = function([x], x_split, mode="FAST_COMPILE")

    x_split_value = fn(x_value)
    np.testing.assert_allclose(x_split_value, x_value.squeeze(0))


def test_make_replacements_with_pack_unpack():
    rng = np.random.default_rng()

    x = pt.tensor("x", shape=())
    y = pt.tensor("y", shape=(5,))
    z = pt.tensor("z", shape=(3, 3))

    loss = (x + y.sum() + z.sum()) ** 2

    flat_packed, packed_shapes = pack(x, y, z)
    new_input = flat_packed.type()
    new_outputs = unpack(new_input, packed_shapes=packed_shapes)

    loss = pytensor.graph.graph_replace(loss, dict(zip([x, y, z], new_outputs)))
    rewrite_graph(loss, include=("ShapeOpt", "canonicalize"))

    fn = pytensor.function([new_input], loss, mode="FAST_COMPILE")

    input_vals = [
        rng.normal(size=(var.type.shape)).astype(config.floatX) for var in [x, y, z]
    ]
    flat_inputs = np.concatenate([input.ravel() for input in input_vals], axis=0)
    output_val = fn(flat_inputs)

    assert np.allclose(output_val, sum([input.sum() for input in input_vals]) ** 2)


class TestPack:
    @pytest.mark.parametrize(
        "axes, expected",
        [
            (None, [0, 0, 0]),  # '*'
            ([0, 1], [2, 0, 2]),  # 'i j *'
            ([-1], [0, 1, 1]),  # '* k'
            ([-2, -1], [0, 2, 2]),  # '* i j'
            ([0, -1], [1, 1, 2]),  # 'i * k'
            ([0, 1, 2, -1], [3, 1, 4]),  # 'i j k * l'
        ],
        ids=[
            "ravel_all",
            "keep_first_two",
            "keep_last",
            "ravel_start",
            "first_and_last",
            "complex_case",
        ],
    )
    def test_analyze_axes_list_valid(self, axes, expected):
        outputs = _analyze_axes_list(axes)
        names = ["n_before", "n_after", "min_axes"]
        for out, exp, name in zip(outputs, expected, names, strict=True):
            assert out == exp, f"Expected {exp}, got {out} for {name}"

    def test_analyze_axes_list_invalid(self):
        # Positive only but not contiguous
        with pytest.raises(ValueError, match="Positive axes must be contiguous"):
            _analyze_axes_list([1, 3])

        # Negative only but not contiguous
        with pytest.raises(ValueError, match="Negative axes must be contiguous"):
            _analyze_axes_list([-3, -1])

        # Mixed up positive and negative
        with pytest.raises(ValueError, match="Negative axes must come after positive"):
            _analyze_axes_list([0, 1, -2, 4])

        # Duplicate axes
        with pytest.raises(ValueError, match="axes must have no duplicates"):
            _analyze_axes_list([0, 0])

        # Not monotonic
        with pytest.raises(ValueError, match="Axes must be strictly increasing"):
            _analyze_axes_list([0, 2, 1])

        # Negative before positive
        with pytest.raises(ValueError, match="Negative axes must come after positive"):
            _analyze_axes_list([-1, 0])

    def test_pack_basic(self):
        # rng = np.random.default_rng()
        x = pt.tensor("x", shape=())
        y = pt.tensor("y", shape=(5,))
        z = pt.tensor("z", shape=(3, 3))

        input_dict = {
            variable.name: np.zeros(variable.type.shape, dtype=config.floatX)
            for variable in [x, y, z]
        }

        # Simple case, reduce all axes, equivalent to einops '*'
        packed_tensor, packed_shapes = pack(x, y, z)
        assert packed_tensor.type.shape == (15,)
        for tensor, packed_shape in zip([x, y, z], packed_shapes):
            assert packed_shape.type.shape == (tensor.ndim,)
            np.testing.assert_allclose(
                packed_shape.eval(input_dict, on_unused_input="ignore"),
                tensor.type.shape,
            )

        # To preserve an axis, all inputs need at least one dimension, and the preserved axis has to agree.
        # x is scalar, so pack will raise:
        with pytest.raises(
            ValueError,
            match=r"Input 0 \(zero indexed\) to pack has 0 dimensions, but keep_axes=0 assumes at least 1 dimension\.",
        ):
            pack(x, y, z, keep_axes=0)

        # With valid x, pack should still raise, because the axis of concatenation doesn't agree across all inputs
        x = pt.tensor("x", shape=(3,))
        input_dict["x"] = np.zeros((3,), dtype=config.floatX)

        with pytest.raises(
            ValueError,
            match=r"all input array dimensions other than the specified `axis` \(1\) must match exactly, or be unknown "
            r"\(None\), but along dimension 0, the inputs shapes are incompatible: \[3 5 3\]",
        ):
            packed_tensor, packed_shapes = pack(x, y, z, keep_axes=0)
            packed_tensor.eval(input_dict)

        # Valid case, preserve first axis, equivalent to einops 'i *'
        y = pt.tensor("y", shape=(3, 5))
        z = pt.tensor("z", shape=(3, 3, 3))
        packed_tensor, packed_shapes = pack(x, y, z, keep_axes=0)
        input_dict = {
            variable.name: np.zeros(variable.type.shape, dtype=config.floatX)
            for variable in [x, y, z]
        }
        assert packed_tensor.type.shape == (3, 15)
        for tensor, packed_shape in zip([x, y, z], packed_shapes):
            assert packed_shape.type.shape == (tensor.ndim - 1,)
            np.testing.assert_allclose(
                packed_shape.eval(input_dict, on_unused_input="ignore"),
                tensor.type.shape[1:],
            )

        # More complex case, preserve last axis implicitly, equivalent to einops 'i * k'. This introduces a max
        # dimension condition on the input shapes
        x = pt.tensor("x", shape=(3, 2))
        y = pt.tensor("y", shape=(3, 5, 2))
        z = pt.tensor("z", shape=(3, 1, 7, 5, 2))

        with pytest.raises(
            ValueError,
            match=r"Positive axes must be contiguous",
        ):
            pack(x, y, z, keep_axes=[0, 3])

        z = pt.tensor("z", shape=(3, 1, 7, 2))
        packed_tensor, packed_shapes = pack(x, y, z, keep_axes=[0, -1])
        input_dict = {
            variable.name: np.zeros(variable.type.shape, dtype=config.floatX)
            for variable in [x, y, z]
        }
        assert packed_tensor.type.shape == (3, 13, 2)
        for tensor, packed_shape in zip([x, y, z], packed_shapes):
            assert packed_shape.type.shape == (tensor.ndim - 2,)
            np.testing.assert_allclose(
                packed_shape.eval(input_dict, on_unused_input="ignore"),
                tensor.type.shape[1:-1],
            )

    @pytest.mark.parametrize("axes", [-1])
    def test_pack_unpack_round_trip(self, axes):
        rng = np.random.default_rng()

        x = pt.tensor("x", shape=(3, 5))
        y = pt.tensor("y", shape=(3, 3, 5))
        z = pt.tensor("z", shape=(1, 3, 5))

        flat_packed, packed_shapes = pack(x, y, z, keep_axes=axes)
        new_outputs = unpack(flat_packed, packed_shapes=packed_shapes, keep_axes=axes)

        fn = pytensor.function([x, y, z], new_outputs, mode="FAST_COMPILE")

        input_dict = {
            var.name: rng.normal(size=var.type.shape).astype(config.floatX)
            for var in [x, y, z]
        }
        output_vals = fn(**input_dict)

        for input_val, output_val in zip(input_dict.values(), output_vals, strict=True):
            np.testing.assert_allclose(input_val, output_val, strict=True)

    def test_single_input(self):
        x = pt.matrix("x", shape=(2, 5))
        packed_x, packed_shapes = pt.pack(x)
        assert packed_x.type.shape == (10,)
        [x_again] = unpack(packed_x, packed_shapes)
        assert x_again.type.shape == (2, 5)

    def test_unpack_connection(self):
        x = pt.vector("x")
        d0 = pt.scalar("d0", dtype=int)
        d1 = pt.scalar("d1", dtype=int)
        x0, x1 = pt.unpack(x, packed_shapes=[d0, d1])
        out = x0.sum() + x1.sum()
        assert io_connection_pattern([x, d0, d1], [out]) == [[True], [False], [False]]
