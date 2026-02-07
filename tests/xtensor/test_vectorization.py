import numpy as np
import pytest

from pytensor.tensor import TensorVariable, broadcast_to, tensor
from pytensor.xtensor.basic import xtensor_from_tensor
from pytensor.xtensor.type import XTensorType, as_xtensor, xtensor
from pytensor.xtensor.vectorization import vectorize_graph
from tests.unittest_tools import assert_equal_computations


class TestVectorizeGraph:
    def test_pure_xtensor_graph(self):
        x = xtensor("x", dims=("a",))
        out = x + 1

        x_new = xtensor("x_new", dims=("c", "a", "b"))
        [out_vec] = vectorize_graph([out], {x: x_new})

        assert isinstance(out_vec.type, XTensorType)
        assert out_vec.type.dims == ("c", "b", "a")
        expected = x_new.transpose("c", "b", "a") + 1
        assert_equal_computations([out_vec], [expected])

    def test_pure_tensor_graph(self):
        x = tensor("x", shape=())
        out = x + 1

        x_new = tensor("x_new", shape=(5,))
        [out_vec] = vectorize_graph([out], {x: x_new}, new_tensor_dims=["b"])

        assert isinstance(out_vec, TensorVariable)
        assert out_vec.ndim == 1
        expected = x_new + 1
        assert_equal_computations([out_vec], [expected])

    def test_intermediate_tensor_graph(self):
        x = xtensor("x", dims=("a",))
        t = x.values  # Convert to TensorVariable
        t2 = t + np.ones(1)
        out = xtensor_from_tensor(t2, dims=("a",))

        x_new = xtensor("x_new", dims=("a", "b"))
        [out_vec] = vectorize_graph([out], {x: x_new})

        assert isinstance(out_vec.type, XTensorType)
        assert out_vec.type.dims == ("b", "a")
        expected = as_xtensor(
            x_new.transpose("b", "a").values + np.ones(1), dims=("b", "a")
        )
        assert_equal_computations([out_vec], [expected])

    def test_intermediate_tensor_multiple_inputs_graph(self):
        x = xtensor("x", dims=("a",))
        y = xtensor("y", dims=("a",))
        t = x.values + y.values
        out = xtensor_from_tensor(t, dims=("a",))

        x_new = xtensor("x_new", dims=("a", "c"))

        # Both inputs have the same batch dims
        y_new = xtensor("y_new", dims=("c", "a"))
        [out_vec] = vectorize_graph([out], {x: x_new, y: y_new})

        assert isinstance(out_vec.type, XTensorType)
        assert out_vec.type.dims == ("c", "a")
        expected = as_xtensor(
            (x_new.transpose("c", "a").values + y_new.transpose("c", "a").values),
            dims=("c", "a"),
        )
        assert_equal_computations([out_vec], [expected])

        # Inputs have different batch dims
        y_new = xtensor("y_new", dims=("b", "a"))
        [out_vec] = vectorize_graph([out], {x: x_new, y: y_new})

        assert isinstance(out_vec.type, XTensorType)
        assert out_vec.type.dims == ("c", "b", "a")
        expected = as_xtensor(
            (
                x_new.transpose("c", "a").values[:, None]
                + y_new.transpose("b", "a").values[None, :]
            ),
            dims=("c", "b", "a"),
        )
        assert_equal_computations([out_vec], [expected])

    def test_intermediate_xtensor_graph(self):
        x = tensor("x", shape=(3,))
        t = as_xtensor(x, dims=("a",))
        t2 = t + 1
        out = t2.values

        x_new = tensor("x_new", shape=(5, 3))
        [out_vec] = vectorize_graph([out], {x: x_new}, new_tensor_dims=["b"])

        assert isinstance(out_vec, TensorVariable)
        assert out_vec.ndim == 2
        expected = (as_xtensor(x_new, dims=("b", "a")) + 1).values
        assert_equal_computations([out_vec], [expected])

    def test_mixed_type_inputs(self):
        x = xtensor("x", dims=("a",), shape=(3,))
        y = tensor("y", shape=(5,))

        out = as_xtensor(y[2:], dims=("b",)) + x

        x_new = xtensor("x_new", dims=("a", "d"), shape=(3, 7))
        y_new = tensor("y_new", shape=(7, 5))

        # New dimension of y is aligned with the new dimension of x
        [out_vec] = vectorize_graph([out], {x: x_new, y: y_new}, new_tensor_dims=["d"])
        assert isinstance(out_vec.type, XTensorType)
        assert out_vec.type.dims == ("d", "b", "a")
        expected = as_xtensor(y_new[:, 2:], dims=("d", "b")) + x_new.transpose("d", "a")
        assert_equal_computations([out_vec], [expected])

        # New dimension of y is distinct from that of x
        [out_vec] = vectorize_graph([out], {x: x_new, y: y_new}, new_tensor_dims=["c"])
        assert isinstance(out_vec.type, XTensorType)
        assert out_vec.type.dims == ("d", "c", "b", "a")
        # x introduced a new dimension "d" which causes y to be broadcasted
        y_broadcasted = broadcast_to(
            y_new, (x_new.sizes["d"], y_new.shape[0], y_new.shape[1])
        )
        expected = as_xtensor(
            y_broadcasted[:, :, 2:], dims=("d", "c", "b")
        ) + x_new.transpose("d", "a")
        assert_equal_computations([out_vec], [expected])

    def test_mixed_type_inputs_complex_broadcasting(self):
        a = xtensor("a", dims=("a",), shape=(3,))
        b = xtensor("b", dims=("b"), shape=(5,))
        y = tensor("y", shape=(7,))
        z = tensor("z", shape=(11,))

        out = a + b + y.sum() + z.sum()
        assert out.dims == ("a", "b")

        a_new = xtensor("a_new", dims=("a*", "a"), shape=(33, 3))
        b_new = xtensor("b_new", dims=("b*", "b"), shape=(55, 5))
        y_new = tensor("y_new", shape=(1, 55, 2, 1, 7))
        z_new = tensor("z_new", shape=(33, 1, 1, 2, 11))

        [out_vec] = vectorize_graph(
            [out],
            {a: a_new, b: b_new, y: y_new, z: z_new},
            new_tensor_dims=["a*", "b*", "y*", "z*"],
        )
        assert isinstance(out_vec.type, XTensorType)
        assert out_vec.type.dims == ("a*", "b*", "y*", "z*", "a", "b")
        batch_shape_truth = (
            a_new.sizes["a*"],
            b_new.sizes["b*"],
            y_new.shape[2],
            z_new.shape[3],
        )
        y_new_bcast = broadcast_to(y_new, (*batch_shape_truth, y_new.shape[4]))
        z_new_bcast = broadcast_to(z_new, (*batch_shape_truth, z_new.shape[4]))
        expected_out = (
            (a_new + b_new)
            + as_xtensor(y_new_bcast.sum(axis=-1), dims=("a*", "b*", "y*", "z*"))
            + as_xtensor(z_new_bcast.sum(axis=-1), dims=("a*", "b*", "y*", "z*"))
        ).transpose("a*", "b*", "y*", "z*", ...)
        assert_equal_computations([out_vec], [expected_out])

    def test_invalid_cases(self):
        x = xtensor("x", dims=("a",))
        out = x + 1

        # Missing xtensor dims
        x_bad = xtensor("x_bad", dims=("b",))  # Missing "a"
        with pytest.raises(ValueError, match="missing pre-existing dims"):
            vectorize_graph([out], {x: x_bad})

        # New xtensor dims that were present in original graph
        y = xtensor("y", dims=("b",))
        out2 = x + y
        x_new_conflict = xtensor("x_new", dims=("a", "b"))
        # "b" is new to x, but present in graph (in y)
        with pytest.raises(ValueError, match="new dimensions that were present"):
            vectorize_graph([out2], {x: x_new_conflict})

        # Missing tensor dims
        t = tensor("t", shape=(3,))
        out_t = t + 1
        # Replacement has fewer dims (rank 0)
        t_bad_rank = tensor("t_bad", shape=())
        with pytest.raises(ValueError, match="missing pre-existing dims"):
            vectorize_graph([out_t], {t: t_bad_rank})

        # Missing new_tensor_dims
        t_new = tensor("t_new", shape=(5, 5, 3))
        with pytest.raises(ValueError, match="You must specify `new_tensor_dims`"):
            vectorize_graph([out_t], {t: t_new})
        with pytest.raises(ValueError, match=r"but only .* were specified"):
            vectorize_graph([out_t], {t: t_new}, new_tensor_dims=["a"])

        # Excess new_tensor_dims
        # Replacement adds 1 dim, but 2 are specified
        t_new_1dim = tensor("t_new_1dim", shape=(5, 3))
        with pytest.raises(ValueError, match="tensor dims were specified, but only"):
            vectorize_graph([out_t], {t: t_new_1dim}, new_tensor_dims=["a", "b"])
