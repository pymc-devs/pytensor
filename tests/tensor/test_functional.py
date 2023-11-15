import numpy as np
import pytest

from pytensor.graph.basic import equal_computations
from pytensor.tensor import full, tensor
from pytensor.tensor.functional import vectorize
from pytensor.tensor.random.type import RandomGeneratorType


class TestVectorize:
    def test_vectorize_no_signature(self):
        """Unlike numpy we don't assume outputs of vectorize without signature are scalar."""

        def func(x):
            return full((5, 3), x)

        vec_func = vectorize(func)

        x = tensor("x", shape=(4,), dtype="float64")
        out = vec_func(x)

        assert out.type.ndim == 3
        test_x = np.array([1, 2, 3, 4])
        np.testing.assert_allclose(
            out.eval({x: test_x}), np.full((len(test_x), 5, 3), test_x[:, None, None])
        )

    def test_vectorize_outer_product(self):
        def func(x, y):
            return x[:, None] * y[None, :]

        vec_func = vectorize(func, signature="(a),(b)->(a,b)")

        x = tensor("x", shape=(2, 3, 5))
        y = tensor("y", shape=(2, 3, 7))
        out = vec_func(x, y)

        assert out.type.shape == (2, 3, 5, 7)
        assert equal_computations([out], [x[..., :, None] * y[..., None, :]])

    def test_vectorize_outer_inner_product(self):
        def func(x, y):
            return x[:, None] * y[None, :], (x * y).sum()

        vec_func = vectorize(func, signature="(a),(b)->(a,b),()")

        x = tensor("x", shape=(2, 3, 5))
        y = tensor("y", shape=(2, 3, 5))
        outer, inner = vec_func(x, y)

        assert outer.type.shape == (2, 3, 5, 5)
        assert inner.type.shape == (2, 3)
        assert equal_computations([outer], [x[..., :, None] * y[..., None, :]])
        assert equal_computations([inner], [(x * y).sum(axis=-1)])

    def test_errors(self):
        def func(x, y):
            return x + y, x - y

        x = tensor("x", shape=(5,))
        y = tensor("y", shape=())

        with pytest.raises(ValueError, match="Number of inputs"):
            vectorize(func, signature="(),()->()")(x)

        with pytest.raises(ValueError, match="Number of outputs"):
            vectorize(func, signature="(),()->()")(x, y)

        with pytest.raises(ValueError, match="Input y has less dimensions"):
            vectorize(func, signature="(a),(a)->(a),(a)")(x, y)

        bad_input = RandomGeneratorType()

        with pytest.raises(TypeError, match="must be TensorVariable"):
            vectorize(func)(bad_input, x)

        def bad_func(x, y):
            x + y

        with pytest.raises(ValueError, match="no outputs"):
            vectorize(bad_func)(x, y)
