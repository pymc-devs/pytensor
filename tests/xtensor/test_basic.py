import pytest


pytest.importorskip("xarray")

import numpy as np

from pytensor import function
from pytensor.graph import vectorize_graph
from pytensor.tensor import matrix, vector
from pytensor.xtensor.basic import (
    Rename,
    rename,
    tensor_from_xtensor,
    xtensor_from_tensor,
)
from pytensor.xtensor.type import xtensor
from tests.unittest_tools import assert_equal_computations

# from pytensor.xtensor.vectorization import vectorize_graph
from tests.xtensor.util import check_vectorization


def test_shape_feature_does_not_see_xop():
    CALLED = False

    x = xtensor("x", dims=("a",), dtype="int64")

    class XOpWithBadInferShape(Rename):
        def infer_shape(self, node, inputs, outputs):
            global CALLED
            CALLED = True
            raise NotImplementedError()

    test_xop = XOpWithBadInferShape(new_dims=("b",))

    out = test_xop(x) - test_xop(x)
    assert out.dims == ("b",)

    fn = function([x], out)
    np.testing.assert_allclose(fn([1, 2, 3]), [0, 0, 0])
    assert not CALLED


def test_rename_vectorize():
    ab = xtensor("ab", dims=("a", "b"), shape=(2, 3), dtype="float64")
    check_vectorization(ab, rename(ab, a="c"))


def test_xtensor_from_tensor_vectorize():
    t = vector("t")
    x = xtensor_from_tensor(t, dims=("a",))

    t_batched = matrix("t_batched")
    with pytest.raises(
        NotImplementedError, match=r"Vectorization of .* not implemented"
    ):
        vectorize_graph([x], {t: t_batched})


def test_tensor_from_xtensor_vectorize():
    x = xtensor("x", dims=("a",), shape=(3,))
    y = tensor_from_xtensor(x)

    x_batched = xtensor("x", dims=("a", "b"), shape=(3, 5))

    y_batched = vectorize_graph(y, {x: x_batched})
    # vectorize_graph should place output batch dimension on the left
    assert y_batched.type.shape == (5, 3)
    assert_equal_computations([y_batched], [x_batched.transpose("b", ...).values])

    x_batched = xtensor("x", dims=("c", "a", "b"), shape=(7, 3, 5))
    # vectorize_graph can't handle multiple batch dimensions safely
    with pytest.raises(NotImplementedError):
        vectorize_graph(y, {x: x_batched})
