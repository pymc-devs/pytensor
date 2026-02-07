import numpy as np
import pytest


pytest.importorskip("xarray")

from pytensor import function
from pytensor.graph import vectorize_graph
from pytensor.tensor import matrix, vector
from pytensor.xtensor import as_xtensor, xtensor
from pytensor.xtensor.basic import rename, tensor_from_xtensor, xtensor_from_tensor
from tests.xtensor.util import check_vectorization, xr_random_like


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

    x_val = xr_random_like(x)
    x_batched_val = x_val.expand_dims({"batch": 2})
    x_batched = as_xtensor(x_batched_val).type("x_batched")

    [y_batched] = vectorize_graph([y], {x: x_batched})

    # y_batched should be a Matrix (batch, a) -> (2, 3)
    assert y_batched.type.shape == (2, 3)

    fn = function([x_batched], y_batched)
    res = fn(x_batched_val)

    np.testing.assert_allclose(res, x_batched_val.values)
