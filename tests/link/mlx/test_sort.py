import numpy as np
import pytest

from pytensor.tensor.sort import argsort, sort
from pytensor.tensor.type import matrix, tensor3
from tests.link.mlx.test_basic import compare_mlx_and_py


@pytest.mark.parametrize("axis", [None, 0, -1, -2])
@pytest.mark.parametrize("func", (sort, argsort))
def test_sort(func, axis):
    x = tensor3("x", shape=(2, 3, 2), dtype="float64")
    out = func(x, axis=axis)
    arr = np.random.default_rng(0).permutation(np.arange(12.0)).reshape(2, 3, 2)
    _, res = compare_mlx_and_py([x], [out], [arr])
    assert np.asarray(res).dtype == out.dtype


def test_sort_invalid_kind_warning():
    x = matrix("x", shape=(2, 2), dtype="float64")
    z = sort(x, axis=-1, kind="mergesort")
    with pytest.warns(UserWarning, match="MLX sort does not support the kind argument"):
        z.eval({x: np.array([[3.0, 1.0], [2.0, 4.0]])}, mode="MLX")
