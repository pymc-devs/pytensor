import numpy as np
import pytest

from pytensor.tensor.sort import argsort, sort
from pytensor.tensor.type import matrix
from tests.link.mlx.test_basic import compare_mlx_and_py


@pytest.mark.parametrize("axis", [None, -1])
@pytest.mark.parametrize("func", (sort, argsort))
def test_sort(func, axis):
    x = matrix("x", shape=(2, 2), dtype="float64")
    out = func(x, axis=axis)
    arr = np.array([[1.0, 4.0], [5.0, 2.0]])
    compare_mlx_and_py([x], [out], [arr])


def test_sort_invalid_kind_warning():
    x = matrix("x", shape=(2, 2), dtype="float64")
    z = sort(x, axis=-1, kind="mergesort")
    with pytest.warns(UserWarning, match="MLX sort does not support the kind argument"):
        z.eval({x: np.array([[3.0, 1.0], [2.0, 4.0]])}, mode="MLX")
