import numpy as np
import pytest

from pytensor.tensor.sort import argsort, sort
from pytensor.tensor.type import iscalar, matrix, tensor3
from tests.link.mlx.test_basic import compare_mlx_and_py, mlx_mode_no_compile


@pytest.mark.parametrize("axis", [None, 0, -1, -2])
@pytest.mark.parametrize("func", (sort, argsort))
def test_sort(func, axis):
    x = tensor3("x", shape=(2, 3, 2), dtype="float64")
    out = func(x, axis=axis)
    arr = np.random.default_rng(0).permutation(np.arange(12.0)).reshape(2, 3, 2)
    compare_mlx_and_py([x], [out], [arr])


@pytest.mark.parametrize("func", (sort, argsort))
def test_sort_symbolic_axis(func):
    x = matrix("x", shape=(2, 3), dtype="float64")
    axis = iscalar("axis")
    out = func(x, axis=axis)
    arr = np.random.default_rng(0).permutation(np.arange(6.0)).reshape(2, 3)
    compare_mlx_and_py([x, axis], [out], [arr, 1], mlx_mode=mlx_mode_no_compile)


def test_sort_invalid_kind_warning():
    x = matrix("x", shape=(2, 2), dtype="float64")
    z = sort(x, axis=-1, kind="mergesort")
    with pytest.warns(UserWarning, match="MLX sort does not support the kind argument"):
        z.eval({x: np.array([[3.0, 1.0], [2.0, 4.0]])}, mode="MLX")
