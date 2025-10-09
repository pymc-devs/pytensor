import contextlib

import numpy as np
import pytest

from pytensor import tensor as pt
from pytensor.tensor.sort import ArgSortOp, SortOp
from tests.link.numba.test_basic import compare_numba_and_py


@pytest.mark.parametrize(
    "x",
    [
        [],  # Empty list
        [3, 2, 1],  # Simple list
        np.random.randint(0, 10, (3, 2, 3, 4, 4)),  # Multi-dimensional array
    ],
)
@pytest.mark.parametrize("axis", [0, -1, None])
@pytest.mark.parametrize(
    ("kind", "exc"),
    [
        ["quicksort", None],
        ["mergesort", UserWarning],
        ["heapsort", UserWarning],
        ["stable", UserWarning],
    ],
)
def test_Sort(x, axis, kind, exc):
    if axis:
        g = SortOp(kind)(pt.as_tensor_variable(x), axis)
    else:
        g = SortOp(kind)(pt.as_tensor_variable(x))

    cm = contextlib.suppress() if not exc else pytest.warns(exc)

    with cm:
        compare_numba_and_py([], [g], [])


@pytest.mark.parametrize(
    "x",
    [
        [],  # Empty list
        [3, 2, 1],  # Simple list
        None,  # Multi-dimensional array (see below)
    ],
)
@pytest.mark.parametrize("axis", [0, -1, None])
@pytest.mark.parametrize(
    ("kind", "exc"),
    [
        ["quicksort", None],
        ["heapsort", None],
        ["stable", UserWarning],
    ],
)
def test_ArgSort(x, axis, kind, exc):
    if x is None:
        x = np.arange(5 * 5 * 5 * 5)
        np.random.shuffle(x)
        x = np.reshape(x, (5, 5, 5, 5))

    if axis:
        g = ArgSortOp(kind)(pt.as_tensor_variable(x), axis)
    else:
        g = ArgSortOp(kind)(pt.as_tensor_variable(x))

    cm = contextlib.suppress() if not exc else pytest.warns(exc)

    with cm:
        compare_numba_and_py([], [g], [])
