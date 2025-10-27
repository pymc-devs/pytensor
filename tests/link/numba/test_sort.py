import contextlib

import numpy as np
import pytest

from pytensor import tensor as pt
from pytensor.tensor.sort import ArgSortOp, SortOp
from tests.link.numba.test_basic import compare_numba_and_py


@pytest.mark.parametrize(
    "x_test",
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
def test_Sort(x_test, axis, kind, exc):
    x = pt.as_tensor(x_test).type("x")
    if axis:
        g = SortOp(kind)(x, axis)
    else:
        g = SortOp(kind)(x)

    cm = contextlib.suppress() if not exc else pytest.warns(exc)

    with cm:
        compare_numba_and_py([x], [g], [x_test])


@pytest.mark.parametrize(
    "x_test",
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
def test_ArgSort(x_test, axis, kind, exc):
    if x_test is None:
        x_test = np.arange(5 * 5 * 5 * 5)
        np.random.shuffle(x_test)
        x_test = np.reshape(x_test, (5, 5, 5, 5))
    x = pt.as_tensor(x_test).type("x")

    if axis:
        g = ArgSortOp(kind)(x, axis)
    else:
        g = ArgSortOp(kind)(x)

    cm = contextlib.suppress() if not exc else pytest.warns(exc)

    with cm:
        compare_numba_and_py([x], [g], [x_test])
